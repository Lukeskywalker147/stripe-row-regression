"""v3: two heads (row-softmax over x, column-softmax over y) so the net itself handles stripe orientation."""
import glob, os, sys
import numpy as np, cv2, torch
import segmentation_models_pytorch as smp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
synth3 = importlib.import_module("synth3f" if os.environ.get("SYNTH_FAST") else "synth3")
from core import ROOT, CH, CW, MEAN, STD, to_tensor, shear_crop, crop_with_labels, photometric, row_logits, decode, row_loss

S = 512  # square training crop


class FullRes(torch.nn.Module):
    """U-Net + a shallow full-resolution branch from the raw image fused after the last upsample."""
    def __init__(self, base):
        super().__init__()
        nn = torch.nn
        self.encoder, self.decoder = base.encoder, base.decoder
        c = lambda i, o: [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True)]
        self.shallow = nn.Sequential(*c(3, 16), *c(16, 16))
        self.fuse = nn.Sequential(*c(32, 32), nn.Conv2d(32, 4, 1))

    def forward(self, x):
        d = self.decoder(self.encoder(x))
        return self.fuse(torch.cat([d, self.shallow(x)], 1))


def build_model(encoder="resnet18", weights=None, arch="base"):
    m = smp.Unet(encoder, encoder_weights=weights, classes=4, decoder_channels=(128, 64, 32, 16, 16))
    return FullRes(m) if arch == "fullres" else m


def two_logits_seg(out):
    """segmentation-style head: pixel logits, constant 'absent' logit 0 so the shared softmax decoder still works"""
    r = out[:, 0]
    c = out[:, 2].transpose(1, 2)
    z = lambda t: torch.zeros_like(t[:, :, :1])
    return torch.cat([r, z(r)], 2), torch.cat([c, z(c)], 2)


def seg_loss(logit, lab, sigma=1.5, pos_w=20.0):
    """per-pixel BCE against a Gaussian heat-map target (classic centre-line segmentation head)"""
    B, H, W1 = logit.shape
    W = W1 - 1
    valid = ~torch.isnan(lab)
    if valid.sum() == 0:
        return logit.sum() * 0, {}
    lg = logit[valid][:, :W].float(); lb = lab[valid]
    xs = torch.arange(W, device=lg.device)[None, :].float()
    tgt = torch.exp(-(xs - lb[:, None]) ** 2 / (2 * sigma ** 2)) * (lb >= 0)[:, None]
    w = 1 + pos_w * tgt
    loss = (torch.nn.functional.binary_cross_entropy_with_logits(lg, tgt, reduction="none") * w).mean()
    return loss, dict(bce=float(loss))


def two_logits(out):
    """out (B,4,H,W) -> row logits (B,H,W+1), column logits (B,W,H+1)"""
    return row_logits(out[:, :2]), row_logits(out[:, 2:].transpose(2, 3))


def make_lc(lr, W):
    """Column labels implied by row labels: columns the stripe never crosses are absent (-1), the rest unsupervised."""
    pres = lr[lr >= 0]
    lc = np.full(W, -1, np.float32)
    if len(pres):
        lc[max(0, int(np.floor(pres.min())) - 4): int(np.ceil(pres.max())) + 5] = np.nan
    return lc


def _pm(v, f):
    return np.where(v >= 0, f(v), v).astype(np.float32)


def rot_ccw(img, lr, lc):
    H, W = img.shape[:2]
    return np.rot90(img), lc[::-1].copy(), _pm(lr, lambda v: W - 1 - v)


def flip_x(img, lr, lc):
    W = img.shape[1]
    return img[:, ::-1], _pm(lr, lambda v: W - 1 - v), lc[::-1].copy()


def flip_y(img, lr, lc):
    H = img.shape[0]
    return img[::-1], lr[::-1].copy(), _pm(lc, lambda v: H - 1 - v)


def square_crop(img, lr, lc, rng, jitter=0.3):
    H, W = img.shape[:2]
    if W > S:
        pres = lr[lr >= 0]
        cx = float(np.median(pres)) + rng.uniform(-jitter, jitter) * S if len(pres) else rng.uniform(S / 2, W - S / 2)
        x0 = int(np.clip(cx - S / 2, 0, W - S))
        lr2 = lr.copy(); pm = lr2 >= 0
        lr2[pm] -= x0; lr2[pm & ((lr2 < 2) | (lr2 > S - 3))] = np.nan
        return img[:, x0:x0 + S], lr2, lc[x0:x0 + S]
    if H > S:
        pres = lc[lc >= 0]
        cy = float(np.median(pres)) + rng.uniform(-jitter, jitter) * S if len(pres) else rng.uniform(S / 2, H - S / 2)
        y0 = int(np.clip(cy - S / 2, 0, H - S))
        lc2 = lc.copy(); pm = lc2 >= 0
        lc2[pm] -= y0; lc2[pm & ((lc2 < 2) | (lc2 > S - 3))] = np.nan
        return img[y0:y0 + S], lr[y0:y0 + S], lc2
    return img, lr, lc


class MixDS3(torch.utils.data.Dataset):
    def __init__(self, split="train", n=1000, p_synth=0.5, seed=0, p_rot=0.5, bg_dir="bg"):
        self.n, self.p_synth, self.seed, self.split, self.p_rot = n, p_synth, seed, split, p_rot
        D = os.path.dirname(os.path.abspath(__file__))
        d = np.load(os.path.join(D, f"labels/{split}.npz"))
        self.names, self.labels = d["names"], d["labels"]
        self.bgs = sorted(glob.glob(os.path.join(D, bg_dir, "*.jpg")))
        self.train = split == "train"

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng((self.seed, i, wi.id if wi else 0, int(torch.randint(0, 2 ** 30, (1,)))) if self.train else (self.seed, i))
        if rng.random() < self.p_synth and self.bgs:
            bg = cv2.imread(self.bgs[int(rng.integers(len(self.bgs)))])
            H, W = bg.shape[:2]
            y0 = int(rng.integers(0, H - CH + 1)); x0 = int(rng.integers(0, W - CW + 1))
            img, lr = synth3.render(np.ascontiguousarray(bg[y0:y0 + CH, x0:x0 + CW]), rng)
        else:
            k = int(rng.integers(len(self.names))) if self.train else i % len(self.names)
            img = cv2.imread(f"{ROOT}/{self.split}/images/{self.names[k]}")
            if self.train and rng.random() < 0.6:
                img, lr = shear_crop(img, self.labels[k], rng, float(rng.uniform(-1.2, 1.2)))
            else:
                img, lr = crop_with_labels(img, self.labels[k], rng)
            if self.train:
                img = photometric(img, rng)
                if rng.random() < 0.5:
                    img = img[:, :, rng.permutation(3)]
        lc = make_lc(lr, img.shape[1])
        if self.train and rng.random() < self.p_rot:
            for _ in range(int(rng.choice([1, 3]))):
                img, lr, lc = rot_ccw(img, lr, lc)
        img, lr, lc = square_crop(img, lr, lc, rng)
        if self.train and rng.random() < 0.5:
            img, lr, lc = flip_x(img, lr, lc)
        if self.train and rng.random() < 0.5:
            img, lr, lc = flip_y(img, lr, lc)
        return (to_tensor(np.ascontiguousarray(img)), torch.from_numpy(np.ascontiguousarray(lr)),
                torch.from_numpy(np.ascontiguousarray(lc)))
