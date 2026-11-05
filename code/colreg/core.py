import glob, os, sys
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth

ROOT = os.path.expanduser("~/stripe_seg/data_scan_v3")
CH, CW = 512, 768
MEAN = np.array([0.406, 0.456, 0.485], np.float32) * 255  # BGR
STD = np.array([0.225, 0.224, 0.229], np.float32) * 255


def to_tensor(img):
    x = (img.astype(np.float32) - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def crop_with_labels(img, lab, rng, ch=CH, cw=CW):
    H, W = img.shape[:2]
    y0 = int(rng.integers(0, H - ch + 1))
    pres = lab[y0:y0 + ch]
    pres = pres[pres >= 0]
    if len(pres):
        cx = float(np.median(pres)) + rng.uniform(-0.3, 0.3) * cw
    else:
        cx = rng.uniform(cw / 2, W - cw / 2)
    x0 = int(np.clip(cx - cw / 2, 0, W - cw))
    im = img[y0:y0 + ch, x0:x0 + cw]
    lb = lab[y0:y0 + ch].copy()
    pm = lb >= 0
    lb[pm] -= x0
    lb[pm & ((lb < 2) | (lb > cw - 3))] = np.nan
    return im, lb


def shear_crop(img, lab, rng, s, ch=CH, cw=CW):
    """Horizontal shear x' = x + s*(y-yc): rows stay rows, labels shift exactly."""
    H, W = img.shape[:2]
    y0 = int(rng.integers(0, H - ch + 1)); yc = y0 + ch / 2
    ys = np.arange(y0, y0 + ch)
    lb = lab[y0:y0 + ch].copy()
    pm = lb >= 0
    if pm.any():
        cx = float(np.median(lb[pm] + s * (ys[pm] - yc))) + rng.uniform(-0.3, 0.3) * cw
    else:
        cx = rng.uniform(cw / 2, W - cw / 2)
    xoff = cx - cw / 2
    M = np.array([[1, s, -s * yc - xoff], [0, 1, -y0]], np.float32)
    im = cv2.warpAffine(img, M, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    lb[pm] = lb[pm] + s * (ys[pm] - yc) - xoff
    lb[pm & ((lb < 2) | (lb > cw - 3))] = np.nan
    return im, lb


def photometric(img, rng):
    f = img.astype(np.float32)
    f = f * rng.uniform(0.6, 1.3) * (1 + rng.uniform(-0.1, 0.1, 3))[None, None, :]
    if rng.random() < 0.5:
        f = f * rng.uniform(0.5, 1.5, 3)[None, None, :] + rng.uniform(-15, 15, 3)[None, None, :]
    g = rng.uniform(0.8, 1.25)
    f = 255 * (np.clip(f, 0, 255) / 255) ** g
    f = f + rng.standard_normal(f.shape).astype(np.float32) * rng.uniform(0, 3)
    return np.clip(f, 0, 255).astype(np.uint8)


class MixDS(torch.utils.data.Dataset):
    def __init__(self, split="train", n=1000, p_synth=0.5, seed=0, bg_dir="bg"):
        self.n, self.p_synth, self.seed, self.split = n, p_synth, seed, split
        d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"labels/{split}.npz"))
        self.names, self.labels = d["names"], d["labels"]
        self.bgs = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), bg_dir, "*.jpg")))
        self.train = split == "train"

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i, torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 0, int(torch.randint(0, 2 ** 30, (1,)))) if self.train else (self.seed, i))
        if rng.random() < self.p_synth and self.bgs:
            bg = cv2.imread(self.bgs[int(rng.integers(len(self.bgs)))])
            H, W = bg.shape[:2]
            y0 = int(rng.integers(0, H - CH + 1)); x0 = int(rng.integers(0, W - CW + 1))
            img, lab = synth.render(np.ascontiguousarray(bg[y0:y0 + CH, x0:x0 + CW]), rng)
        else:
            k = int(rng.integers(len(self.names))) if self.train else i % len(self.names)
            img = cv2.imread(f"{ROOT}/{self.split}/images/{self.names[k]}")
            if self.train and rng.random() < 0.6:
                img, lab = shear_crop(img, self.labels[k], rng, float(rng.uniform(-1.2, 1.2)))
            else:
                img, lab = crop_with_labels(img, self.labels[k], rng)
            if self.train:
                img = photometric(img, rng)
        if self.train and rng.random() < 0.5:
            img = img[:, ::-1]
            lab = np.where(lab >= 0, CW - 1 - lab, lab)
        if self.train and rng.random() < 0.5:
            img = img[::-1]
            lab = lab[::-1]
        return to_tensor(np.ascontiguousarray(img)), torch.from_numpy(np.ascontiguousarray(lab).astype(np.float32))


def build_model(encoder="resnet18", weights=None):
    return smp.Unet(encoder, encoder_weights=weights, classes=2, decoder_channels=(128, 64, 32, 16, 16))


def row_logits(out):
    """out (B,2,H,W) -> (B,H,W+1): per-row [pixel scores..., absent]."""
    s = out[:, 0]
    a = out[:, 1].amax(dim=2, keepdim=True)
    return torch.cat([s, a], dim=2)


def decode(logits, win=3):
    """logits (B,H,W+1) -> pos (B,H) sub-pixel x, p_abs (B,H), conf (B,H) mass in window."""
    p = F.softmax(logits.float(), dim=2)
    pa = p[..., -1]
    pp = p[..., :-1]
    W = pp.shape[-1]
    k = pp.argmax(2)
    xs = torch.arange(W, device=p.device)[None, None, :].float()
    m = ((xs - k[..., None].float()).abs() <= win).float()
    pw = pp * m
    mass = pw.sum(2)
    pos = (pw * xs).sum(2) / mass.clamp_min(1e-9)
    return pos, pa, mass


def row_loss(logits, lab, sigma=1.0, lam=0.5, win=3):
    B, H, W1 = logits.shape
    W = W1 - 1
    valid = ~torch.isnan(lab)
    if valid.sum() == 0:
        return logits.sum() * 0, {}
    lg = logits[valid].float()
    lb = lab[valid]
    pres = lb >= 0
    xs = torch.arange(W, device=lg.device)[None, :].float()
    tgt = torch.zeros_like(lg)
    g = torch.exp(-(xs - lb[:, None]) ** 2 / (2 * sigma ** 2))
    g = g / g.sum(1, keepdim=True).clamp_min(1e-9)
    tgt[:, :W] = torch.where(pres[:, None], g, torch.zeros_like(g))
    tgt[:, W] = (~pres).float()
    logp = F.log_softmax(lg, 1)
    ce = -(tgt * logp).sum(1)
    ce = torch.where(pres, ce, ce * 2.0)
    l1 = lg.new_zeros(())
    if pres.any():
        lp = lg[pres][:, :W]
        t = lb[pres]
        m = ((xs - t[:, None]).abs() <= win)
        pw = F.softmax(lp.masked_fill(~m, -1e4), 1)
        pos = (pw * xs).sum(1)
        l1 = (pos - t).abs().mean()
    loss = ce.mean() + lam * l1
    return loss, dict(ce=float(ce.mean()), l1=float(l1))
