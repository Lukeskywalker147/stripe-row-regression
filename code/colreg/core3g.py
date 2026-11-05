"""GPU-synth training data: workers return either a real-image sample or a synth 'plan'; assemble() composes plans on the GPU."""
import numpy as np, cv2, torch
from core3 import *
import synth_gpu as SG

MEAN_T = torch.from_numpy(MEAN).view(1, 3, 1, 1)
STD_T = torch.from_numpy(STD).view(1, 3, 1, 1)


def passthrough(b):
    return b


class MixDSG(MixDS3):
    def __getitem__(self, i):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng((self.seed, i, wi.id if wi else 0, int(torch.randint(0, 2 ** 30, (1,)))))
        if rng.random() < self.p_synth and self.bgs:
            def load_bg(r):
                bg = cv2.imread(self.bgs[int(r.integers(len(self.bgs)))])
                H, W = bg.shape[:2]
                y0 = int(r.integers(0, H - CH + 1)); x0 = int(r.integers(0, W - CW + 1))
                return np.ascontiguousarray(bg[y0:y0 + CH, x0:x0 + CW])
            plan = SG.plan_render(rng, load_bg)
            plan["post_seed"] = int(rng.integers(2 ** 31 - 1))
            return ("synth", plan)
        k = int(rng.integers(len(self.names)))
        img = cv2.imread(f"{ROOT}/{self.split}/images/{self.names[k]}")
        if rng.random() < 0.6:
            img, lr = shear_crop(img, self.labels[k], rng, float(rng.uniform(-1.2, 1.2)))
        else:
            img, lr = crop_with_labels(img, self.labels[k], rng)
        pp = photo_params(rng)
        lc = make_lc(lr, img.shape[1])
        if rng.random() < self.p_rot:
            for _ in range(int(rng.choice([1, 3]))):
                img, lr, lc = rot_ccw(img, lr, lc)
        img, lr, lc = square_crop(img, lr, lc, rng)
        if rng.random() < 0.5:
            img, lr, lc = flip_x(img, lr, lc)
        if rng.random() < 0.5:
            img, lr, lc = flip_y(img, lr, lc)
        return ("real", np.ascontiguousarray(img), torch.from_numpy(np.ascontiguousarray(lr)), torch.from_numpy(np.ascontiguousarray(lc)), pp)


def photo_params(rng):
    """parameters of core.photometric + random channel permutation, applied later on the GPU"""
    mult = rng.uniform(0.6, 1.3) * (1 + rng.uniform(-0.1, 0.1, 3))
    aff = rng.random() < 0.5
    gain = rng.uniform(0.5, 1.5, 3) if aff else np.ones(3)
    off = rng.uniform(-15, 15, 3) if aff else np.zeros(3)
    gamma = rng.uniform(0.8, 1.25); sig = rng.uniform(0, 3)
    perm = rng.permutation(3) if rng.random() < 0.5 else np.arange(3)
    return np.concatenate([mult, gain, off, [gamma, sig], perm]).astype(np.float32)


def photo_gpu(u8, pp, dev):
    """u8 (B,S,S,3) BGR uint8 cuda, pp (B,14) -> float (B,3,S,S) 0..255"""
    x = u8.permute(0, 3, 1, 2).float()
    v = lambda a, b: pp[:, a:b, None, None]
    x = x * v(0, 3)
    x = (x * v(3, 6) + v(6, 9)).clamp(0, 255)
    x = 255 * (x / 255) ** v(9, 10)
    x = (x + torch.randn_like(x) * v(10, 11)).clamp(0, 255).round()
    return torch.gather(x, 1, pp[:, 11:14].long()[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3]))


def post_geometry(lr, rng, p_rot):
    """same label/geometry logic as MixDS3, applied to an index image so the GPU image can be gathered afterwards"""
    Hh, Ww = SG.H, SG.W
    idx = np.arange(Hh * Ww, dtype=np.int64).reshape(Hh, Ww, 1)
    lc = make_lc(lr, Ww)
    if rng.random() < p_rot:
        for _ in range(int(rng.choice([1, 3]))):
            idx, lr, lc = rot_ccw(idx, lr, lc)
    idx, lr, lc = square_crop(idx, lr, lc, rng)
    if rng.random() < 0.5:
        idx, lr, lc = flip_x(idx, lr, lc)
    if rng.random() < 0.5:
        idx, lr, lc = flip_y(idx, lr, lc)
    return np.ascontiguousarray(idx[:, :, 0]), np.ascontiguousarray(lr), np.ascontiguousarray(lc)


def assemble(batch, dev, p_rot=0.5):
    """list of worker items -> (x (B,3,S,S) normalised, lr (B,S), lc (B,S)) on dev"""
    real = [it for it in batch if it[0] == "real"]
    synth = [it[1] for it in batch if it[0] != "real"]
    xs, lrs, lcs = [], [], []
    if real:
        u8 = torch.from_numpy(np.stack([it[1] for it in real])).to(dev, non_blocking=True)
        pp = torch.from_numpy(np.stack([it[4] for it in real])).to(dev, non_blocking=True)
        xs += list((photo_gpu(u8, pp, dev) - MEAN_T.to(dev)) / STD_T.to(dev))
        lrs += [it[2] for it in real]; lcs += [it[3] for it in real]
    if synth:
        U = SG.Uploader(dev)
        for p in synth:
            U.walk(p)
        U.flush()
        outs = [SG.compose(p, dev, U) for p in synth]
        labs = torch.stack([o[1] for o in outs]).cpu().numpy()
        dec = SG.jpeg_roundtrip([o[0] for o in outs], [p["q"] for p in synth], dev)
        imgs = torch.stack([d.reshape(3, -1) for d in dec])
        geo = [post_geometry(labs[j], np.random.default_rng(p["post_seed"]), p_rot) for j, p in enumerate(synth)]
        idx = torch.from_numpy(np.stack([g[0].reshape(-1) for g in geo])).to(dev)
        g = torch.gather(imgs, 2, idx[:, None, :].expand(-1, 3, -1)).float().view(len(synth), 3, S, S)
        g = (g - MEAN_T.to(dev)) / STD_T.to(dev)
        xs += list(g)
        lrs += [torch.from_numpy(x[1]) for x in geo]; lcs += [torch.from_numpy(x[2]) for x in geo]
    return torch.stack(xs), torch.stack(lrs).to(dev), torch.stack(lcs).to(dev)
