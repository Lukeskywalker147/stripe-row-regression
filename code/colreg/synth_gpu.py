"""GPU port of synth3.render. Workers draw all random parameters ("plan", cheap 1-D work only);
the main process composes the full-resolution images on the GPU (noise/blur/stripes/glare/JPEG via nvjpeg).
Distribution-equivalent to synth3 (not bit-identical: noise comes from torch generators)."""
import math
import numpy as np, cv2, torch
import torch.nn.functional as F
from torchvision.io import encode_jpeg, decode_jpeg
from synth import smooth_noise, make_curve, occluders
from synth3 import laser_colour

H, W = 512, 768


def stripe_params(xc, rng, A0, col, w_range=(0.8, 5.5)):
    cos_t = 1.0 / np.sqrt(1 + np.gradient(xc) ** 2)
    sc0 = np.exp(rng.uniform(np.log(w_range[0]), np.log(w_range[1])))
    sc = sc0 * np.clip(1 + 0.35 * smooth_noise(H, rng.uniform(10, 60), rng), 0.5, 2.5)
    A = A0 * np.exp(0.35 * smooth_noise(H, rng.uniform(8, 50), rng))
    A = A * np.clip(1 + 0.2 * smooth_noise(H, 1.5, rng), 0.3, 2)
    if rng.random() < 0.5:
        ya = rng.integers(0, H // 3) if rng.random() < 0.5 else 0
        yb = rng.integers(2 * H // 3, H) if rng.random() < 0.5 else H
        A[:ya] = 0; A[yb:] = 0
    for _ in range(rng.integers(0, 3)):
        a = rng.integers(0, H - 10); b = min(H, a + rng.integers(5, 60))
        A[a:b] *= rng.uniform(0, 0.08)
    rh = rng.uniform(0, 0.9)
    sh = sc * rng.uniform(2.5, 6.0)
    one = rng.random() < 0.4
    side = float(rng.choice([-1, 1])) if one else 0.0
    arr = np.stack([xc, cos_t, sc, sh, A]).astype(np.float32)
    sca = np.array([rh, float(one), side, *col], np.float32)
    return arr, sca


def plan_proc(rng):
    mean = rng.uniform(15, 235)
    tint = 1 + rng.uniform(-0.35, 0.35, 3)
    if rng.random() < 0.3:
        tint = np.array([rng.uniform(0.1, 0.5), rng.uniform(0.6, 1.0), 1.0])
    con = rng.uniform(0, 60) * (rng.random() < 0.85)
    d = dict(kind="proc", mean=mean, tint=tint.astype(np.float32), con=con,
             scales=[float(s) for s in rng.choice([1, 2, 4, 8, 16, 40, 100], 3, replace=False)], w=rng.uniform(0.2, 1.0, 3))
    d["ripple"] = None
    if rng.random() < 0.4:
        ang = rng.uniform(0, np.pi); per = rng.uniform(4, 60)
        d["ripple"] = (ang, per, rng.uniform(0, 6.28), rng.uniform(3, 30))
    d["blob"] = None
    if rng.random() < 0.3:
        d["blob"] = (rng.uniform(0.2, 1.0, 2), rng.uniform(1.0, 2.0), rng.uniform(-80, 80))
    d["grad"] = None
    if rng.random() < 0.4:
        d["grad"] = (rng.uniform(0, 2 * np.pi), rng.uniform(-120, 120))
    return d


def plan_real(rng):
    d = dict(kind="real", scale=rng.uniform(0.4, 1.9), tint=(1 + rng.uniform(-0.15, 0.15, 3)).astype(np.float32), cf=None)
    if rng.random() < 0.5:
        c0 = rng.uniform(0, 255, 3).astype(np.float32)
        if rng.random() < 0.5:
            c0 = np.array([rng.uniform(0, 60), rng.uniform(150, 240), rng.uniform(180, 255)], np.float32)
        c1 = np.clip(c0 * rng.uniform(0.4, 1.3), 0, 255)
        ang = rng.uniform(0, 2 * np.pi); k = rng.uniform(1, 6)
        cy, cx = rng.uniform(0, H), rng.uniform(0, W); sy, sx = rng.uniform(80, 400), rng.uniform(80, 500)
        d["cf"] = (c0, c1, ang, k, cy, cx, sy, sx, rng.uniform(0.3, 1.0))
    return d


def plan_render(rng, load_bg, p_empty=0.08, p_proc=0.4):
    """load_bg(rng) -> HxWx3 uint8 crop; only called if a real background is used."""
    if rng.random() < p_proc:
        base = plan_proc(rng); bg = None
    else:
        base = plan_real(rng); bg = load_bg(rng)
    xc = make_curve(H, W, rng)
    glares = []
    for _ in range(rng.integers(0, 4)):
        if rng.random() < 0.5:
            cy = rng.uniform(0, H); cx = float(np.interp(cy, np.arange(H), xc)) + rng.uniform(-60, 60)
        else:
            cy = rng.uniform(0, H); cx = rng.uniform(0, W)
        sy, sx = rng.uniform(8, 90), rng.uniform(8, 90)
        ang = rng.uniform(0, np.pi); amp = rng.uniform(40, 420)
        glares.append([cy, cx, sy, sx, ang, amp, 1.0, rng.uniform(0.7, 1.0), rng.uniform(0.5, 1.0)])
    col = laser_colour(rng)
    A0 = float(np.exp(rng.uniform(np.log(30), np.log(900))))
    nd = int(rng.choice([0, 1, 2, 3, 4], p=[0.45, 0.2, 0.15, 0.1, 0.1]))
    dist = []
    for _ in range(nd):
        xg = make_curve(H, W, rng)
        if rng.random() < 0.6:
            xg = xc + rng.choice([-1, 1]) * rng.uniform(25, 220) + 0.3 * (xg - xg.mean())
        c2 = col if rng.random() < 0.8 else laser_colour(rng)
        dist.append(stripe_params(xg.astype(np.float32), rng, A0 * rng.uniform(0.05, 0.7), c2))
    occ = occluders(H, W, rng)
    ok_ = rng.uniform(0.02, 0.3)
    tgt = None if rng.random() < p_empty else stripe_params(xc.astype(np.float32), rng, A0, col)
    gamma = rng.uniform(0.8, 1.3); sb = rng.uniform(0, 1.2); noise = rng.uniform(0.5, 6)
    q = int(rng.integers(40, 96))
    return dict(base=base, bg=bg, glares=np.array(glares, np.float32).reshape(-1, 9), dist=dist, tgt=tgt, xc=xc.astype(np.float32),
                occ=(np.round(occ * 255).astype(np.uint8) if occ.any() else None), ok=ok_, gamma=gamma, sb=sb, noise=noise, q=q,
                seed=int(rng.integers(2 ** 31 - 1)))


# ---------------------------------------------------------------- GPU side
def gblur(x, sigma):
    """separable gaussian over the last two dims of (N,1,h,w), reflect padding"""
    r = max(1, int(math.ceil(4 * sigma)))
    t = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-t * t / (2 * sigma * sigma)); k = k / k.sum()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))


def smooth_field(g, sigma, dev):
    f = 1 if sigma <= 8 else int(math.ceil(sigma / 8))
    h, w = -(-H // f), -(-W // f)
    n = torch.randn(1, 1, h, w, generator=g, device=dev)
    n = gblur(n, sigma / f)
    if f > 1:
        n = F.interpolate(n, size=(H, W), mode="bilinear", align_corners=False)
    n = n[0, 0]
    return n / (n.std() + 1e-6)


def tex(g, scales, w, dev):
    f = sum(float(wi) * smooth_field(g, s, dev) for s, wi in zip(scales, w))
    return f / (f.std() + 1e-6)


class Uploader:
    """collects every ndarray in the plans, ships them in two pinned H2D copies (many tiny pageable copies stall on a shared GPU)"""
    _pin = {}

    def __init__(self, dev):
        self.dev, self.meta, self.parts = dev, {}, dict(f=[], u=[])
        self.n = dict(f=0, u=0)

    def walk(self, o):
        if isinstance(o, np.ndarray):
            if o.size and id(o) not in self.meta:
                k = "u" if o.dtype == np.uint8 else "f"
                a = np.ascontiguousarray(o, dtype=np.uint8 if k == "u" else np.float32).ravel()
                self.meta[id(o)] = (k, self.n[k], o.shape); self.parts[k].append(a); self.n[k] += a.size
        elif isinstance(o, dict):
            for v in o.values(): self.walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o: self.walk(v)

    def flush(self):
        self.g = {}
        for k, dt in (("f", torch.float32), ("u", torch.uint8)):
            n = self.n[k]
            buf = self._pin.get(k)
            if buf is None or buf.numel() < n:
                buf = self._pin[k] = torch.empty(max(n, 1) * 2, dtype=dt, pin_memory=True)
            if n:
                buf.numpy()[:n] = np.concatenate(self.parts[k])
            self.g[k] = buf[:n].to(self.dev, non_blocking=True)

    def __call__(self, a):
        k, off, shape = self.meta[id(a)]
        return self.g[k][off:off + int(np.prod(shape))].view(shape)


def make_base(p, g, yy, xx, dev, U):
    b = p["base"]
    if b["kind"] == "proc":
        base = (float(b["mean"]) * U(b["tint"])).view(3, 1, 1).expand(3, H, W).clone()
        base += (tex(g, b["scales"], b["w"], dev) * b["con"])[None]
        if b["ripple"]:
            ang, per, ph, amp = b["ripple"]
            base += (torch.sin(2 * math.pi * (xx * math.cos(ang) + yy * math.sin(ang)) / per + ph) * amp)[None]
        if b["blob"]:
            w2, thr, val = b["blob"]
            base += ((tex(g, [2, 5], w2, dev) > thr).float() * val)[None]
        if b["grad"]:
            ang, amt = b["grad"]
            base += ((((xx - W / 2) * math.cos(ang) + (yy - H / 2) * math.sin(ang)) / max(H, W)) * amt)[None]
        return base.clamp_(0, 255)
    bg = U(p["bg"]).permute(2, 0, 1).float()
    base = bg * (float(b["scale"]) * U(b["tint"])).view(3, 1, 1)
    if b["cf"] is not None:
        c0, c1, ang, k, cy, cx, sy, sx, mix = b["cf"]
        t = ((math.cos(ang) * (xx - W / 2) + math.sin(ang) * (yy - H / 2)) / (0.6 * max(H, W))) * k
        t = torch.sigmoid(t)[None]
        f = U(c0).view(3, 1, 1) * (1 - t) + U(c1).view(3, 1, 1) * t
        m = torch.exp(-(((yy - cy) / sy) ** 4 + ((xx - cx) / sx) ** 4))[None]
        base = base * (1 - m) + f * m * mix
    return base


def stripes_layer(items, U, xx):
    arr = torch.stack([U(a) for a, _ in items])          # K,5,H
    sca = torch.stack([U(s) for _, s in items])          # K,6
    xc, cs, sc, sh, A = (arr[:, i, :, None] for i in range(5))
    rh, one, side = (sca[:, i].view(-1, 1, 1) for i in range(3))
    d = (xx[None] - xc) * cs
    v = torch.exp(-d * d / (2 * sc * sc)) + rh * torch.exp(-d * d / (2 * sh * sh)) * torch.where(one > 0, torch.sigmoid(side * d / 1.5), torch.ones_like(d))
    return torch.einsum("khw,kc->chw", v * A, sca[:, 3:6])


def compose(p, dev, U):
    """plan -> (uint8 RGB (3,H,W) cuda before JPEG, labels (H,) cuda float with NaN=ignore, -1=absent). No host syncs."""
    g = torch.Generator(device=dev); g.manual_seed(p["seed"])
    yy = torch.arange(H, device=dev, dtype=torch.float32)[:, None]
    xx = torch.arange(W, device=dev, dtype=torch.float32)[None, :]
    base = make_base(p, g, yy, xx, dev, U)
    if len(p["glares"]):
        gl = U(p["glares"])
        cy, cx, sy, sx, ang, amp = (gl[:, i].view(-1, 1, 1) for i in range(6))
        dy, dx = yy[None] - cy, xx[None] - cx
        u = dx * torch.cos(ang) + dy * torch.sin(ang); v = -dx * torch.sin(ang) + dy * torch.cos(ang)
        base = base + torch.einsum("nhw,nc->chw", amp * torch.exp(-(u * u / (2 * sx * sx) + v * v / (2 * sy * sy))), gl[:, 6:9])
    if p["dist"]:
        base = base + stripes_layer(p["dist"], U, xx)
    base0 = base.clamp(0, 255)
    lab = torch.full((H,), -1.0, device=dev)
    occ = U(p["occ"]).float() / 255 if p["occ"] is not None else None
    if p["tgt"] is None:
        full = base0
    else:
        full = (base + stripes_layer([p["tgt"]], U, xx)).clamp(0, 255)
        xc = U(p["xc"])
        xi = xc.round().long().clamp(0, W - 1)
        o5 = torch.arange(-2, 3, device=dev)
        cs = (xi[:, None] + o5[None]).clamp(0, W - 1)
        diff = (full - base0).abs().amax(0)
        vis = torch.gather(diff, 1, cs).amax(1)
        if occ is not None:
            c3 = (xi[:, None] + o5[None, 1:4]).clamp(0, W - 1)
            vis = vis * (1 - torch.gather(occ, 1, c3).mean(1) * (1 - p["ok"]))
        inside = (xc >= 2) & (xc <= W - 3)
        nan = torch.full_like(xc, float("nan"))
        lab = torch.where(inside & (vis >= 25), xc, torch.where(inside & (vis >= 12), nan, lab))
    img = full if occ is None else full * (1 - occ[None] * (1 - p["ok"]))
    img = 255 * (img.clamp(0, 255) / 255) ** p["gamma"]
    if p["sb"] > 0.2:
        img = gblur(img.unsqueeze(1), p["sb"]).squeeze(1)
    img = img + torch.randn(3, H, W, generator=g, device=dev) * p["noise"]
    return img.clamp(0, 255).round().to(torch.uint8).flip(0).contiguous(), lab


def jpeg_roundtrip(rgbs, qs, dev):
    """nvjpeg encode (grouped by quality bucket) -> host bytes -> nvjpeg decode; returns BGR uint8 cuda tensors"""
    qb = [40 + 8 * ((q - 40) // 8) for q in qs]
    enc = [None] * len(rgbs)
    for q in sorted(set(qb)):
        ids = [i for i, x in enumerate(qb) if x == q]
        for i, e in zip(ids, encode_jpeg([rgbs[i] for i in ids], quality=q)):
            enc[i] = e
    dec = decode_jpeg([e.cpu() for e in enc], device=dev)
    return [d.flip(0) for d in dec]
