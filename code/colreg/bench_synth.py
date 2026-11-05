"""Scene-agnostic synthetic benchmark: one factor varied at a time, exact per-row GT (x / -1 absent / NaN ignore).
GT centre = geometric centre of the stripe core (same assumption as training data); stripes are near-vertical.
"""
import glob, os
import numpy as np, cv2
from synth import smooth_noise

H, W = 512, 768
BASE = dict(width=1.8, amp=250.0, halo=0.2, halo_mult=4.0, slope=0.2, tex=20.0, bgkind="gray", bgval=110.0,
            ndist=0, nocc=0, noise=2.0, jpeg=90, colour="blue", nglare=0, blur=0.6, empty=False, rot=0)

# levels are ordered easy -> hard
AXES = {
    "width":      [dict(width=v) for v in (0.9, 1.5, 2.5, 4.0, 6.0)],
    "contrast":   [dict(amp=v) for v in (500.0, 250.0, 120.0, 60.0, 30.0)],
    "saturation": [dict(amp=v) for v in (250.0, 500.0, 1000.0, 2000.0, 4000.0)],
    "halo":       [dict(halo=v) for v in (0.0, 0.3, 0.6, 1.0, 1.5)],
    "slope":      [dict(slope=v) for v in (0.0, 0.3, 0.6, 0.9, 1.2)],
    "texture":    [dict(tex=v) for v in (0.0, 10.0, 25.0, 45.0, 70.0)],
    "background": [dict(bgkind=k, bgval=v) for k, v in (("gray", 110.0), ("dark", 30.0), ("bright", 200.0), ("yellow", 200.0), ("real", 0.0))],
    "distractors": [dict(ndist=v) for v in (0, 1, 2, 3, 4)],
    "occlusion":  [dict(nocc=v) for v in (0, 1, 2, 3, 5)],
    "noise_jpeg": [dict(noise=n, jpeg=q) for n, q in ((1, 95), (3, 80), (6, 65), (10, 50), (16, 35))],
    "colour":     [dict(colour=c) for c in ("blue", "cyan_white", "green", "red", "white")],
    "orient":     [dict(rot=v) for v in (0, 1, 2, 3)],
    "glare":      [dict(nglare=v) for v in (0, 1, 2, 4, 6)],
}
COL = dict(blue=(1.0, 0.45, 0.1), cyan_white=(1.0, 0.9, 0.75), green=(0.15, 1.0, 0.2), red=(0.1, 0.2, 1.0), white=(1.0, 1.0, 1.0))
_BGS = None


def real_bg(rng):
    global _BGS
    if _BGS is None:
        _BGS = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bg", "*.jpg")))
    b = cv2.imread(_BGS[int(rng.integers(len(_BGS)))])
    y0 = int(rng.integers(0, b.shape[0] - H + 1)); x0 = int(rng.integers(0, b.shape[1] - W + 1))
    return b[y0:y0 + H, x0:x0 + W].astype(np.float32)


def tex_field(rng, contrast):
    f = np.zeros((H, W), np.float32)
    for s, w in ((2, 0.3), (6, 0.4), (20, 0.5), (60, 0.6)):
        n = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), s)
        f += w * n / (n.std() + 1e-6)
    return f / (f.std() + 1e-6) * contrast


def background(rng, p):
    if p["bgkind"] == "real":
        return real_bg(rng)
    base = np.full((H, W, 3), p["bgval"], np.float32)
    if p["bgkind"] == "yellow":
        base = base * np.array([0.2, 0.85, 1.0], np.float32)
    elif p["bgkind"] != "gray":
        base = base * (1 + rng.uniform(-0.1, 0.1, 3)).astype(np.float32)
    return base + tex_field(rng, p["tex"])[:, :, None]


def stripe(xc, sigma, A, halo, hmult, col, rng):
    xx = np.arange(W, dtype=np.float32)[None, :]
    cos_t = 1.0 / np.sqrt(1 + np.gradient(xc) ** 2)
    d = (xx - xc[:, None]) * cos_t[:, None]
    prof = np.exp(-d ** 2 / (2 * sigma ** 2)) + halo * np.exp(-d ** 2 / (2 * (sigma * hmult) ** 2))
    amp = A * np.clip(1 + 0.08 * smooth_noise(H, 20, rng), 0.7, 1.3)
    return (prof * amp[:, None])[:, :, None] * np.array(col, np.float32)[None, None, :]


def glare(rng, xc):
    L = np.zeros((H, W, 3), np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy = rng.uniform(0, H); cx = float(np.interp(cy, np.arange(H), xc)) + rng.uniform(-80, 80)
    sy, sx = rng.uniform(8, 60), rng.uniform(8, 60)
    g = np.exp(-(((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2) / 2)
    return (rng.uniform(150, 350) * g)[:, :, None] * np.array([1.0, 0.9, 0.8], np.float32)


def render(rng, over):
    p = dict(BASE); p.update(over)
    y = np.arange(H, dtype=np.float32)
    sgn = rng.choice([-1, 1])
    xc = (W / 2 + rng.uniform(-60, 60) + sgn * p["slope"] * (y - H / 2)
          + 8 * np.sin(2 * np.pi * y / H * rng.uniform(0.5, 2) + rng.uniform(0, 6.28))).astype(np.float32)
    base = background(rng, p)
    col = COL[p["colour"]]
    for _ in range(p["ndist"]):
        xg = xc + rng.choice([-1, 1]) * rng.uniform(30, 220) + rng.uniform(-0.15, 0.15) * (y - H / 2)
        base = base + stripe(xg.astype(np.float32), p["width"], p["amp"] * 0.6, p["halo"], p["halo_mult"], col, rng)
    for _ in range(p["nglare"]):
        base = base + glare(rng, xc)
    base0 = np.clip(base, 0, 255)
    occ = np.zeros((H, W), np.float32)
    for _ in range(p["nocc"]):
        yo = rng.uniform(0.15 * H, 0.85 * H); xo = float(np.interp(yo, y, xc))
        a = np.arctan2(1.0, -np.gradient(xc)[int(yo)]) + rng.uniform(-0.4, 0.4)
        dx, dy = 320 * np.cos(a), 320 * np.sin(a)
        cv2.line(occ, (int(xo - dx), int(yo - dy)), (int(xo + dx), int(yo + dy)), 1.0, 4, cv2.LINE_AA)
    ok_ = 0.05
    lab = np.full(H, -1, np.float32)
    if p["empty"]:
        full = base0
    else:
        full = np.clip(base + stripe(xc, p["width"], p["amp"], p["halo"], p["halo_mult"], col, rng), 0, 255)
        diff = np.abs(full - base0).max(2)
        xi = np.clip(np.round(xc).astype(int), 0, W - 1)
        vis = np.take_along_axis(diff, np.clip(xi[:, None] + np.arange(-2, 3)[None, :], 0, W - 1), 1).max(1)
        cover = np.take_along_axis(occ, np.clip(xi[:, None] + np.arange(-1, 2)[None, :], 0, W - 1), 1).mean(1)
        vis = vis * (1 - cover * (1 - ok_))
        inside = (xc >= 2) & (xc <= W - 3)
        lab = np.where(inside & (vis >= 25), xc, np.where(inside & (vis >= 12), np.nan, -1)).astype(np.float32)
    img = full * (1 - occ[:, :, None] * (1 - ok_))
    if p["blur"] > 0.2:
        img = cv2.GaussianBlur(img, (0, 0), p["blur"])
    img = img + rng.standard_normal(img.shape).astype(np.float32) * p["noise"]
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(p["jpeg"])])[1], cv2.IMREAD_COLOR)
    return img, lab


def sample(axis, level, k):
    """deterministic sample k of (axis, level); axis 'empty' = stripe-free images."""
    names = list(AXES)
    ai = -1 if axis == "empty" else names.index(axis)
    rng = np.random.default_rng((2026, ai + 1, level, k))
    over = dict(empty=True) if axis == "empty" else AXES[axis][level]
    return render(rng, over)
