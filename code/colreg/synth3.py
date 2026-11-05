"""v3 renderer: wider generality (laser colour, width, amplitude, procedural backgrounds, up to 4 distractor stripes).
Convention: the strongest stripe is the target; distractors are weaker (0.05-0.7 x). Label = core centre, as in v2."""
import numpy as np, cv2
from synth import smooth_noise, make_curve, glare_layer, colour_field, occluders


def laser_colour(rng):
    r = rng.random()
    if r < 0.38:   c = [1.0, rng.uniform(0.3, 0.9), rng.uniform(0.05, 0.5)]      # blue (BGR)
    elif r < 0.48: c = [1.0, rng.uniform(0.7, 1.0), rng.uniform(0.4, 0.95)]      # cyan-white
    elif r < 0.63: c = [rng.uniform(0.05, 0.5), 1.0, rng.uniform(0.05, 0.5)]     # green
    elif r < 0.83: c = [rng.uniform(0.05, 0.4), rng.uniform(0.05, 0.5), 1.0]     # red
    elif r < 0.93: c = [1.0, rng.uniform(0.9, 1.0), rng.uniform(0.9, 1.0)]       # white
    else:
        c = rng.uniform(0.05, 1.0, 3); c = c / c.max()
    return np.array(c, np.float32)


def stripe_layer(H, W, xc, rng, A0, col, w_range=(0.8, 5.5)):
    xx = np.arange(W, dtype=np.float32)[None, :]
    cos_t = 1.0 / np.sqrt(1 + np.gradient(xc) ** 2)
    d = (xx - xc[:, None]) * cos_t[:, None]
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
    prof = np.exp(-d ** 2 / (2 * sc[:, None] ** 2))
    rh = rng.uniform(0, 0.9)
    sh = sc * rng.uniform(2.5, 6.0)
    halo = rh * np.exp(-d ** 2 / (2 * sh[:, None] ** 2))
    if rng.random() < 0.4:
        halo = halo * (1 / (1 + np.exp(-rng.choice([-1, 1]) * d / 1.5)))
    return ((prof + halo) * A[:, None])[:, :, None] * col[None, None, :]


def _tex(H, W, rng, scales):
    f = np.zeros((H, W), np.float32)
    for s in scales:
        n = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), s)
        f += rng.uniform(0.2, 1.0) * n / (n.std() + 1e-6)
    return f / (f.std() + 1e-6)


def proc_bg(H, W, rng):
    mean = rng.uniform(15, 235)
    tint = (1 + rng.uniform(-0.35, 0.35, 3)).astype(np.float32)
    if rng.random() < 0.3:
        tint = np.array([rng.uniform(0.1, 0.5), rng.uniform(0.6, 1.0), 1.0], np.float32)   # yellow
    base = np.full((H, W, 3), mean, np.float32) * tint
    con = rng.uniform(0, 60) * (rng.random() < 0.85)
    base += (_tex(H, W, rng, rng.choice([1, 2, 4, 8, 16, 40, 100], 3, replace=False)) * con)[:, :, None]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if rng.random() < 0.4:      # machined ripples / bands
        ang = rng.uniform(0, np.pi); per = rng.uniform(4, 60)
        base += (np.sin(2 * np.pi * (xx * np.cos(ang) + yy * np.sin(ang)) / per + rng.uniform(0, 6.28)) * rng.uniform(3, 30))[:, :, None]
    if rng.random() < 0.3:      # blobs / specks
        m = (_tex(H, W, rng, [2, 5]) > rng.uniform(1.0, 2.0)).astype(np.float32)
        base += (m * rng.uniform(-80, 80))[:, :, None]
    if rng.random() < 0.4:      # gradient
        ang = rng.uniform(0, 2 * np.pi)
        base += (((xx - W / 2) * np.cos(ang) + (yy - H / 2) * np.sin(ang)) / max(H, W) * rng.uniform(-120, 120))[:, :, None]
    return np.clip(base, 0, 255)


def render(bg, rng, p_empty=0.08, p_proc=0.4):
    H, W = bg.shape[:2]
    if rng.random() < p_proc:
        base = proc_bg(H, W, rng)
    else:
        base = bg.astype(np.float32) * rng.uniform(0.4, 1.9)
        base = base * (1 + rng.uniform(-0.15, 0.15, 3)[None, None, :])
        if rng.random() < 0.5:
            f, m = colour_field(H, W, rng)
            base = base * (1 - m) + f * m * rng.uniform(0.3, 1.0)
    xc = make_curve(H, W, rng)
    base = base + glare_layer(H, W, xc, rng)
    col = laser_colour(rng)
    A0 = float(np.exp(rng.uniform(np.log(30), np.log(900))))
    nd = int(rng.choice([0, 1, 2, 3, 4], p=[0.45, 0.2, 0.15, 0.1, 0.1]))
    for _ in range(nd):
        xg = make_curve(H, W, rng)
        if rng.random() < 0.6:
            xg = xc + rng.choice([-1, 1]) * rng.uniform(25, 220) + 0.3 * (xg - xg.mean())
        c2 = col if rng.random() < 0.8 else laser_colour(rng)
        base = base + stripe_layer(H, W, xg, rng, A0 * rng.uniform(0.05, 0.7), c2)
    base0 = np.clip(base, 0, 255)
    lab = np.full(H, -1, np.float32)
    occ = occluders(H, W, rng)
    ok_ = rng.uniform(0.02, 0.3)
    if rng.random() < p_empty:
        full = base0
    else:
        S = stripe_layer(H, W, xc, rng, A0, col)
        full = np.clip(base + S, 0, 255)
        diff = np.abs(full - base0).max(2)
        xi = np.clip(np.round(xc).astype(int), 0, W - 1)
        vis = np.take_along_axis(diff, np.clip(xi[:, None] + np.arange(-2, 3)[None, :], 0, W - 1), 1).max(1)
        cover = np.take_along_axis(occ, np.clip(xi[:, None] + np.arange(-1, 2)[None, :], 0, W - 1), 1).mean(1)
        vis = vis * (1 - cover * (1 - ok_))
        inside = (xc >= 2) & (xc <= W - 3)
        lab = np.where(inside & (vis >= 25), xc, np.where(inside & (vis >= 12), np.nan, -1)).astype(np.float32)
    img = full * (1 - occ[:, :, None] * (1 - ok_))
    img = 255 * (img / 255) ** rng.uniform(0.8, 1.3)
    sb = rng.uniform(0, 1.2)
    if sb > 0.2:
        img = cv2.GaussianBlur(img, (0, 0), sb)
    img = img + rng.standard_normal(img.shape).astype(np.float32) * rng.uniform(0.5, 6)
    img = np.clip(img, 0, 255).astype(np.uint8)
    enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 96))])[1]
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), lab
