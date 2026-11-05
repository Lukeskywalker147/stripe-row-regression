"""Faster, output-equivalent version of synth3 (banded stripe/glare evaluation, LUT gamma, float32). Same rng draw order as synth3 where possible."""
import numpy as np, cv2
from synth import smooth_noise, make_curve, colour_field, occluders
from synth3 import laser_colour, proc_bg


def stripe_layer(H, W, xc, rng, A0, col, w_range=(0.8, 5.5)):
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
    one_sided = rng.random() < 0.4
    side = rng.choice([-1, 1]) if one_sided else 0
    # band: halo is e^-(4.5^2/2)=4e-5 of peak at 4.5*sh; beyond that nothing visible
    sc, sh, A = sc.astype(np.float32), sh.astype(np.float32), A.astype(np.float32)
    tmax = np.sqrt(2 * np.log(max(float(A.max()) * (1 + rh) / 0.3, 1.001)))
    half = int(min(W, np.ceil(tmax * sh.max() / cos_t.min()) + 2))
    cx = np.round(xc).astype(np.int64)
    offs = np.arange(-half, half + 1)
    cols = cx[:, None] + offs[None, :]
    ok = (cols >= 0) & (cols < W)
    colsc = np.clip(cols, 0, W - 1)
    d = (colsc - xc[:, None]).astype(np.float32) * cos_t[:, None]
    prof = np.exp(-d ** 2 / (2 * sc[:, None] ** 2))
    halo = np.float32(rh) * np.exp(-d ** 2 / (2 * sh[:, None] ** 2))
    if one_sided:
        halo = halo * (1 / (1 + np.exp(np.clip(-side * d / 1.5, -50, 50))))
    v = ((prof + halo) * A[:, None] * ok).astype(np.float32)
    out = np.zeros((H, W, 3), np.float32)
    rows = np.arange(H)[:, None]
    # columns within a row are unique except clipped edge duplicates (masked by ok), so assignment is exact
    contrib = v[:, :, None] * col[None, None, :]
    out[rows, colsc] = np.where(ok[:, :, None], contrib, out[rows, colsc])
    return out


def glare_layer(H, W, xc, rng):
    L = np.zeros((H, W, 3), np.float32)
    for _ in range(rng.integers(0, 4)):
        if rng.random() < 0.5:
            cy = rng.uniform(0, H); cx = float(np.interp(cy, np.arange(H), xc)) + rng.uniform(-60, 60)
        else:
            cy = rng.uniform(0, H); cx = rng.uniform(0, W)
        sy, sx = rng.uniform(8, 90), rng.uniform(8, 90)
        ang = rng.uniform(0, np.pi)
        amp = rng.uniform(40, 420)
        tint = np.array([1.0, rng.uniform(0.7, 1.0), rng.uniform(0.5, 1.0)], np.float32)
        r = int(4.5 * max(sx, sy)) + 2
        y0, y1 = max(0, int(cy) - r), min(H, int(cy) + r + 1)
        x0, x1 = max(0, int(cx) - r), min(W, int(cx) + r + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dy, dx = yy - cy, xx - cx
        u = dx * np.cos(ang) + dy * np.sin(ang); v = -dx * np.sin(ang) + dy * np.cos(ang)
        g = np.exp(-(u ** 2 / (2 * sx ** 2) + v ** 2 / (2 * sy ** 2)))
        L[y0:y1, x0:x1] += (amp * g)[:, :, None] * tint[None, None, :]
    return L


def render(bg, rng, p_empty=0.08, p_proc=0.4):
    H, W = bg.shape[:2]
    if rng.random() < p_proc:
        base = proc_bg(H, W, rng)
    else:
        base = bg.astype(np.float32) * np.float32(rng.uniform(0.4, 1.9))
        base *= (1 + rng.uniform(-0.15, 0.15, 3)).astype(np.float32)[None, None, :]
        if rng.random() < 0.5:
            f, m = colour_field(H, W, rng)
            base = base * (1 - m) + f * m * rng.uniform(0.3, 1.0)
    xc = make_curve(H, W, rng)
    base += glare_layer(H, W, xc, rng)
    col = laser_colour(rng)
    A0 = float(np.exp(rng.uniform(np.log(30), np.log(900))))
    nd = int(rng.choice([0, 1, 2, 3, 4], p=[0.45, 0.2, 0.15, 0.1, 0.1]))
    for _ in range(nd):
        xg = make_curve(H, W, rng)
        if rng.random() < 0.6:
            xg = xc + rng.choice([-1, 1]) * rng.uniform(25, 220) + 0.3 * (xg - xg.mean())
        c2 = col if rng.random() < 0.8 else laser_colour(rng)
        base += stripe_layer(H, W, xg, rng, A0 * rng.uniform(0.05, 0.7), c2)
    base0 = np.clip(base, 0, 255)
    lab = np.full(H, -1, np.float32)
    occ = occluders(H, W, rng)
    ok_ = rng.uniform(0.02, 0.3)
    if rng.random() < p_empty:
        full = base0
    else:
        S = stripe_layer(H, W, xc, rng, A0, col)
        full = np.clip(base + S, 0, 255)
        xi = np.clip(np.round(xc).astype(int), 0, W - 1)
        cs = np.clip(xi[:, None] + np.arange(-2, 3)[None, :], 0, W - 1)
        r = np.arange(H)[:, None]
        vis = np.abs(full[r, cs] - base0[r, cs]).max(axis=(1, 2))
        cover = occ[r, np.clip(xi[:, None] + np.arange(-1, 2)[None, :], 0, W - 1)].mean(1)
        vis = vis * (1 - cover * (1 - ok_))
        inside = (xc >= 2) & (xc <= W - 3)
        lab = np.where(inside & (vis >= 25), xc, np.where(inside & (vis >= 12), np.nan, -1)).astype(np.float32)
    img = full * (1 - occ[:, :, None] * np.float32(1 - ok_))
    g = rng.uniform(0.8, 1.3)
    lut = (255 * (np.arange(256) / 255.0) ** g).astype(np.uint8)
    img = cv2.LUT(np.clip(img, 0, 255).astype(np.uint8), lut).astype(np.float32)
    sb = rng.uniform(0, 1.2)
    if sb > 0.2:
        img = cv2.GaussianBlur(img, (0, 0), sb)
    img += rng.standard_normal(img.shape, dtype=np.float32) * np.float32(rng.uniform(0.5, 6))
    img = np.clip(img, 0, 255).astype(np.uint8)
    enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 96))])[1]
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), lab
