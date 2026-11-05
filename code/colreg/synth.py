"""Synthetic near-vertical laser-stripe renderer with exact per-row sub-pixel ground truth.

Label per row: x>=0 center, -1 absent (invisible / not present), NaN ignore.
Assumption baked in: the true center is the geometric symmetric center of the core;
halos may be one-sided but do not move the label.
v2: slopes up to ~1.4, arcs/rim kinks, colour diversity, thin dark occluders (cables).
"""
import numpy as np, cv2


def smooth_noise(n, sigma, rng):
    k = int(max(3, sigma * 6)) | 1
    z = cv2.GaussianBlur(rng.standard_normal((1, n + 2 * k)).astype(np.float32), (k, 1), sigma)[0, k:k + n]
    return z / (z.std() + 1e-6)


def make_curve(H, W, rng):
    y = np.arange(H, dtype=np.float32)
    steep = rng.random() < 0.5
    slope_max = 1.4 if steep else 0.4
    s0 = rng.uniform(-slope_max, slope_max)
    x = rng.uniform(0.2 * W, 0.8 * W) + s0 * (y - H / 2)
    kmax = 1.2 if steep else 0.5
    for _ in range(rng.integers(0, 3)):
        yk = rng.uniform(0, H)
        x = x + rng.uniform(-kmax, kmax) * np.maximum(0, y - yk)
    if rng.random() < 0.4:
        y0 = rng.uniform(0, H); a = rng.uniform(-1, 1) * (0.9 / H)
        x = x + a * (y - y0) ** 2
    rounded = np.exp(rng.uniform(np.log(0.5), np.log(8.0)))
    x = cv2.GaussianBlur(x[None], (int(rounded * 6) | 1, 1), rounded)[0]
    if rng.random() < 0.3:
        yj = rng.integers(H // 6, 5 * H // 6)
        x = x + np.where(y >= yj, rng.choice([-1, 1]) * rng.uniform(15, 70), 0)
    x = x + 0.6 * smooth_noise(H, rng.uniform(4, 20), rng) * rng.uniform(0, 1)
    return x.astype(np.float32)


def stripe_layer(H, W, xc, rng, strength=1.0, w_scale=1.0):
    xx = np.arange(W, dtype=np.float32)[None, :]
    slope = np.gradient(xc)
    cos_t = 1.0 / np.sqrt(1 + slope ** 2)
    d = (xx - xc[:, None]) * cos_t[:, None]
    sc0 = np.exp(rng.uniform(np.log(0.9), np.log(3.2))) * w_scale
    sc = sc0 * np.clip(1 + 0.35 * smooth_noise(H, rng.uniform(10, 60), rng), 0.5, 2.5)
    A0 = np.exp(rng.uniform(np.log(60), np.log(700))) * strength
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
    rh = rng.uniform(0, 0.5)
    sh = sc * rng.uniform(2.5, 6.0)
    halo = rh * np.exp(-d ** 2 / (2 * sh[:, None] ** 2))
    if rng.random() < 0.4:
        side = rng.choice([-1, 1])
        halo = halo * (1 / (1 + np.exp(-side * d / 1.5)))
    p = (prof + halo) * A[:, None]
    if rng.random() < 0.35:
        col = np.array([1.0, rng.uniform(0.7, 1.0), rng.uniform(0.4, 0.95)], np.float32)   # near-white / cyan core
    else:
        col = np.array([1.0, rng.uniform(0.3, 0.9), rng.uniform(0.05, 0.5)], np.float32)
    return p[:, :, None] * col[None, None, :], A


def glare_layer(H, W, xc, rng):
    L = np.zeros((H, W, 3), np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for _ in range(rng.integers(0, 4)):
        if rng.random() < 0.5:
            cy = rng.uniform(0, H); cx = float(np.interp(cy, np.arange(H), xc)) + rng.uniform(-60, 60)
        else:
            cy = rng.uniform(0, H); cx = rng.uniform(0, W)
        sy, sx = rng.uniform(8, 90), rng.uniform(8, 90)
        ang = rng.uniform(0, np.pi)
        dy, dx = yy - cy, xx - cx
        u = dx * np.cos(ang) + dy * np.sin(ang); v = -dx * np.sin(ang) + dy * np.cos(ang)
        g = np.exp(-(u ** 2 / (2 * sx ** 2) + v ** 2 / (2 * sy ** 2)))
        amp = rng.uniform(40, 420)
        tint = np.array([1.0, rng.uniform(0.7, 1.0), rng.uniform(0.5, 1.0)], np.float32)
        L += (amp * g)[:, :, None] * tint[None, None, :]
    return L


def colour_field(H, W, rng):
    """Smooth coloured surface (yellow plastic / white / dark shell etc.) blended with the real background."""
    c0 = rng.uniform(0, 255, 3).astype(np.float32)
    if rng.random() < 0.5:
        c0 = np.array([rng.uniform(0, 60), rng.uniform(150, 240), rng.uniform(180, 255)], np.float32)   # yellow-ish (BGR)
    c1 = np.clip(c0 * rng.uniform(0.4, 1.3), 0, 255)
    ang = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = (np.cos(ang) * (xx - W / 2) + np.sin(ang) * (yy - H / 2)) / (0.6 * max(H, W))
    t = 1 / (1 + np.exp(-t * rng.uniform(1, 6)))
    f = c0[None, None, :] * (1 - t[:, :, None]) + c1[None, None, :] * t[:, :, None]
    cy, cx = rng.uniform(0, H), rng.uniform(0, W); sy, sx = rng.uniform(80, 400), rng.uniform(80, 500)
    m = np.exp(-(((yy - cy) / sy) ** 4 + ((xx - cx) / sx) ** 4))
    return f, m[:, :, None]


def occluders(H, W, rng):
    occ = np.zeros((H, W), np.float32)
    for _ in range(rng.integers(0, 4)):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        ang = rng.uniform(0, 2 * np.pi); pts = [(x, y)]
        for _ in range(rng.integers(3, 8)):
            ang += rng.normal(0, 0.35); x += 60 * np.cos(ang); y += 60 * np.sin(ang)
            pts.append((x, y))
        cv2.polylines(occ, [np.array(pts, np.int32)], False, 1.0, int(rng.integers(2, 8)), cv2.LINE_AA)
    return occ


def render(bg, rng, p_empty=0.08):
    """bg: HxWx3 uint8 BGR crop. Returns img uint8, labels float32 (H,)."""
    H, W = bg.shape[:2]
    base = bg.astype(np.float32) * rng.uniform(0.5, 1.4)
    base = base * (1 + rng.uniform(-0.15, 0.15, 3)[None, None, :])
    if rng.random() < 0.6:
        f, m = colour_field(H, W, rng)
        base = base * (1 - m) + f * m * rng.uniform(0.3, 1.0)
    xc = make_curve(H, W, rng)
    base = base + glare_layer(H, W, xc, rng)
    for _ in range(rng.integers(0, 3)):
        xg = make_curve(H, W, rng)
        if rng.random() < 0.6:
            xg = xc + rng.choice([-1, 1]) * rng.uniform(25, 220) + 0.3 * (xg - xg.mean())
        S, _ = stripe_layer(H, W, xg, rng, strength=rng.uniform(0.05, 0.45))
        base = base + S
    base0 = np.clip(base, 0, 255)
    empty = rng.random() < p_empty
    lab = np.full(H, -1, np.float32)
    occ = occluders(H, W, rng)
    ok_ = rng.uniform(0.02, 0.3)
    if empty:
        full = base0
    else:
        S, A = stripe_layer(H, W, xc, rng)
        full = np.clip(base + S, 0, 255)
        diff = np.abs(full - base0).max(2)
        xi = np.clip(np.round(xc).astype(int), 0, W - 1)
        cols = np.clip(xi[:, None] + np.arange(-2, 3)[None, :], 0, W - 1)
        vis = np.take_along_axis(diff, cols, 1).max(1)
        cover = np.take_along_axis(occ, np.clip(xi[:, None] + np.arange(-1, 2)[None, :], 0, W - 1), 1).mean(1)
        vis = vis * (1 - cover * (1 - ok_))
        inside = (xc >= 2) & (xc <= W - 3)
        lab = np.where(inside & (vis >= 25), xc, np.where(inside & (vis >= 12), np.nan, -1)).astype(np.float32)
    img = full * (1 - occ[:, :, None] * (1 - ok_))
    g = rng.uniform(0.8, 1.3)
    img = 255 * (img / 255) ** g
    sb = rng.uniform(0, 1.2)
    if sb > 0.2:
        img = cv2.GaussianBlur(img, (0, 0), sb)
    img = img + rng.standard_normal(img.shape).astype(np.float32) * rng.uniform(0.5, 4)
    img = np.clip(img, 0, 255).astype(np.uint8)
    q = int(rng.integers(55, 96))
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img, lab
