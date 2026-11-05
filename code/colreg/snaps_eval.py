"""cam_snaps: v1 (rotated, single pass) vs v2 (dual orientation). Overlays + continuity stats."""
import sys, glob, os, json
import numpy as np, cv2
from infer import RowRegressor, extract_dual

ck1, ck2, src, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
m1, m2 = RowRegressor(ck1), RowRegressor(ck2)


def chain_stats(pts, gap=6):
    """points sorted along the dominant axis; longest run with consecutive spacing <= gap, and #breaks."""
    if len(pts) < 2:
        return 0, 0, 0
    p = pts[:, :2]
    ax = 0 if np.ptp(p[:, 0]) >= np.ptp(p[:, 1]) else 1
    p = p[np.argsort(p[:, ax])]
    d = np.hypot(*np.diff(p, axis=0).T)
    br = d > gap
    runs = np.split(np.arange(len(p)), np.nonzero(br)[0] + 1)
    return len(p), int(br.sum()), max(len(r) for r in runs)


def draw(img, pts, col):
    v = img.copy()
    for x, y, _ in pts:
        cv2.circle(v, (int(round(x)), int(round(y))), 2, col, -1)
    return v


rows = []; stats = {}
for f in sorted(glob.glob(f"{src}/*.jpg")):
    img = cv2.imread(f); H, W = img.shape[:2]; n = os.path.basename(f)[:-4]
    r = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    p1 = m1.extract(r)
    p1 = np.stack([p1[:, 1], H - 1 - p1[:, 0], p1[:, 2]], 1) if len(p1) else np.zeros((0, 3), np.float32)
    p2 = extract_dual(m2, img)
    p2r = m2.extract(r)
    p2r = np.stack([p2r[:, 1], H - 1 - p2r[:, 0], p2r[:, 2]], 1) if len(p2r) else np.zeros((0, 3), np.float32)
    stats[n] = dict(v1=chain_stats(p1), v2_rot_only=chain_stats(p2r), v2_dual=chain_stats(p2))
    a, b = draw(img, p1, (0, 0, 255)), draw(img, p2, (0, 255, 0))
    t = np.hstack([cv2.resize(a, (960, 540)), cv2.resize(b, (960, 540))])
    cv2.putText(t, f"{n}  v1 (red)  |  v2 dual (green)", (10, 25), 0, 0.8, (255, 255, 255), 2)
    cv2.imwrite(f"{out}/{n}_cmp.jpg", t)
    rows.append(t)
print(json.dumps(stats, indent=0))
for k in ("v1", "v2_rot_only", "v2_dual"):
    a = np.array([s[k] for s in stats.values()], float)
    print(k, "mean pts", a[:, 0].mean(), "mean breaks", a[:, 1].mean(), "mean longest-run frac", (a[:, 2] / np.maximum(a[:, 0], 1)).mean())
