"""Score stripe extractors against hand-labelled polylines (cam_snaps, rotated-90 pass).
python eval_hand.py LABEL_DIR IMG_DIR OUT_DIR
Hand clicks are only ~2-3 px accurate, so only coarse distances are meaningful (no sub-pixel claims).
"""
import sys, os, glob, json
import numpy as np, cv2
from infer import RowRegressor, OrientRegressor
from labels import CFG
from laser_ref import extract_centers

LAB, IMG, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
m1, m2 = RowRegressor("runs/r18_scratch_s0/last.pt"), RowRegressor("runs/r18_v2/last.pt")
m3 = OrientRegressor("runs/r18_v3/last.pt")
m3b = OrientRegressor("runs/r18_v3/last.pt", mode="both")


def back(p, H):
    p = np.asarray(p, np.float32).reshape(-1, p.shape[1] if len(p) else 2)
    return np.stack([p[:, 1], H - 1 - p[:, 0]], 1) if len(p) else np.zeros((0, 2), np.float32)


def seg_dist(P, a, b):
    ab = b - a; L2 = float((ab ** 2).sum()) + 1e-9
    t = np.clip(((P - a) @ ab) / L2, 0, 1)
    return np.hypot(*(P - (a + t[:, None] * ab)).T)


def dist_to_poly(P, segs):
    d = np.full(len(P), np.inf)
    for s in segs:
        s = np.asarray(s, np.float32)
        for a, b in zip(s[:-1], s[1:]):
            d = np.minimum(d, seg_dist(P, a, b))
    return d


def sample_poly(segs, step=2.0):
    out = []
    for s in segs:
        s = np.asarray(s, np.float32)
        for a, b in zip(s[:-1], s[1:]):
            n = max(2, int(np.hypot(*(b - a)) / step))
            out.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
        out.append(s[-1:])
    return np.concatenate(out, 0)


res = {k: dict(prec=[], cov=[], covd=[], n=0, near=0) for k in ("v1", "v2", "v3", "v3both", "steger")}
rows = []
for jf in sorted(glob.glob(f"{LAB}/*.json")):
    d = json.load(open(jf)); segs = [s for s in d["segments"] if len(s) >= 2]
    if not segs:
        continue
    name = d["image"]; p = f"{IMG}/{name}.png"
    img = cv2.imread(p if os.path.exists(p) else f"{IMG}/{name}.jpg"); H = img.shape[0]
    r = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    preds = {"v1": back(m1.extract(r)[:, :2], H), "v2": back(m2.extract(r)[:, :2], H),
             "v3": m3.extract(img)[:, :2], "v3both": m3b.extract(img)[:, :2], "steger": back(np.asarray(extract_centers(r, CFG, None)).reshape(-1, 2)[:, :2], H)}
    G = sample_poly(segs)
    vis = img.copy()
    for s in segs:
        cv2.polylines(vis, [np.round(np.asarray(s)).astype(np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA)
    line = [name]
    for k, P in preds.items():
        # points near the labelled span count towards precision; the rest are "elsewhere" (labels may be incomplete)
        dp = dist_to_poly(P, segs) if len(P) else np.zeros(0)
        near = dp <= 15
        res[k]["prec"].append(dp[near]); res[k]["n"] += len(P); res[k]["near"] += int(near.sum())
        if len(P):
            dg = np.array([np.hypot(*(P - g).T).min() for g in G])
        else:
            dg = np.full(len(G), np.inf)
        res[k]["cov"].append(dg <= 5); res[k]["covd"].append(dg[dg <= 15])
        line.append(f"{k}: pts {len(P)} near {int(near.sum())} cov5 {np.mean(dg <= 5):.2f}")
        col = {"v1": (0, 0, 255), "v2": (0, 255, 0), "v3": (255, 200, 0), "v3both": (0, 200, 255), "steger": (255, 0, 255)}[k]
        for x, y in P[::3]:
            cv2.circle(vis, (int(round(x)), int(round(y))), 1, col, -1)
    print(" | ".join(line))
    cv2.imwrite(f"{OUT}/{name}_hand.jpg", vis)

print("\nmethod   n_pts  frac_pts_within15px_of_label  median_dist(px)  p90   coverage(label pts with pred within 5px)  median_dist_cov")
for k, v in res.items():
    pr = np.concatenate(v["prec"]); cv = np.concatenate(v["cov"]); cd = np.concatenate(v["covd"])
    print(f"{k:7s} {v['n']:6d}  {v['near'] / max(v['n'], 1):.2f}                          {np.median(pr):5.2f}            {np.percentile(pr, 90):5.2f}  {cv.mean():.2f}                                  {np.median(cd):.2f}")
