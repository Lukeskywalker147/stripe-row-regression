import sys, json, numpy as np, cv2
sys.path.insert(0, "LASER25_DIR")
from pathlib import Path
from reanalyze_steger import *
from vision.src.geometry import fit_plane_lstsq
IM = Path("LASER25_DIR/images")
nn = np.load("/tmp/nn_pts.npz")
K, dist = load_camera(); target = CharucoTarget(CharucoConfig.from_cfg(CFG))
M = ("steger", "v3", "v3g"); P = {k: {} for k in M}
for path in sorted(IM.glob("*.png")):
    name = path.stem; img = cv2.imread(str(path)); det = target.detect(img[:, :, 1])
    hull = cv2.convexHull(det.corners.astype(np.float32).reshape(-1, 1, 2))
    mask = np.zeros(img.shape[:2], np.uint8); cv2.fillConvexPoly(mask, np.rint(hull).astype(np.int32), 1)
    st = extract_laser_centers(img, CFG, image_mask_override=mask.astype(bool))
    used, rvec, tvec, _, _ = robust_pose(target, det, st, K, dist); plane = board_plane_in_camera(rvec, tvec)
    for k, c in {"steger": st, "v3": nn[f"{name}_v3"], "v3g": nn[f"{name}_v3g"]}.items():
        c = np.asarray(c, np.float64).reshape(-1, 2)[:, :2]
        c = c[mask[np.clip(np.rint(c[:, 1]).astype(int), 0, 1079), np.clip(np.rint(c[:, 0]).astype(int), 0, 1919)] > 0]
        pts, v = ray_plane_intersect_masked(pixels_to_rays(c, K, dist), plane); pts = pts[v]
        P[k][name] = pts[line_inliers_3d(pts)[0]]
names = sorted(P["steger"])


def fit(pts):
    pl, inl, _ = fit_plane_ransac(pts, threshold=.002, iters=3000)
    return fit_plane_lstsq(pts[inl]), inl


def stat(r): return f"RMS {np.sqrt(np.mean(r**2))*1000:.3f} med {np.median(np.abs(r))*1000:.3f} p95 {np.percentile(np.abs(r),95)*1000:.3f} mm"


planes = {}
print("== pooled fit (all 25 frames), residual of inliers")
for k in M:
    pts = np.vstack([P[k][n] for n in names]); pl, inl = fit(pts); planes[k] = pl
    print(f"{k:7s} pts {len(pts)} inl {inl.sum()} ({inl.mean():.0%})  {stat(plane_point_distance(pts[inl], pl))}")
print("\n== plane agreement between methods")
for a, b in (("steger", "v3"), ("v3", "v3g"), ("steger", "v3g")):
    n1, n2 = planes[a][:3], planes[b][:3]; n1 = n1 / np.linalg.norm(n1); n2 = n2 / np.linalg.norm(n2)
    ang = np.degrees(np.arccos(np.clip(abs(n1 @ n2), -1, 1)))
    cen = np.vstack([P["v3"][n] for n in names]).mean(0)
    dd = abs(plane_point_distance(cen[None], planes[a])[0] - plane_point_distance(cen[None], planes[b])[0]) * 1000
    print(f"{a} vs {b}: normal angle {ang:.3f} deg, offset at stripe centroid {dd:.3f} mm")

print("\n== 5-fold leave-poses-out: fit on 20 frames, evaluate ALL line-inlier points of the 5 held-out frames (no outlier removal on test)")
folds = [names[i::5] for i in range(5)]
held = {k: {j: [] for j in M} for k in M}   # held[fit][eval]
for f in folds:
    tr = [n for n in names if n not in f]
    for k in M:
        pl, _ = fit(np.vstack([P[k][n] for n in tr]))
        for j in M:
            held[k][j].append(plane_point_distance(np.vstack([P[j][n] for n in f]), pl))
print("rows = plane fitted from, cols = held-out points of (RMS mm / p95 mm)")
print("fit\\eval " + "  ".join(f"{j:>15s}" for j in M))
for k in M:
    print(f"{k:8s} " + "  ".join(f"{np.sqrt(np.mean(np.concatenate(held[k][j])**2))*1000:6.3f}/{np.percentile(np.abs(np.concatenate(held[k][j])),95)*1000:6.3f}" for j in M))
print("\n== per-frame held-out signed mean (mm) = plane offset bias per pose, spread across poses")
for k in M:
    m = []
    for f in folds:
        tr = [n for n in names if n not in f]; pl, _ = fit(np.vstack([P[k][n] for n in tr]))
        m += [plane_point_distance(P[k][n], pl).mean() * 1000 for n in f]
    print(k, f"mean-abs per-pose offset {np.mean(np.abs(m)):.3f}, std {np.std(m):.3f}")
print("\n== per-frame (held-out) steger vs v3: n pts, signed mean offset mm, RMS mm")
for f in folds:
    tr = [n for n in names if n not in f]
    ps, _ = fit(np.vstack([P["steger"][n] for n in tr])); pn, _ = fit(np.vstack([P["v3"][n] for n in tr]))
    for n in f:
        a = plane_point_distance(P["steger"][n], ps) * 1000; b = plane_point_distance(P["v3"][n], pn) * 1000
        if np.sqrt(np.mean(a**2)) > 2: print(n, f"steger n={len(a)} mean {a.mean():.1f} rms {np.sqrt(np.mean(a**2)):.1f} | v3 n={len(b)} mean {b.mean():.2f} rms {np.sqrt(np.mean(b**2)):.2f}")
