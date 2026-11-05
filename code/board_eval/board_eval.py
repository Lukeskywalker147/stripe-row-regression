import sys, json, numpy as np, cv2
sys.path.insert(0, "LASER25_DIR")
from pathlib import Path
from reanalyze_steger import *
IM = Path("LASER25_DIR/images")
nn = np.load("/tmp/nn_pts.npz")
K, dist = load_camera(); target = CharucoTarget(CharucoConfig.from_cfg(CFG))


def line_fit(P, thr=0.002):
    inl, _ = line_inliers_3d(P, thr)
    p = P[inl]; o = p.mean(0); d = np.linalg.svd(p - o, full_matrices=False)[2][0]
    return o, d, inl


rows = []; allres = {k: [] for k in ("steger", "v3", "v3g")}; alled = {k: [] for k in allres}
for path in sorted(IM.glob("*.png")):
    name = path.stem; img = cv2.imread(str(path))
    det = target.detect(img[:, :, 1])
    hull = cv2.convexHull(det.corners.astype(np.float32).reshape(-1, 1, 2))
    mask = np.zeros(img.shape[:2], np.uint8); cv2.fillConvexPoly(mask, np.rint(hull).astype(np.int32), 1)
    st = extract_laser_centers(img, CFG, image_mask_override=mask.astype(bool))
    used, rvec, tvec, pnp_err, _ = robust_pose(target, det, st, K, dist)
    plane = board_plane_in_camera(rvec, tvec); Rm = cv2.Rodrigues(rvec)[0]; t = tvec.reshape(3)
    sets = {"steger": st, "v3": nn[f"{name}_v3"], "v3g": nn[f"{name}_v3g"]}
    rec = {"pose": name}
    for k, c in sets.items():
        c = np.asarray(c, np.float64).reshape(-1, 2)[:, :2]
        ins = mask[np.clip(np.rint(c[:, 1]).astype(int), 0, img.shape[0] - 1), np.clip(np.rint(c[:, 0]).astype(int), 0, img.shape[1] - 1)] > 0
        c = c[ins]
        pts, valid = ray_plane_intersect_masked(pixels_to_rays(c, K, dist), plane); pts = pts[valid]
        o, d, inl = line_fit(pts)
        v = pts - o; along = v @ d; res = np.linalg.norm(v - along[:, None] * d, axis=1)
        Xb = (pts - t.reshape(1, 3)) @ Rm
        de = np.abs(((Xb[:, :2] + 0.01) % 0.02) - 0.01).min(1)
        g = np.sort(along[inl]); span = g[-1] - g[0]
        bins = np.floor((along[inl] - g[0]) / 0.002).astype(int); occ = len(np.unique(bins)) / (span / 0.002 + 1)
        rec[k] = dict(n=int(len(pts)), inl=int(inl.sum()), inl_frac=float(inl.mean()), rms_mm=float(np.sqrt(np.mean(res[inl] ** 2)) * 1000),
                      span_mm=float(span * 1000), maxgap_mm=float(np.max(np.diff(g)) * 1000), occ=float(occ))
        allres[k].append(res[inl]); alled[k].append(de[inl])
    rows.append(rec)
json.dump(rows, open("board_eval.json", "w"), indent=1)
print("mean over frames: inl_frac, rms_mm, span_mm, maxgap_mm, 2mm-bin occupancy, inliers/frame")
for k in allres:
    print(k, *[f"{np.mean([r[k][m] for r in rows]):.3f}" for m in ("inl_frac", "rms_mm", "span_mm", "maxgap_mm", "occ", "inl")])
print("\ninlier residual RMS (mm) by distance to nearest checker edge")
edges = [0, 0.0005, 0.001, 0.002, 0.005, 0.011]
for k in allres:
    r = np.concatenate(allres[k]); e = np.concatenate(alled[k]); out = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (e >= a) & (e < b); out.append(f"[{a*1000:.1f},{b*1000:.1f}) n={s.sum()} rms={np.sqrt(np.mean(r[s]**2))*1000:.2f}")
    print(k, " | ".join(out))
