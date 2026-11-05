import glob, os, cv2, numpy as np
from infer import RowRegressor, OrientRegressor
R = os.path.expanduser("~/stripe_seg")
wood = sorted(glob.glob(f"{R}/data_scan_v3/val/images/*.jpg")); wood = [wood[i] for i in np.linspace(5, len(wood) - 6, 5).astype(int)]
snap = lambda ids: [f"{R}/cam_snaps/snap_20261030_{i}.png" for i in ids]
cats = dict(wood=wood, helmet=snap(["115400", "115425", "115435", "115439", "115443"]), cup=snap(["115458", "115501", "115503", "115505", "115507"]))
m2 = RowRegressor("runs/r18_v2/last.pt"); m3 = OrientRegressor("runs/r18_v3/last.pt"); m3g = OrientRegressor("runs/r18_v3g/last.pt")
def back(p, H):
    p = np.asarray(p, np.float32).reshape(-1, 2)
    return np.stack([p[:, 1], H - 1 - p[:, 0]], 1) if len(p) else p
for cat, fs in cats.items():
    rows = []
    for f in fs:
        im = cv2.imread(f); H = im.shape[0]
        r = im if cat == "wood" else cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        outs = [("v2", m2.extract(r)[:, :2] if cat == "wood" else back(m2.extract(r)[:, :2], H)), ("v3", m3.extract(im)[:, :2]), ("v3g", m3g.extract(im)[:, :2])]
        tiles = []
        for n, P in outs:
            v = im.copy()
            for x, y in P[::2]: cv2.circle(v, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)
            t = cv2.resize(v, (640, int(640 * H / im.shape[1])), interpolation=cv2.INTER_AREA)
            cv2.putText(t, f"{n} {os.path.basename(f)[-10:-4]} pts={len(P)}", (6, 22), 0, 0.65, (0, 255, 255), 2); tiles.append(t)
        rows.append(np.hstack(tiles)); print(cat, os.path.basename(f), [len(P) for _, P in outs], flush=True)
    cv2.imwrite(f"cat_{cat}.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
