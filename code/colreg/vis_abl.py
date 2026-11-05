import numpy as np, cv2, glob, os
import bench_synth as B
from infer import OrientRegressor
M = {"base": "runs/abl_base_s0/last.pt", "pre": "runs/abl_pre/last.pt", "seg": "runs/abl_seg/last.pt", "v3 16k": "runs/r18_v3/last.pt"}
models = {k: OrientRegressor(v) for k, v in M.items()}
cases = [("empty", 0, 0, "no stripe"), ("distractors", 3, 0, "3 distractors"), ("distractors", 3, 1, "3 distractors"),
         ("glare", 3, 0, "glare x4"), ("occlusion", 4, 0, "occlusion x5")]
imgs = []
for ax, lv, k, name in cases:
    im, lab = B.sample(ax, lv, k); imgs.append((im, lab, name))
for n in ("snap_20261030_115403", "snap_20261030_115507"):
    p = f"{os.path.expanduser('~')}/stripe_seg/cam_snaps/{n}.png"
    im = cv2.imread(p)
    if im is not None: imgs.append((im, None, "real " + n[-6:]))
TW, TH = 440, 300
rows = []
for im, lab, name in imgs:
    tiles = []
    for mk, m in models.items():
        P = np.asarray(m.extract(im)[:, :2]).reshape(-1, 2)
        v = im.copy(); s = max(1, im.shape[1] // 768)
        if lab is not None:
            for y in range(0, len(lab), 3):
                if lab[y] >= 0: cv2.circle(v, (int(round(lab[y])), y), 2 * s, (0, 255, 0), -1)
        for x, y in P[::2]: cv2.circle(v, (int(round(x)), int(round(y))), s, (0, 0, 255), -1)
        info = f"{mk}: {len(P)} pts"
        if lab is not None:
            gt = np.full(len(lab), np.nan); 
            err = []
            for x, y in P:
                yi = int(round(y))
                if 0 <= yi < len(lab): err.append(np.nan if np.isnan(lab[yi]) else (abs(x - lab[yi]) if lab[yi] >= 0 else 99))
            err = np.array(err); bad = int((err > 5).sum()) if len(err) else 0
            info += f", >5px/false {bad}"
        t = cv2.resize(v, (TW, TH))
        cv2.rectangle(t, (0, 0), (TW, 22), (0, 0, 0), -1)
        cv2.putText(t, f"{name} | {info}", (4, 16), 0, 0.45, (255, 255, 255), 1)
        tiles.append(t)
    rows.append(np.hstack(tiles))
cv2.imwrite("abl_examples.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
