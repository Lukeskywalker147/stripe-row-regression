"""Hand-label coverage for ablation runs (unrotated frames, orientation chosen by the model).  python abl_hand.py run1 run2 ..."""
import sys, os, glob, json
import numpy as np, cv2
from infer import OrientRegressor

IMG = os.path.expanduser("~/stripe_seg/cam_snaps"); LAB = "hand_labels"


def sample_poly(segs, step=2.0):
    out = []
    for s in segs:
        s = np.asarray(s, np.float32)
        for a, b in zip(s[:-1], s[1:]):
            n = max(2, int(np.hypot(*(b - a)) / step))
            out.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
        out.append(s[-1:])
    return np.concatenate(out, 0)


frames = []
for jf in sorted(glob.glob(f"{LAB}/*.json")):
    d = json.load(open(jf)); segs = [s for s in d["segments"] if len(s) >= 2]
    if segs:
        frames.append((cv2.imread(f"{IMG}/{d['image']}.png"), sample_poly(segs)))
res = {}
for r in sys.argv[1:]:
    m = OrientRegressor(f"runs/{r}/last.pt")
    cov, dd = [], []
    for img, G in frames:
        P = m.extract(img)[:, :2]
        dg = np.array([np.hypot(*(P - g).T).min() for g in G]) if len(P) else np.full(len(G), np.inf)
        cov.append(dg <= 5); dd.append(dg[dg <= 15])
    res[r] = dict(cov5=float(np.concatenate(cov).mean()), med=float(np.median(np.concatenate(dd))))
    print(r, res[r], flush=True)
json.dump(res, open("bench_out/abl_hand.json", "w"), indent=1)
