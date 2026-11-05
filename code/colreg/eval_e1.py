"""E1: synthetic held-out (exact GT) + real val (Steger labels). Compares NN(+Viterbi) with system Steger."""
import argparse, glob, os, sys, json
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth
from core import CH, CW, ROOT, crop_with_labels
from infer import RowRegressor
from labels import CFG
from laser_ref import extract_centers

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--encoder", default="resnet18")
a = ap.parse_args()
D = os.path.dirname(os.path.abspath(__file__))
nn_ = RowRegressor(a.ckpt, a.encoder)
bgs = sorted(glob.glob(f"{D}/bg/*.jpg"))


def score(pts, lab):
    """pts (N,2+) (u,v). returns per-row errors on present rows (inf if no output), fp rate on absent rows."""
    H = len(lab)
    byrow = {}
    for p in pts:
        byrow.setdefault(int(round(p[1])), []).append(p[0])
    err = []; fp = []
    for y in range(H):
        if np.isnan(lab[y]): continue
        us = byrow.get(y, [])
        if lab[y] >= 0:
            err.append(min([abs(u - lab[y]) for u in us]) if us else np.inf)   # best-of-row: favours the baseline
        else:
            fp.append(1.0 if us else 0.0)
    return np.array(err), np.array(fp)


def summarize(E, FP):
    E = np.concatenate(E); FP = np.concatenate(FP)
    fin = E[np.isfinite(E)]
    return dict(median=float(np.median(fin)), p95=float(np.percentile(fin, 95)), within1px=float((E <= 1).mean()),
                gross_gt5px=float((E > 5).mean()), missing=float(np.isinf(E).mean()), fp_on_absent=float(FP.mean()))


res = {}
# ---- synthetic held-out
rng = np.random.default_rng(777)
acc = {"steger": ([], []), "nn_argmax": ([], []), "nn_viterbi": ([], [])}
for i in range(a.n):
    bg = cv2.imread(bgs[int(rng.integers(len(bgs)))])
    H, W = bg.shape[:2]; y0 = int(rng.integers(0, H - CH)); x0 = int(rng.integers(0, W - CW))
    img, lab = synth.render(np.ascontiguousarray(bg[y0:y0 + CH, x0:x0 + CW]), rng)
    for k, pts in (("steger", extract_centers(img, CFG, None)),
                   ("nn_argmax", nn_.extract(img, viterbi=False)[:, :2]),
                   ("nn_viterbi", nn_.extract(img)[:, :2])):
        e, f = score(np.asarray(pts).reshape(-1, 2), lab)
        acc[k][0].append(e); acc[k][1].append(f)
res["synthetic"] = {k: summarize(*v) for k, v in acc.items()}
# ---- real val (labels are Steger-derived: circular for steger, only checks agreement/robustness)
d = np.load(f"{D}/labels/val.npz"); names, labs = d["names"], d["labels"]
acc = {"steger": ([], []), "nn_argmax": ([], []), "nn_viterbi": ([], [])}
rng = np.random.default_rng(1)
for k_ in range(0, len(names), 3):
    img = cv2.imread(f"{ROOT}/val/images/{names[k_]}")
    img, lab = crop_with_labels(img, labs[k_], rng)
    for k, pts in (("steger", extract_centers(img, CFG, None)),
                   ("nn_argmax", nn_.extract(img, viterbi=False)[:, :2]),
                   ("nn_viterbi", nn_.extract(img)[:, :2])):
        e, f = score(np.asarray(pts).reshape(-1, 2), lab)
        acc[k][0].append(e); acc[k][1].append(f)
res["real_val_steger_labels"] = {k: summarize(*v) for k, v in acc.items()}
print(json.dumps(res, indent=1))
