"""Run v1 / v2 / Steger on the synthetic difficulty-axis benchmark.  python bench_eval.py [--n 12] [--out bench_out]
Per (axis, level): median err, within-1px, gross>5px, missing (all over GT-present rows, best-of-row),
false-output rate (outputs >5px from GT, or on GT-absent rows, / all outputs), fp on absent rows.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_synth as B
from infer import RowRegressor
from labels import CFG
from laser_ref import extract_centers

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--out", default="bench_out")
ap.add_argument("--tag", default="all")
ap.add_argument("--nosteger", action="store_true")
ap.add_argument("--models", nargs="*", default=["v1=runs/r18_scratch_s0/last.pt", "v2=runs/r18_v2/last.pt", "v3=runs/r18_v3/last.pt:orient", "v3both=runs/r18_v3/last.pt:orient_both"])
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
from infer import OrientRegressor
models = {}
for m in a.models:
    k, p = m.split("=")
    if ":orient" in p:
        path, spec = p.split(":", 1)
        models[k] = OrientRegressor(path, mode="both" if spec.endswith("both") else "auto")
    else:
        models[k] = RowRegressor(p)


def unrot(pts, k, H, W):
    """pts in coordinates of np.rot90(img, k) -> original coordinates"""
    sizes = [(H, W), (W, H), (H, W), (W, H)]
    pts = np.asarray(pts, np.float32).reshape(-1, 2).copy()
    for j in range(k, 0, -1):
        hB = sizes[j][0]
        pts = np.stack([hB - 1 - pts[:, 1], pts[:, 0]], 1)
    return pts


def methods(img, rot=0):
    H, W = img.shape[:2]
    im = np.ascontiguousarray(np.rot90(img, rot))
    out = {k: unrot(m.extract(im)[:, :2], rot, H, W) for k, m in models.items()}
    if not a.nosteger:
        out["steger"] = unrot(np.asarray(extract_centers(im, CFG, None)).reshape(-1, 2)[:, :2], rot, H, W)
    return out


def score(pts, lab):
    H = len(lab)
    gt = np.where(np.isnan(lab), np.nan, lab)
    rows = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1) if len(pts) else np.zeros(0, int)
    byrow = [[] for _ in range(H)]
    for r, u in zip(rows, pts[:, 0] if len(pts) else []):
        byrow[r].append(u)
    err, fp = [], []
    nout = nbad = 0
    for y in range(H):
        us = byrow[y]
        if np.isnan(gt[y]):
            continue                      # ambiguous rows ignored for everything
        nout += len(us)
        if gt[y] >= 0:
            d = [abs(u - gt[y]) for u in us]
            err.append(min(d) if d else np.inf)
            nbad += sum(x > 5 for x in d)
        else:
            fp.append(1.0 if us else 0.0); nbad += len(us)
    return np.array(err), np.array(fp), nout, nbad


def summ(rec):
    E = np.concatenate([r[0] for r in rec]); FP = np.concatenate([r[1] for r in rec]) if any(len(r[1]) for r in rec) else np.zeros(1)
    nout = sum(r[2] for r in rec); nbad = sum(r[3] for r in rec)
    fin = E[np.isfinite(E)]
    return dict(n_rows=int(len(E)), median=float(np.median(fin)) if len(fin) else None,
                p95=float(np.percentile(fin, 95)) if len(fin) else None,
                within1=float((E <= 1).mean()), gross5=float((E > 5).mean()), missing=float(np.isinf(E).mean()),
                false_out=float(nbad / max(nout, 1)), fp_absent=float(FP.mean()))


res = {}
for axis in list(B.AXES) + ["empty"]:
    nl = 1 if axis == "empty" else len(B.AXES[axis])
    res[axis] = []
    for lv in range(nl):
        acc = {}
        for k in range(a.n):
            img, lab = B.sample(axis, lv, k)
            for m, pts in methods(img, 0 if axis == 'empty' else B.AXES[axis][lv].get('rot', 0)).items():
                acc.setdefault(m, []).append(score(np.asarray(pts, np.float32).reshape(-1, 2), lab))
        row = {m: summ(r) if axis != "empty" else dict(false_pts_per_img=float(sum(x[2] for x in r) / len(r))) for m, r in acc.items()}
        res[axis].append(row)
        print(axis, lv, {m: (round(v["median"], 2) if v.get("median") is not None else None, round(v["gross5"], 3), round(v["false_out"], 3)) if "median" in v else v for m, v in row.items()}, flush=True)
json.dump(res, open(f"{a.out}/bench_{a.tag}.json", "w"), indent=1)
