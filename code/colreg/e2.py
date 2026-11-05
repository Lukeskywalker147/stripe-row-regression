import sys, glob, os, json, numpy as np, cv2, yaml
sys.path.insert(0, os.path.expanduser("~/calib-and-reconstruction"))
from vision_core.laser import extract_centers
from vision_core.io import load_intrinsic
from multiprocessing import Pool
cfg = yaml.safe_load(open(os.path.expanduser("~/Downloads/V03_150/frozen/extraction.yaml")))["laser"]
K, dist, _ = load_intrinsic(os.path.expanduser("~/Downloads/V03_150/frozen/camera_intrinsic.yaml"))
nnf = sys.argv[1]
NN = np.load(nnf)
names = sorted(NN.files)

def steger(n):
    return n, extract_centers(cv2.imread("v03/" + n), cfg)

def und(pts):
    if len(pts) == 0: return pts
    return cv2.undistortPoints(pts[:, :2].reshape(-1, 1, 2).astype(np.float64), K, dist, P=K).reshape(-1, 2)

def windows(c, W=41, step=20):
    """c: (N,2) (u,v) undistorted. per-window quadratic-fit rms of u(v) over full-coverage windows."""
    out = {}
    if len(c) < W: return out
    v = np.round(c[:, 1]).astype(int)
    byrow = {}
    for (u, vv), r in zip(c, v): byrow.setdefault(r, []).append(u)
    rows = sorted(byrow)
    r0, r1 = rows[0], rows[-1]
    for s in range(r0, r1 - W + 2, step):
        rr = [r for r in range(s, s + W) if r in byrow]
        if len(rr) >= W - 4 and all(len(byrow[r]) == 1 for r in rr):
            y = np.array(rr, float); x = np.array([byrow[r][0] for r in rr])
            p = np.polyfit(y - y.mean(), x, 2)
            out[s] = float(np.sqrt(np.mean((x - np.polyval(p, y - y.mean())) ** 2)))
    return out

if __name__ == "__main__":
    with Pool(8) as p: ST = dict(p.map(steger, names))
    rs, rn, both = [], [], []
    cov = dict(st=0, nn=0); disagree = []
    for n in names:
        cs = und(ST[n]); cn = und(NN[n])
        cov["st"] += len(cs); cov["nn"] += len(cn)
        ws, wn = windows(cs), windows(cn)
        for s in set(ws) & set(wn):
            rs.append(ws[s]); rn.append(wn[s])
        # per-row disagreement on rows both output (raw pixel coords)
        a = {int(round(v)): u for u, v in ST[n][:, :2]}
        b = {int(round(v)): u for u, v, *_ in NN[n]}
        d = [abs(a[r] - b[r]) for r in a if r in b]
        if d: disagree.append(d)
    rs, rn = np.array(rs), np.array(rn)
    d = np.concatenate(disagree)
    print("frames", len(names), "points steger", cov["st"], "nn", cov["nn"])
    print("windows common", len(rs))
    for k, r in (("steger", rs), ("nn", rn)):
        print(k, "quad-fit rms px: median %.3f mean %.3f p90 %.3f" % (np.median(r), r.mean(), np.percentile(r, 90)))
    print("frac windows nn<steger %.2f" % (rn < rs).mean())
    print("row disagreement px: median %.3f p90 %.3f frac>2px %.3f" % (np.median(d), np.percentile(d, 90), (d > 2).mean()))
