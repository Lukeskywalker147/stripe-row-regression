import sys, glob, os, numpy as np, cv2
sys.path.insert(0, os.path.expanduser("~/stripe_seg/colreg")); os.chdir(os.path.expanduser("~/stripe_seg/colreg"))
from infer import OrientRegressor
m = OrientRegressor("runs/r18_v3/last.pt")
out = {}
for f in sorted(glob.glob(os.path.expanduser("~/scan_v03/frame_*.jpg"))):
    p = m.extract(cv2.imread(f))[:, :2]
    out[os.path.basename(f)] = np.asarray(p, np.float64)
np.savez_compressed(os.path.expanduser("~/scan_v03_nn.npz"), **out); print(len(out), np.mean([len(v) for v in out.values()]))
