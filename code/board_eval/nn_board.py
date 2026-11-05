import glob, os, numpy as np, cv2, sys
sys.path.insert(0, os.path.expanduser("~/stripe_seg/colreg")); os.chdir(os.path.expanduser("~/stripe_seg/colreg"))
from infer import OrientRegressor
ms = {"v3": OrientRegressor("runs/r18_v3/last.pt"), "v3g": OrientRegressor("runs/r18_v3g/last.pt")}
out = {}
for f in sorted(glob.glob(os.path.expanduser("~/board25/*.png"))):
    im = cv2.imread(f); n = os.path.basename(f)[:-4]
    for k, m in ms.items():
        out[f"{n}_{k}"] = np.asarray(m.extract(im)[:, :2], np.float64)
    print(n, {k: len(out[f"{n}_{k}"]) for k in ms}, flush=True)
np.savez(os.path.expanduser("~/board25/nn_pts.npz"), **out)
