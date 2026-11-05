import sys, glob, os, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer import RowRegressor
ckpt, src, out = sys.argv[1:4]
m = RowRegressor(ckpt)
res = {}
for p in sorted(glob.glob(src + "/*.jpg")):
    img = cv2.imread(p)
    res[os.path.basename(p)] = m.extract(img)
np.savez_compressed(out, **res)
print(len(res), "frames")
