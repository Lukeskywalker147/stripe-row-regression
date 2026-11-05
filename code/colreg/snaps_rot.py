import sys, glob, os
import numpy as np, cv2
from infer import RowRegressor
m1, m2 = RowRegressor("runs/r18_scratch_s0/last.pt"), RowRegressor("runs/r18_v2/last.pt")
os.makedirs("snaps_v2r", exist_ok=True)
def run(m, img):
    H = img.shape[0]; r = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE); p = m.extract(r)
    return np.stack([p[:, 1], H - 1 - p[:, 0]], 1) if len(p) else np.zeros((0, 2))
tiles = []
for f in sorted(glob.glob("snaps/*.jpg")):
    img = cv2.imread(f); n = os.path.basename(f)[5:-4]
    out = []
    for m, c in ((m1, (0, 0, 255)), (m2, (0, 255, 0))):
        v = img.copy()
        for x, y in run(m, img): cv2.circle(v, (int(round(x)), int(round(y))), 2, c, -1)
        out.append(cv2.resize(v, (640, 360)))
    t = np.hstack(out); cv2.putText(t, n, (8, 22), 0, 0.7, (255, 255, 255), 2); tiles.append(t)
cv2.imwrite("snaps_v2r/sheet.jpg", np.vstack([np.hstack(tiles[i:i+1]) for i in (2, 7, 8, 9, 12, 14)]))
