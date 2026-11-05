import sys, numpy as np, cv2, torch
from core3 import *
ds = MixDS3("train", n=100, p_synth=float(sys.argv[1]) if len(sys.argv) > 1 else 0.5, seed=5)
tiles = []
for i in range(24):
    x, lr, lc = ds[i]
    img = (x.numpy().transpose(1, 2, 0) * STD + MEAN).clip(0, 255).astype(np.uint8).copy()
    for y in range(0, len(lr), 3):
        if lr[y] >= 0: cv2.circle(img, (int(lr[y]), y), 1, (0, 0, 255), -1)
    for xx in range(0, len(lc), 3):
        if lc[xx] >= 0: cv2.circle(img, (xx, int(lc[xx])), 1, (0, 255, 0), -1)
    t = cv2.resize(img, (256, 256))
    cv2.putText(t, f"R{int((lr>=0).sum())} C{int((lc>=0).sum())}", (4, 16), 0, 0.5, (255, 255, 255), 1)
    tiles.append(t)
cv2.imwrite("vis3.jpg", np.vstack([np.hstack(tiles[r * 6:(r + 1) * 6]) for r in range(4)]))
