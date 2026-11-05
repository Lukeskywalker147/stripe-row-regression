import numpy as np, cv2, glob, time
import synth
bgs = sorted(glob.glob("bg/*.jpg"))
rng = np.random.default_rng(5)
tiles = []
t=time.time()
for i in range(8):
    bg = cv2.imread(bgs[i*7]); y0=300; x0=500
    img, lab = synth.render(np.ascontiguousarray(bg[y0:y0+512, x0:x0+768]), rng)
    v = img.copy()
    for y in range(0, 512, 4):
        if lab[y] >= 0: cv2.circle(v, (int(round(lab[y])), y), 1, (0,0,255), -1)
        elif lab[y] == -1: cv2.circle(v, (6, y), 1, (0,255,0), -1)
    tiles.append(cv2.resize(v, (384, 256)))
print("t/img", (time.time()-t)/8)
g = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
cv2.imwrite("synth_vis.jpg", g)
