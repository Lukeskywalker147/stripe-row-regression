import numpy as np, cv2, glob
import synth
from core import shear_crop, CH, CW, ROOT

bgs = sorted(glob.glob("bg/*.jpg")); rng = np.random.default_rng(11); T = []
d = np.load("labels/train.npz")
for i in range(8):
    if i < 4:
        bg = cv2.imread(bgs[i * 9])
        img, lab = synth.render(np.ascontiguousarray(bg[250:250 + 512, 450:450 + 768]), rng)
    else:
        k = i * 50
        img = cv2.imread(f"{ROOT}/train/images/{d['names'][k]}")
        img, lab = shear_crop(img, d["labels"][k], rng, [-1.0, -0.6, 0.6, 1.0][i - 4])
    v = img.copy()
    for y in range(0, 512, 3):
        if lab[y] >= 0:
            cv2.circle(v, (int(round(lab[y])), y), 1, (0, 0, 255), -1)
    T.append(cv2.resize(v, (384, 256)))
cv2.imwrite("vis2.jpg", np.vstack([np.hstack(T[:4]), np.hstack(T[4:])]))
