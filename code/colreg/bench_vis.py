import sys
import numpy as np, cv2
import bench_synth as B

names = list(B.AXES)
for part, ax in enumerate((names[:6], names[6:])):
    rows = []
    for a in ax:
        if a == "background" and not __import__("glob").glob("bg/*.jpg"):
            B.AXES[a] = B.AXES[a][:4] + B.AXES[a][3:4]
        tiles = []
        for lv in range(5):
            img, lab = B.sample(a, lv, 0)
            v = img.copy()
            for y in range(0, B.H, 4):
                if lab[y] >= 0:
                    cv2.circle(v, (int(round(lab[y])), y), 1, (0, 0, 255), -1)
            t = cv2.resize(v, (384, 256))
            cv2.putText(t, f"{a}:{lv}", (6, 18), 0, 0.55, (255, 255, 255), 2)
            cv2.putText(t, f"{a}:{lv}", (6, 18), 0, 0.55, (0, 0, 0), 1)
            tiles.append(t)
        rows.append(np.hstack(tiles))
    cv2.imwrite(f"bench_vis{part}.jpg", np.vstack(rows))
