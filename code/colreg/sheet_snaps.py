import glob, os, cv2, numpy as np
fs = sorted(glob.glob(os.path.expanduser("~/stripe_seg/cam_snaps/*.png")))
t = []
for f in fs:
    im = cv2.resize(cv2.imread(f), (400, 225)); cv2.putText(im, os.path.basename(f)[-10:-4], (6, 20), 0, 0.6, (0, 255, 255), 2); t.append(im)
cv2.imwrite("snaps_all.jpg", np.vstack([np.hstack(t[i:i + 4]) for i in range(0, 16, 4)]))
