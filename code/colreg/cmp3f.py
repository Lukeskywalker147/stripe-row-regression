import glob, time, numpy as np, cv2
cv2.setNumThreads(1)
import synth3, synth3f
from core import CH, CW
bgs = sorted(glob.glob("bg/*.jpg"))
tt = {"old": 0, "new": 0}; mad = []; lab_agree = []
for i in range(40):
    r = np.random.default_rng(100 + i)
    bg = cv2.imread(bgs[int(r.integers(len(bgs)))]); H, W = bg.shape[:2]
    y0 = int(r.integers(0, H - CH + 1)); x0 = int(r.integers(0, W - CW + 1)); b = np.ascontiguousarray(bg[y0:y0 + CH, x0:x0 + CW])
    out = {}
    for k, m in (("old", synth3), ("new", synth3f)):
        rr = np.random.default_rng(500 + i); t = time.time(); out[k] = m.render(b, rr); tt[k] += time.time() - t
    mad.append(np.abs(out["old"][0].astype(float) - out["new"][0]).mean())
    a, c = out["old"][1], out["new"][1]
    lab_agree.append(np.mean((np.isnan(a) == np.isnan(c)) & ((a >= 0) == (c >= 0)) | (np.isnan(a) & np.isnan(c))))
print("s/img old %.3f new %.3f" % (tt["old"] / 40, tt["new"] / 40))
print("mean abs pixel diff (0-255): median %.2f p90 %.2f max %.2f" % (np.median(mad), np.percentile(mad, 90), max(mad)))
print("label state agreement: min %.3f mean %.4f" % (min(lab_agree), np.mean(lab_agree)))
