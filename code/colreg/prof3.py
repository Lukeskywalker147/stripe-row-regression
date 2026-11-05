import cProfile, pstats, glob, time, numpy as np, cv2
cv2.setNumThreads(1)
import synth3f as synth3
from core3 import *
import synth3f as synth3
bgs = sorted(glob.glob("bg/*.jpg"))
rng = np.random.default_rng(0)
def synth_n(n):
    for _ in range(n):
        bg = cv2.imread(bgs[int(rng.integers(len(bgs)))]); H, W = bg.shape[:2]
        y0 = int(rng.integers(0, H - CH + 1)); x0 = int(rng.integers(0, W - CW + 1))
        synth3.render(np.ascontiguousarray(bg[y0:y0 + CH, x0:x0 + CW]), rng)
t = time.time(); synth_n(3); print("warm", time.time() - t)
pr = cProfile.Profile(); t = time.time(); pr.enable(); synth_n(30); pr.disable()
print("synth s/img", (time.time() - t) / 30)
pstats.Stats(pr).sort_stats("tottime").print_stats(14)
ds = MixDS3("train", n=100, p_synth=0.0, seed=1)
t = time.time()
for i in range(20): ds[i]
print("real-path s/img", (time.time() - t) / 20)
pr = cProfile.Profile(); pr.enable()
for i in range(20): ds[i]
pr.disable(); pstats.Stats(pr).sort_stats("tottime").print_stats(8)
