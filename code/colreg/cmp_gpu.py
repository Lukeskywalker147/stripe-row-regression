import sys, glob, time, numpy as np, cv2, torch
import synth3, synth_gpu as SG
from core import CH, CW
dev = "cuda"
bgs = sorted(glob.glob("bg/*.jpg")); N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
PP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
cache = {}
def load_bg(r):
    f = bgs[int(r.integers(len(bgs)))]
    if f not in cache: cache[f] = cv2.imread(f)
    b = cache[f]; H, W = b.shape[:2]
    y0 = int(r.integers(0, H - CH + 1)); x0 = int(r.integers(0, W - CW + 1))
    return np.ascontiguousarray(b[y0:y0 + CH, x0:x0 + CW])
def stats(img, lab):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = np.abs(np.diff(g, axis=1)).mean()
    return [g.mean(), g.std(), gx, img.reshape(-1, 3).mean(0)[0] - img.reshape(-1, 3).mean(0)[2],
            (lab >= 0).mean(), np.isnan(lab).mean(), (img.max(2) >= 250).mean(), img.max()]
names = ["gray_mean", "gray_std", "grad_x", "B-R", "frac_present", "frac_nan", "frac_sat", "max"]
C, G = [], []
t = time.time()
for i in range(N):
    rng = np.random.default_rng((1, i)); img, lab = synth3.render(load_bg(rng), rng, p_proc=PP); C.append(stats(img, lab))
tc = (time.time() - t) / N
plans = [SG.plan_render(np.random.default_rng((2, i)), load_bg, p_proc=PP) for i in range(N)]
def run(ps):
    U = SG.Uploader(dev)
    for p in ps: U.walk(p)
    U.flush()
    o = [SG.compose(p, dev, U) for p in ps]
    lab = torch.stack([x[1] for x in o]).cpu()
    im = SG.jpeg_roundtrip([x[0] for x in o], [p["q"] for p in ps], dev)
    torch.cuda.synchronize()
    return list(zip(im, lab))
run(plans[:4]); run(plans[4:8])
t = time.time()
outs = []
for i in range(0, N - N % 4, 4): outs += run(plans[i:i + 4])
tg = (time.time() - t) / len(outs)
t = time.time(); [SG.plan_render(np.random.default_rng((3, i)), load_bg) for i in range(50)]; tp = (time.time() - t) / 50
sheet = []
for im, lab in outs:
    a = im.permute(1, 2, 0).cpu().numpy(); l = lab.cpu().numpy(); G.append(stats(a, l))
    if len(sheet) < 12:
        v = a.copy()
        for y in range(0, 512, 4):
            if l[y] >= 0: cv2.circle(v, (int(round(l[y])), y), 1, (0, 0, 255), -1)
        sheet.append(cv2.resize(v, (384, 256)))
C, G = np.array(C), np.array(G)
print(f"cpu render {tc*1000:.0f} ms/img | gpu compose {tg*1000:.1f} ms/img | plan {tp*1000:.1f} ms/img (cache warm)")
print("%-13s %10s %10s   | %10s %10s" % ("stat", "cpu mean", "gpu mean", "cpu med", "gpu med"))
for k, n in enumerate(names):
    print("%-13s %10.3f %10.3f   | %10.3f %10.3f" % (n, C[:, k].mean(), G[:, k].mean(), np.median(C[:, k]), np.median(G[:, k])))
cv2.imwrite("gpu_sheet.jpg", np.vstack([np.hstack(sheet[i:i + 4]) for i in range(0, 12, 4)]))
