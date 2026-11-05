import numpy as np, cv2, torch
from core3g import *
tr = MixDSG("train", n=64, p_synth=0.5, seed=5)
dl = torch.utils.data.DataLoader(tr, batch_size=16, num_workers=4, collate_fn=passthrough)
bt = next(iter(dl))
x, lr, lc = assemble(bt, "cuda")
print(x.shape, lr.shape, lc.shape, [it[0] for it in bt])
im = (x.cpu() * STD_T + MEAN_T).clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
tiles = []
for k in range(16):
    v = np.ascontiguousarray(im[k]); l = lr[k].cpu().numpy(); c = lc[k].cpu().numpy()
    for y in range(0, 512, 4):
        if l[y] >= 0: cv2.circle(v, (int(round(l[y])), y), 1, (0, 0, 255), -1)
    for xx in range(0, 512, 4):
        if c[xx] >= 0: cv2.circle(v, (xx, int(round(c[xx]))), 1, (0, 255, 0), -1)
    cv2.putText(v, bt[k][0], (8, 24), 0, 0.8, (255, 255, 255), 2)
    tiles.append(cv2.resize(v, (320, 320)))
cv2.imwrite("g_batch.jpg", np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, 16, 4)]))
