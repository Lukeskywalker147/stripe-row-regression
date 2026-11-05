import torch, time, glob, cv2, numpy as np
from core import build_model, row_logits, to_tensor
from infer import RowRegressor
dev = "cuda"
net = build_model("resnet18", None).to(dev).eval().to(memory_format=torch.channels_last)
net.load_state_dict(torch.load("runs/r18_v2/last.pt"))
def bench(bs, H, W, n=6):
    x = torch.randn(bs, 3, H, W, device=dev).to(memory_format=torch.channels_last)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(2): row_logits(net(x))
        torch.cuda.synchronize(); t = time.time()
        for _ in range(n): lg = torch.softmax(row_logits(net(x)).float(), 2)
        torch.cuda.synchronize()
    return (time.time() - t) / (n * bs)
for bs, H, W in [(1, 1088, 1920), (8, 1088, 1920), (8, 512, 768)]:
    print(f"GPU bf16 forward+softmax bs{bs} {H}x{W}: {bench(bs,H,W)*1000:.1f} ms/img")
f = sorted(glob.glob("snaps/*.jpg"))[:8]
t = time.time(); ims = [cv2.imread(p) for p in f]; print("imread ms/img", (time.time() - t) / 8 * 1000)
m = RowRegressor("runs/r18_v2/last.pt")
r = [cv2.rotate(i, cv2.ROTATE_90_CLOCKWISE) for i in ims]
m.extract(r[0])
t = time.time()
for i in r: m.probs(i)
print("probs (GPU+copy to cpu) ms/img", (time.time() - t) / 8 * 1000)
t = time.time()
for i in r: m.extract(i)
print("full extract incl Viterbi python ms/img", (time.time() - t) / 8 * 1000)
t = time.time()
for i in r: m.extract(i, viterbi=False)
print("full extract no Viterbi ms/img", (time.time() - t) / 8 * 1000)
