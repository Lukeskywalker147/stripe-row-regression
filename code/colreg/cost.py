import torch, time, os
from core import build_model
from torch.utils.flop_counter import FlopCounterMode
net = build_model("resnet18", None).eval()
n = sum(p.numel() for p in net.parameters())
print("params M", n / 1e6, "fp32 MB", n * 4 / 1e6, "int8 MB", n / 1e6)
torch.set_num_threads(4)
print("cpu", os.popen("lscpu | grep 'Model name'").read().strip(), "| threads", torch.get_num_threads())
for H, W in [(512, 768), (544, 960), (1088, 1920)]:
    x = torch.randn(1, 3, H, W)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        net(x)
    g = fc.get_total_flops() / 2e9
    with torch.no_grad():
        net(x)
        t = time.time()
        for _ in range(3): net(x)
        dt = (time.time() - t) / 3
    print(f"{H}x{W}: {g:.1f} GMACs, 4-thread CPU fp32 {dt*1000:.0f} ms")
