import argparse, os, sys, time, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import *

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--iters", type=int, default=6000)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--p_synth", type=float, default=0.5)
ap.add_argument("--encoder", default="resnet18")
ap.add_argument("--weights", default=None)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", a.name)
os.makedirs(out, exist_ok=True)

tr = MixDS("train", n=a.iters * a.bs, p_synth=a.p_synth, seed=a.seed)
dl = torch.utils.data.DataLoader(tr, batch_size=a.bs, num_workers=a.workers, shuffle=False, drop_last=True, persistent_workers=True, prefetch_factor=4)
va = MixDS("val", n=240, p_synth=0.0, seed=123)
dlv = torch.utils.data.DataLoader(va, batch_size=8, num_workers=4)
dev = "cuda"
net = build_model(a.encoder, a.weights).to(dev).to(memory_format=torch.channels_last)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.05)
scaler = torch.amp.GradScaler()


def validate():
    net.eval(); errs = []; fp = []; fn = []
    with torch.no_grad():
        for x, lab in dlv:
            x = x.to(dev).to(memory_format=torch.channels_last); lab = lab.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = row_logits(net(x))
            pos, pa, _ = decode(lg)
            pres = lab >= 0; ab = lab == -1
            errs.append((pos - lab)[pres & (pa < 0.5)].abs().cpu());
            fp.append(((pa < 0.5)[ab]).float().cpu()); fn.append(((pa >= 0.5)[pres]).float().cpu())
    net.train()
    e = torch.cat(errs)
    return dict(med=float(e.median()), p95=float(e.quantile(0.95)), fp_absent=float(torch.cat(fp).mean()), miss_present=float(torch.cat(fn).mean()))


t0 = time.time(); log = []
net.train()
for it, (x, lab) in enumerate(dl):
    x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last); lab = lab.to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = row_logits(net(x))
    loss, st = row_loss(lg, lab)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
    opt.step(); sched.step()
    if it % 100 == 0:
        print(f"it {it} loss {float(loss):.4f} {st} {time.time() - t0:.0f}s", flush=True)
    if (it + 1) % 1000 == 0 or it + 1 == a.iters:
        v = validate(); v["it"] = it + 1; log.append(v)
        print("VAL", v, flush=True)
        torch.save(net.state_dict(), os.path.join(out, "last.pt"))
json.dump(log, open(os.path.join(out, "log.json"), "w"))
