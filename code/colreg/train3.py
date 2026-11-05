import argparse, os, sys, time, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core3 import *

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--iters", type=int, default=16000)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--p_synth", type=float, default=0.5)
ap.add_argument("--p_rot", type=float, default=0.5)
ap.add_argument("--encoder", default="resnet18")
ap.add_argument("--weights", default=None)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--arch", default="base")
ap.add_argument("--head", default="row")
ap.add_argument("--gpu_synth", action="store_true")
a = ap.parse_args()
torch.manual_seed(a.seed)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", a.name)
os.makedirs(out, exist_ok=True)
json.dump(dict(arch=a.arch, head=a.head, encoder=a.encoder), open(os.path.join(out, "meta.json"), "w"))
TL = two_logits if a.head == "row" else two_logits_seg
if a.gpu_synth:
    from core3g import MixDSG, assemble, passthrough
else:
    MixDSG = MixDS3
tr = MixDSG("train", n=a.iters * a.bs, p_synth=a.p_synth, seed=a.seed, p_rot=a.p_rot)
dl = torch.utils.data.DataLoader(tr, batch_size=a.bs, num_workers=a.workers, shuffle=False, drop_last=True, persistent_workers=True, prefetch_factor=4, **(dict(collate_fn=passthrough) if a.gpu_synth else {}))
va = MixDS3("val", n=240, p_synth=0.0, seed=123)
dlv = torch.utils.data.DataLoader(va, batch_size=8, num_workers=4)
dev = "cuda"
net = build_model(a.encoder, a.weights, a.arch).to(dev).to(memory_format=torch.channels_last)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.05)


def validate():
    net.eval(); errs = []; fp = []; fn = []
    with torch.no_grad():
        for x, lr, lc in dlv:
            x = x.to(dev).to(memory_format=torch.channels_last); lr = lr.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, _ = TL(net(x))
            pos, pa, _ = decode(lg)
            pres = lr >= 0; ab = lr == -1
            errs.append((pos - lr)[pres & (pa < 0.5)].abs().cpu())
            fp.append(((pa < 0.5)[ab]).float().cpu()); fn.append(((pa >= 0.5)[pres]).float().cpu())
    net.train()
    e = torch.cat(errs)
    return dict(med=float(e.median()), p95=float(e.quantile(0.95)), fp_absent=float(torch.cat(fp).mean()), miss_present=float(torch.cat(fn).mean()))


t0 = time.time(); log = []
net.train()
for it, bt in enumerate(dl):
    if a.gpu_synth:
        x, lr_, lc_ = assemble(bt, dev, a.p_rot)
    else:
        x, lr_, lc_ = bt
    x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last); lr_ = lr_.to(dev); lc_ = lc_.to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lgr, lgc = TL(net(x))
    LF = row_loss if a.head == "row" else seg_loss
    l1, s1 = LF(lgr, lr_)
    l2, s2 = LF(lgc, lc_)
    loss = l1 + l2
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
    opt.step(); sched.step()
    if it % 100 == 0:
        print(f"it {it} loss {float(loss):.4f} row {s1} col {s2} {time.time() - t0:.0f}s", flush=True)
    if (it + 1) % 1000 == 0 or it + 1 == a.iters:
        v = validate(); v["it"] = it + 1; log.append(v)
        print("VAL", v, flush=True)
        torch.save(net.state_dict(), os.path.join(out, "last.pt"))
json.dump(log, open(os.path.join(out, "log.json"), "w"))
