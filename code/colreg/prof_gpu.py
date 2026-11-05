import glob, time, numpy as np, cv2, torch
import synth_gpu as SG
from torch.profiler import profile, ProfilerActivity
dev="cuda"
bgs=sorted(glob.glob("bg/*.jpg")); b=cv2.imread(bgs[0])
lb=lambda r: np.ascontiguousarray(b[:512,:768])
plans=[SG.plan_render(np.random.default_rng((2,i)),lb) for i in range(60)]
for p in plans[:5]: SG.compose(p,dev)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
    for p in plans[5:45]: SG.compose(p,dev)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total",row_limit=12,max_name_column_width=40))
print(prof.key_averages().table(sort_by="cpu_time_total",row_limit=8,max_name_column_width=40))
