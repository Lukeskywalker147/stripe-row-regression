import sys, glob, os, cv2, numpy as np, torch
sys.path.insert(0,os.path.expanduser("~/stripe_seg/stripe_unet"))
from infer import load_model, predict_mask
dev=torch.device("cuda")
out=os.path.expanduser("~/stripe_seg/cam_snaps_out"); os.makedirs(out,exist_ok=True)
runs={"r18":"scan3_r18","upp":"scan3_UnetPlusPlus_resnet18_s0"}
files=sorted(glob.glob(os.path.expanduser("~/stripe_seg/cam_snaps/*.png")))
for k,r in runs.items():
    m,cfg=load_model(os.path.expanduser(f"~/stripe_seg/runs/{r}/best.pt"),dev)
    for p in files:
        im=cv2.cvtColor(cv2.imread(p),cv2.COLOR_BGR2RGB)
        s=os.path.basename(p)[:-4]
        a,_=predict_mask(m,im,dev)
        b,_=predict_mask(m,np.ascontiguousarray(im.swapaxes(0,1)),dev)
        cv2.imwrite(f"{out}/{s}_{k}_orig.png",a.astype(np.uint8)*255)
        cv2.imwrite(f"{out}/{s}_{k}_tran.png",b.T.astype(np.uint8)*255)
print("ok")
