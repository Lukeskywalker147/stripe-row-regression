import sys, glob, os, cv2, numpy as np, torch
sys.path.insert(0,os.path.expanduser("~/stripe_seg/stripe_unet"))
from infer import load_model, predict_mask
dev=torch.device("cuda")
out=os.path.expanduser("~/stripe_seg/cam_snaps_out"); 
m,cfg=load_model(os.path.expanduser("~/stripe_seg/runs/snap_r18_ft/best.pt"),dev)
print("best epoch",torch.load(os.path.expanduser("~/stripe_seg/runs/snap_r18_ft/best.pt"),map_location="cpu")["epoch"])
for p in sorted(glob.glob(os.path.expanduser("~/stripe_seg/cam_snaps/*.png"))):
    im=cv2.cvtColor(cv2.imread(p),cv2.COLOR_BGR2RGB); s=os.path.basename(p)[:-4]
    a,_=predict_mask(m,im,dev); cv2.imwrite(f"{out}/{s}_ft_orig.png",a.astype(np.uint8)*255)
