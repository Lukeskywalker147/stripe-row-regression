"""Per-row sub-pixel center labels from existing mask labels.

Label value per image row y (stripe is near-vertical, scan_axis=row):
  x >= 0   : sub-pixel center (Steger and centroid agree inside the mask)
  -1       : stripe absent (row is well outside the mask extent)
  NaN      : ignore (ambiguous)
"""
import argparse, glob, os, sys
import numpy as np, cv2
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
from laser_ref import extract_centers

CFG = dict(method="steger", score_mode="blue_gated_intensity", blue_gate_threshold=5,
           score_threshold=14, min_intensity=16, scan_axis="row", steger_sigma=2.0)
CFG_C = dict(CFG, method="centroid")
MARGIN = 25
AGREE_PX = 2.0


def label_frame(args):
    img_path, mask_path = args
    img = cv2.imread(img_path)
    m = cv2.imread(mask_path, 0) > 0
    H = img.shape[0]
    lab = np.full(H, np.nan, np.float32)
    ys = np.nonzero(m.any(1))[0]
    if len(ys) == 0:
        return os.path.basename(img_path), lab
    y0, y1 = ys.min(), ys.max()
    lab[: max(0, y0 - MARGIN)] = -1
    lab[min(H, y1 + MARGIN + 1):] = -1
    md = cv2.dilate(m.astype(np.uint8), np.ones((1, 9), np.uint8)) > 0
    st = extract_centers(img, CFG, md)
    ce = extract_centers(img, CFG_C, md)
    sd = {int(round(v)): u for u, v in st}
    cd = {int(round(v)): u for u, v in ce}
    for y in range(H):
        if y in sd and y in cd and abs(sd[y] - cd[y]) <= AGREE_PX:
            # exactly one run in the (undilated) mask row
            r = np.diff(np.concatenate([[0], m[y].astype(np.int8), [0]]))
            if (r == 1).sum() == 1:
                lab[y] = sd[y]
    return os.path.basename(img_path), lab


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/stripe_seg/data_scan_v3"))
    ap.add_argument("--out", default="labels")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for split in ["train", "val"]:
        ims = sorted(glob.glob(f"{a.root}/{split}/images/*.jpg"))
        jobs = [(p, f"{a.root}/{split}/masks/" + os.path.basename(p).rsplit(".", 1)[0] + ".png") for p in ims]
        with Pool(12) as pool:
            res = pool.map(label_frame, jobs, chunksize=4)
        names = np.array([r[0] for r in res])
        labs = np.stack([r[1] for r in res])
        np.savez_compressed(f"{a.out}/{split}.npz", names=names, labels=labs)
        pres = (labs >= 0).sum(); ab = (labs == -1).sum(); ig = np.isnan(labs).sum()
        print(split, len(names), "present", pres, "absent", ab, "ignore", ig,
              "present_frac_of_mask_rows %.2f" % (pres / max(1, pres + ig - 0)))
