import glob, os, sys, numpy as np, cv2
from multiprocessing import Pool
R = os.path.expanduser("~/stripe_seg/data_scan_v3/train")
OUT = "bg"; os.makedirs(OUT, exist_ok=True)

def f(p):
    img = cv2.imread(p)
    m = cv2.imread(R + "/masks/" + os.path.basename(p).rsplit(".", 1)[0] + ".png", 0) > 0
    m = cv2.dilate(m.astype(np.uint8), np.ones((1, 61), np.uint8))
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    sm = cv2.resize(img, None, fx=0.5, fy=0.5); mm = cv2.resize(m, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    out = cv2.inpaint(sm, mm, 5, cv2.INPAINT_TELEA)
    out = cv2.resize(out, (img.shape[1], img.shape[0]))
    out = np.where(m[:, :, None] > 0, out, img)
    cv2.imwrite(OUT + "/" + os.path.basename(p), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return 1

if __name__ == "__main__":
    ims = sorted(glob.glob(R + "/images/*.jpg"))[::4]
    with Pool(10) as pool: print(sum(pool.map(f, ims)), "backgrounds")
