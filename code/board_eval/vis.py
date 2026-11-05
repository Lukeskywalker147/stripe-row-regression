import sys, numpy as np, cv2
sys.path.insert(0, "LASER25_DIR")
from pathlib import Path
from reanalyze_steger import *
IM = Path("LASER25_DIR/images")
nn = np.load("/tmp/nn_pts.npz"); target = CharucoTarget(CharucoConfig.from_cfg(CFG))
outs = []
for name in ("HE_005", "HE_013", "HE_021"):
    img = cv2.imread(str(IM / f"{name}.png")); det = target.detect(img[:, :, 1])
    hull = cv2.convexHull(det.corners.astype(np.float32).reshape(-1, 1, 2))
    mask = np.zeros(img.shape[:2], np.uint8); cv2.fillConvexPoly(mask, np.rint(hull).astype(np.int32), 1)
    st = np.asarray(extract_laser_centers(img, CFG, image_mask_override=mask.astype(bool))).reshape(-1, 2)[:, :2]
    p = nn[f"{name}_v3"][:, :2]
    p = p[mask[np.clip(np.rint(p[:, 1]).astype(int), 0, 1079), np.clip(np.rint(p[:, 0]).astype(int), 0, 1919)] > 0]
    cx, cy = np.median(p, 0).astype(int); x0, y0 = max(cx - 200, 0), max(cy - 130, 0)
    crop = img[y0:y0 + 260, x0:x0 + 400]; S = 3
    big = cv2.resize(crop, None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC); big2 = big.copy()
    for pts, im, col in ((p, big, (0, 255, 0)), (st, big2, (255, 0, 255))):
        for x, y in pts:
            if x0 <= x < x0 + 400 and y0 <= y < y0 + 260:
                cv2.circle(im, (int((x - x0 + .5) * S), int((y - y0 + .5) * S)), 2, col, -1)
    cv2.putText(big, f"{name} NN v3 (green)", (8, 24), 0, .7, (0, 255, 255), 2); cv2.putText(big2, "Steger (magenta)", (8, 24), 0, .7, (0, 255, 255), 2)
    outs.append(np.hstack([big, big2]))
cv2.imwrite("board_zoom.jpg", np.vstack(outs), [cv2.IMWRITE_JPEG_QUALITY, 85])
