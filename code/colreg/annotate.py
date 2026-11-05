#!/usr/bin/env python3
"""Hand-label laser-stripe centre lines on full-resolution frames.

Usage:  python3 annotate.py [--src ~/cam_snaps] [--out ~/Downloads/stripe_results/hand_labels] [--frames a.png b.png ...]

Mouse : left click = add point | right click = undo last point | wheel = zoom at cursor
Keys  : b = break (start a new segment, use it where the stripe is hidden/absent)
        u = undo last point        c = clear this frame
        n / m = next / previous frame (auto-saves)
        w a s d (or i j k l) = pan   + / - = zoom   r = reset view
        g = toggle brightness boost  h = toggle point display   q = save and quit

Labelling rule: click the CENTRE LINE of the stripe's bright core (not the blue halo), about every
10-30 px at zoom >= 3. Where the stripe is not visible, do not click; press b to break the line.
Coordinates are saved in original-image pixel units (pixel centre = integer), one JSON per frame.
"""
import argparse, json, os
import numpy as np, cv2

DEFAULT_FRAMES = ["snap_20261030_115403", "snap_20261030_115425", "snap_20261030_115434",
                  "snap_20261030_115439", "snap_20261030_115443", "snap_20261030_115458",
                  "snap_20261030_115503", "snap_20261030_115507"]
VW, VH = 1600, 900
WIN = "annotate"


class View:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.base = min(VW / W, VH / H)
        self.reset()

    def reset(self):
        self.z, self.cx, self.cy = 1.0, self.W / 2, self.H / 2

    @property
    def scale(self):
        return self.base * self.z

    def to_img(self, sx, sy):
        s = self.scale
        return (sx - VW / 2) / s + self.cx, (sy - VH / 2) / s + self.cy

    def to_screen(self, x, y):
        s = self.scale
        return (x - self.cx) * s + VW / 2, (y - self.cy) * s + VH / 2

    def zoom_at(self, sx, sy, f):
        x, y = self.to_img(sx, sy)
        self.z = float(np.clip(self.z * f, 1.0, 40.0))
        s = self.scale
        self.cx, self.cy = x - (sx - VW / 2) / s, y - (sy - VH / 2) / s

    def pan(self, dx, dy):
        self.cx += dx / self.scale; self.cy += dy / self.scale


def render(img, view, segs, boost=True, show=True, header=""):
    s = view.scale
    M = np.array([[s, 0, VW / 2 - view.cx * s], [0, s, VH / 2 - view.cy * s]], np.float32)
    src = img
    if boost:
        src = np.clip(255.0 * (img / 255.0) ** 0.6 * 1.2, 0, 255).astype(np.uint8)
    interp = cv2.INTER_NEAREST if s >= 4 else cv2.INTER_LINEAR
    out = cv2.warpAffine(src, M, (VW, VH), flags=interp, borderMode=cv2.BORDER_CONSTANT)
    if show:
        for k, seg in enumerate(segs):
            pts = [view.to_screen(x, y) for x, y in seg]
            col = (0, 0, 255) if k % 2 == 0 else (0, 200, 255)
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, 1, cv2.LINE_AA)
            for p in pts:
                cv2.circle(out, (int(p[0]), int(p[1])), 3, col, 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (VW, 26), (0, 0, 0), -1)
    cv2.putText(out, header, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def load(path_json):
    if os.path.exists(path_json):
        return [[tuple(p) for p in seg] for seg in json.load(open(path_json))["segments"]]
    return [[]]


def save(path_json, name, W, H, segs):
    segs = [[(round(float(x), 2), round(float(y), 2)) for x, y in s] for s in segs if len(s) > 0]
    json.dump(dict(image=name, width=W, height=H, segments=segs), open(path_json, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.expanduser("~/cam_snaps"))
    ap.add_argument("--out", default=os.path.expanduser("~/Downloads/stripe_results/hand_labels"))
    ap.add_argument("--frames", nargs="*", default=DEFAULT_FRAMES)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    names = [os.path.splitext(os.path.basename(f))[0] for f in a.frames]
    st = dict(i=0, boost=True, show=True, mouse=(VW // 2, VH // 2))

    def open_frame(i):
        n = names[i]
        img = cv2.imread(os.path.join(a.src, n + ".png"))
        if img is None:
            raise SystemExit(f"cannot read {a.src}/{n}.png")
        st.update(i=i, name=n, img=img, view=View(img.shape[1], img.shape[0]),
                  segs=load(os.path.join(a.out, n + ".json")))

    def do_save():
        save(os.path.join(a.out, st["name"] + ".json"), st["name"], st["img"].shape[1], st["img"].shape[0], st["segs"])

    def on_mouse(ev, x, y, flags, _):
        st["mouse"] = (x, y)
        if ev == cv2.EVENT_LBUTTONDOWN:
            st["segs"][-1].append(st["view"].to_img(x, y)); do_save()
        elif ev == cv2.EVENT_RBUTTONDOWN:
            undo()
        elif ev == cv2.EVENT_MOUSEWHEEL:
            st["view"].zoom_at(x, y, 1.25 if flags > 0 else 0.8)

    def undo():
        if not st["segs"][-1] and len(st["segs"]) > 1:
            st["segs"].pop()
        if st["segs"][-1]:
            st["segs"][-1].pop()
        do_save()

    open_frame(0)
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)
    while True:
        npts = sum(len(s) for s in st["segs"])
        hdr = f"[{st['i'] + 1}/{len(names)}] {st['name']}   points {npts}  segments {sum(1 for s in st['segs'] if s)}   zoom {st['view'].z:.1f}x   (n/m frame, b break, u undo, q quit)"
        cv2.imshow(WIN, render(st["img"], st["view"], st["segs"], st["boost"], st["show"], hdr))
        k = cv2.waitKey(30) & 0xFF
        v = st["view"]; step = 60
        if k == 255:
            continue
        c = chr(k) if 32 <= k < 127 else ""
        if c == "q":
            do_save(); break
        elif c == "b" and st["segs"][-1]:
            st["segs"].append([]); do_save()
        elif c == "u":
            undo()
        elif c == "c":
            st["segs"] = [[]]; do_save()
        elif c == "n" and st["i"] < len(names) - 1:
            do_save(); open_frame(st["i"] + 1)
        elif c == "m" and st["i"] > 0:
            do_save(); open_frame(st["i"] - 1)
        elif c in ("w", "i"): v.pan(0, -step)
        elif c in ("s", "k"): v.pan(0, step)
        elif c in ("a", "j"): v.pan(-step, 0)
        elif c in ("d", "l"): v.pan(step, 0)
        elif c in ("+", "="): v.zoom_at(*st["mouse"], 1.25)
        elif c == "-": v.zoom_at(*st["mouse"], 0.8)
        elif c == "r": v.reset()
        elif c == "g": st["boost"] = not st["boost"]
        elif c == "h": st["show"] = not st["show"]
    cv2.destroyAllWindows()
    print("saved to", a.out)


if __name__ == "__main__":
    main()
