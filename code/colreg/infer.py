import os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import build_model, row_logits, to_tensor


class RowRegressor:
    def __init__(self, ckpt, encoder="resnet18", dev="cuda"):
        self.net = build_model(encoder, None).to(dev).eval()
        self.net.load_state_dict(torch.load(ckpt, map_location=dev))
        self.dev = dev

    @torch.no_grad()
    def probs(self, img):
        H, W = img.shape[:2]
        x = to_tensor(img)[None].to(self.dev)
        ph, pw = (-H) % 32, (-W) % 32
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = row_logits(self.net(x))
        lg = lg[0].float()
        p = F.softmax(lg, 1)
        # drop padded columns, renormalise
        pp = p[:H, :W]; pa = p[:H, -1:]
        z = pp.sum(1, keepdim=True) + pa
        return (pp / z).cpu().numpy(), (pa / z)[:, 0].cpu().numpy()

    def extract(self, img, k=3, win=3, lam=0.08, abs_thr=0.5, viterbi=True, min_mass=0.0, pr=None):
        """returns (N,3) rows of (u, v, conf); v is the image row."""
        pp, pa = pr if pr is not None else self.probs(img)
        H, W = pp.shape
        # local maxima candidates
        cand = []
        for y in range(H):
            r = pp[y]
            idx = np.nonzero((r[1:-1] >= r[:-2]) & (r[1:-1] > r[2:]))[0] + 1
            idx = idx[np.argsort(-r[idx])[:k]]
            cs = []
            for i in idx:
                lo, hi = max(0, i - win), min(W, i + win + 1)
                w = r[lo:hi]; m = w.sum()
                if m < 1e-3:
                    continue
                cs.append(((w * np.arange(lo, hi)).sum() / m, m))
            cand.append(cs)
        if not viterbi:
            out = []
            for y, cs in enumerate(cand):
                if cs and pa[y] < abs_thr:
                    u, m = max(cs, key=lambda t: t[1]); out.append((u, y, m))
            return np.array(out, np.float32).reshape(-1, 3)
        # Viterbi: states = candidates + absent
        INF = 1e9
        prev_cost = None; back = []; states = []
        for y in range(H):
            cs = cand[y]
            c = np.array([-np.log(max(m, 1e-6)) for _, m in cs] + [-np.log(max(pa[y], 1e-6))])
            st = [u for u, _ in cs] + [None]
            if prev_cost is None:
                cost = c; bp = np.zeros(len(c), int)
            else:
                cost = np.empty(len(c)); bp = np.empty(len(c), int)
                for j, uj in enumerate(st):
                    best, bi = INF, 0
                    for i, ui in enumerate(states[-1]):
                        if uj is None or ui is None:
                            tr = 0.0 if (uj is None) == (ui is None) else 1.0
                        else:
                            d = abs(uj - ui)
                            tr = lam * min(d, 20.0) + (3.0 if d > 20 else 0.0)
                        v = prev_cost[i] + tr
                        if v < best: best, bi = v, i
                    cost[j] = best + c[j]; bp[j] = bi
            back.append(bp); states.append(st); prev_cost = cost
        j = int(np.argmin(prev_cost)); path = []
        for y in range(H - 1, -1, -1):
            path.append(j); j = back[y][j]
        path = path[::-1]
        out = []
        for y, j in enumerate(path):
            if states[y][j] is not None:
                out.append((states[y][j], y, cand[y][j][1]))
        return np.array(out, np.float32).reshape(-1, 3)


def _gate(pts, thr):
    """keep points whose local |du/dv| <= thr (pts rows (u,v,conf), v = row index of the pass)."""
    if len(pts) == 0:
        return pts
    d = {int(round(v)): u for u, v, _ in pts}
    keep = []
    for u, v, c in pts:
        y = int(round(v)); s = None
        for h in (4, 2, 1):
            if (y - h) in d and (y + h) in d:
                s = (d[y + h] - d[y - h]) / (2 * h); break
        keep.append(s is not None and abs(s) <= thr)
    return pts[np.array(keep, bool)]


def extract_dual(model, img, thr=1.1, **kw):
    """Run on the image and on its 90deg-CW rotation; keep the pass in which the stripe is steep.
    returns (N,3): x, y, conf in ORIGINAL image coordinates."""
    import cv2
    H, W = img.shape[:2]
    a = _gate(model.extract(img, **kw), thr)
    b = _gate(model.extract(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), **kw), thr)
    out = [a] if len(a) else []
    if len(b):
        out.append(np.stack([b[:, 1], H - 1 - b[:, 0], b[:, 2]], 1))
    return np.concatenate(out, 0).astype(np.float32) if out else np.zeros((0, 3), np.float32)


def _peak(pp, pa):
    k = pp.argmax(1)
    mass = np.array([pp[i, max(0, j - 3):j + 4].sum() for i, j in enumerate(k)])
    return float(((1 - pa) * mass).sum())


class OrientRegressor(RowRegressor):
    """v3 model: row head and column head; orientation chosen per image by total stripe-presence mass."""
    def __init__(self, ckpt, encoder="resnet18", dev="cuda", mode="auto", thr=1.1):
        import core3
        self.mode, self.thr = mode, thr
        import json
        mp = os.path.join(os.path.dirname(ckpt), "meta.json")
        meta = json.load(open(mp)) if os.path.exists(mp) else {}
        self.head = meta.get("head", "row")
        self.net = core3.build_model(meta.get("encoder", encoder), None, meta.get("arch", "base")).to(dev).eval()
        self.net.load_state_dict(torch.load(ckpt, map_location=dev))
        self.dev = dev

    @torch.no_grad()
    def probs2(self, img):
        from core3 import two_logits, two_logits_seg
        TL = two_logits if self.head == "row" else two_logits_seg
        H, W = img.shape[:2]
        x = to_tensor(img)[None].to(self.dev)
        ph, pw = (-H) % 32, (-W) % 32
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.net(x)
            lr, lc = TL(out)
        res = []
        for lg, n_line, n_pos in ((lr[0], H, W), (lc[0], W, H)):
            p = F.softmax(lg.float(), 1)
            pp = p[:n_line, :n_pos]; pa = p[:n_line, -1:]
            z = pp.sum(1, keepdim=True) + pa
            res.append(((pp / z).cpu().numpy(), (pa / z)[:, 0].cpu().numpy()))
        return res

    def extract(self, img, orient=None, return_orient=False, **kw):
        (pr_r, pa_r), (pr_c, pa_c) = self.probs2(img)
        o = orient or self.mode
        if o == "both":
            a = _gate(super().extract(img, pr=(pr_r, pa_r), **kw), self.thr)
            b = _gate(super().extract(img, pr=(pr_c, pa_c), **kw), self.thr)
            if len(b):
                b = b[:, [1, 0, 2]]
            pts = np.concatenate([x for x in (a, b) if len(x)], 0) if (len(a) or len(b)) else np.zeros((0, 3), np.float32)
            return (pts, "both") if return_orient else pts
        if o == "auto":
            o = "row" if _peak(pr_r, pa_r) >= _peak(pr_c, pa_c) else "col"
        if o == "row":
            pts = super().extract(img, pr=(pr_r, pa_r), **kw)
        else:
            p = super().extract(img, pr=(pr_c, pa_c), **kw)
            pts = p[:, [1, 0, 2]] if len(p) else p
        return (pts, o) if return_orient else pts
