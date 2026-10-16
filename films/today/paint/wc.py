"""A small procedural watercolour engine (numpy + OpenCV). First written for 云端的梦游者 (memory-amber-2022-mv),
carried through 独身主义的咖啡; this copy adds the palette of 把今天留一小块给我 (mint, sticky-note yellow, tile).

A Sheet holds pigment as per-channel absorbance A over a paper-white ground: the visible colour is PAPER * exp(-A), so
washes laid over each other mix subtractively the way transparent pigment does. A wash is a soft mask that gets
  - a wobbled outline (hand and water never follow the pencil),
  - pooled pigment along its inside edge (the dark rim a drying wash leaves),
  - granulation (pigment settling in the valleys of the paper tooth),
  - slow flow variation (backruns), an optional wet-in-wet feathered bleed, an optional dry-brush break-up.
Pen lines are stamped along a spline with pressure variation, taper and a slight tremor.
Sheets are saved as RGBA WebP: colour is un-mixed from the paper so that a layer drawn with ordinary alpha blending
still darkens what is behind it like a glaze; `body` marks where the layer is opaque paper (it hides farther layers).
"""
import json
import math
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(os.path.dirname(HERE), "art")
os.makedirs(ART, exist_ok=True)
MANIFEST = os.path.join(ART, "manifest.json")
META = os.path.join(ART, "meta")

PAPER = np.array([0.972, 0.950, 0.906], np.float32)
# pigments, as the colour of a full-strength wash on white
# warm side: the room
CREAM = (0.95, 0.88, 0.72)
OCHRE = (0.92, 0.74, 0.38)        # lamp light on a wall
AMBER = (0.95, 0.66, 0.24)
SIENNA = (0.78, 0.50, 0.28)       # raw sienna: wood in light
BURNT = (0.62, 0.33, 0.20)        # burnt sienna: wood in shade, brick
UMBER = (0.40, 0.27, 0.20)        # burnt umber: coffee, dark wood
SEPIA = (0.25, 0.17, 0.14)
COFFEE = (0.30, 0.18, 0.12)
PERSIMMON = (0.88, 0.37, 0.20)    # the coat: the one loud colour in the room
ROSE = (0.86, 0.46, 0.50)
SKIN = (0.93, 0.76, 0.64)
MOSS = (0.44, 0.52, 0.30)
# cool side: the rain
TEAL = (0.20, 0.42, 0.48)
PRUSSIAN = (0.14, 0.27, 0.40)
INDIGO = (0.17, 0.22, 0.40)
RAIN = (0.52, 0.64, 0.70)         # wet air
MIST = (0.74, 0.80, 0.82)
PAYNE = (0.24, 0.29, 0.36)
SLATE = (0.42, 0.48, 0.54)
VIOLET = (0.46, 0.36, 0.58)
# 把今天留一小块给我: an ordinary flat in an old Taipei lane
MINT = (0.56, 0.80, 0.64)         # the mint on the sill: the one fresh colour, it grows through the film
LEAF = (0.30, 0.56, 0.38)         # mint in shade
NOTE = (0.98, 0.88, 0.42)         # the sticky note: a small square of today
TERRA = (0.80, 0.48, 0.36)        # terracotta: the pot, floor tiles, brick
TILE = (0.70, 0.74, 0.66)         # the pale green-grey ceramic tile of old apartment facades
CELADON = (0.72, 0.82, 0.76)
SOY = (0.96, 0.92, 0.80)          # soy milk
OSMAN = (0.96, 0.66, 0.22)        # osmanthus flowers
DUSK = (0.60, 0.52, 0.66)         # the lane at blue hour
NAVY = (0.20, 0.24, 0.36)         # your jacket
INK = (0.15, 0.11, 0.11)          # warm black pen
INKBLUE = (0.10, 0.13, 0.22)


def rs(seed):
    return np.random.default_rng(int(seed) & 0x7FFFFFFF)


def absorb(color):
    return -np.log(np.clip(np.asarray(color, np.float32), 0.02, 1.0))


def sstep(a, b, x):
    u = np.clip((x - a) / (b - a + 1e-9), 0, 1)
    return u * u * (3 - 2 * u)


def fbm(h, w, scale, seed, octaves=4, gain=0.5):
    """Value noise, roughly 0..1 with mean 0.5; `scale` is the feature size of the first octave in px."""
    r = rs(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot, s = 1.0, 0.0, float(scale)
    for _ in range(octaves):
        gh, gw = int(h / s) + 4, int(w / s) + 4
        g = r.random((gh, gw), dtype=np.float32)
        big = cv2.resize(g, (int(math.ceil(gw * s)), int(math.ceil(gh * s))), interpolation=cv2.INTER_CUBIC)
        oy, ox = int(r.integers(0, int(s) + 1)), int(r.integers(0, int(s) + 1))
        out += big[oy:oy + h, ox:ox + w] * amp
        tot += amp
        amp *= gain
        s = max(1.6, s / 2)
    return np.clip((out / tot - 0.5) * 2.4 + 0.5, 0, 1)


def noise1(n, wavelength, seed):
    """Smooth 1-D noise in -1..1, n samples."""
    r = rs(seed)
    k = max(4, int(n / max(1.0, wavelength)) + 4)
    g = r.random(k) * 2 - 1
    x = np.linspace(0, k - 3, n)
    i = np.floor(x).astype(int)
    f = x - i
    f = f * f * (3 - 2 * f)
    return g[i] * (1 - f) + g[i + 1] * f


_paper = {}


def paper(h, w):
    """Cold-press tooth, 0..1 (1 = peak, 0 = valley)."""
    key = (h, w)
    if key not in _paper:
        p = 0.50 * fbm(h, w, 5.0, 11, 3) + 0.30 * fbm(h, w, 2.2, 12, 2) + 0.20 * fbm(h, w, 46, 13, 2)
        _paper[key] = np.clip((p - p.mean()) / (p.std() + 1e-6) * 0.22 + 0.5, 0, 1).astype(np.float32)
    return _paper[key]


_gran = {}


def granfield(h, w):
    """Where pigment settles: the paper tooth, softened, plus slow mottling."""
    key = (h, w)
    if key not in _gran:
        g = cv2.GaussianBlur(paper(h, w), (0, 0), 1.3)
        g = (g - g.mean()) / (g.std() + 1e-6) * 0.2 + 0.5
        _gran[key] = np.clip(0.7 * g + 0.3 * fbm(h, w, 14, 17, 3), 0, 1).astype(np.float32)
    return _gran[key]


def spline(pts, closed=False, step=2.0):
    """Catmull-Rom through the control points, resampled at about `step` px."""
    P = np.asarray(pts, np.float64)
    if len(P) < 3 and not closed:
        n = max(2, int(np.linalg.norm(P[-1] - P[0]) / step))
        return np.linspace(P[0], P[-1], n)
    P = np.vstack([P[-1], P, P[0], P[1]]) if closed else np.vstack([P[0], P, P[-1]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        n = max(2, int(np.linalg.norm(p2 - p1) / step))
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    if not closed:
        out.append(P[-2][None])
    return np.vstack(out)


def arclen(P):
    d = np.hypot(*np.diff(P, axis=0).T)
    return np.concatenate([[0], np.cumsum(d)])


def resample(P, step):
    s = arclen(P)
    n = max(2, int(s[-1] / step))
    u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, P[:, 0]), np.interp(u, s, P[:, 1])], 1)


class Sheet:
    def __init__(self, w, h, seed=0):
        self.w, self.h = int(w), int(h)
        self.A = np.zeros((self.h, self.w, 3), np.float32)
        self.body = np.zeros((self.h, self.w), np.float32)
        self.P = paper(self.h, self.w)
        self.G = granfield(self.h, self.w)
        self.seed, self.n = seed, 0
        self.Y, self.X = np.mgrid[0:self.h, 0:self.w].astype(np.float32)

    def _s(self):
        self.n += 1
        return self.seed * 7919 + self.n * 104729

    # ------------------------------------------------------------ masks
    def zeros(self):
        return np.zeros((self.h, self.w), np.float32)

    def full(self):
        return np.ones((self.h, self.w), np.float32)

    def poly(self, pts, smooth=True, step=3.0):
        P = spline(pts, True, step) if smooth else np.asarray(pts, np.float64)
        m = np.zeros((self.h, self.w), np.uint8)
        cv2.fillPoly(m, [np.round(P * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
        return m.astype(np.float32) / 255

    def ellipse(self, cx, cy, rx, ry, rot=0.0):
        m = np.zeros((self.h, self.w), np.uint8)
        cv2.ellipse(m, (int(round(cx * 16)), int(round(cy * 16))), (int(round(abs(rx) * 16)), int(round(abs(ry) * 16))), rot, 0, 360, 255, -1, cv2.LINE_AA, 4)
        return m.astype(np.float32) / 255

    def rect(self, x0, y0, x1, y1):
        return self.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], smooth=False)

    def blob(self, cx, cy, rx, ry, irr=0.18, n=11, seed=None, rot=0.0):
        r = rs(self._s() if seed is None else seed)
        a = np.linspace(0, 2 * np.pi, n, endpoint=False) + r.random() * 6
        k = 1 + (r.random(n) * 2 - 1) * irr
        x, y = np.cos(a) * rx * k, np.sin(a) * ry * k
        c, s = math.cos(rot), math.sin(rot)
        return self.poly(np.stack([cx + x * c - y * s, cy + x * s + y * c], 1))

    def ribbon(self, pts, w, taper=(0.0, 0.0), profile=None, smooth=True, closed=False):
        """Mask of a brush ribbon along a path; w px wide, optional width profile f(0..1)->0..1."""
        P = spline(pts, closed, 2.0) if smooth else np.asarray(pts, np.float64)
        P = resample(P, max(0.8, w * 0.12))
        n = len(P)
        t = np.linspace(0, 1, n)
        wd = np.full(n, w * 0.5)
        if profile is not None:
            wd *= profile(t)
        if taper[0] > 0:
            wd *= sstep(0, taper[0], t) * 0.9 + 0.1
        if taper[1] > 0:
            wd *= (1 - sstep(1 - taper[1], 1, t)) * 0.9 + 0.1
        m = np.zeros((self.h, self.w), np.uint8)
        for (x, y), r in zip(P, wd):
            cv2.circle(m, (int(round(x * 16)), int(round(y * 16))), max(1, int(round(r * 16))), 255, -1, cv2.LINE_AA, 4)
        return m.astype(np.float32) / 255

    def vgrad(self, y0, y1, a0=0.0, a1=1.0):
        return (a0 + (a1 - a0) * sstep(y0, y1, self.Y)).astype(np.float32)

    def hgrad(self, x0, x1, a0=0.0, a1=1.0):
        return (a0 + (a1 - a0) * sstep(x0, x1, self.X)).astype(np.float32)

    def rgrad(self, cx, cy, r0, r1, a0=1.0, a1=0.0, sy=1.0):
        d = np.hypot(self.X - cx, (self.Y - cy) / sy)
        return (a0 + (a1 - a0) * sstep(r0, r1, d)).astype(np.float32)

    def noise(self, scale, octaves=4, seed=None):
        return fbm(self.h, self.w, scale, self._s() if seed is None else seed, octaves)

    # ------------------------------------------------------------ paint
    def _roi(self, m, pad):
        ys, xs = np.where(m.max(1) > 0.004)[0], np.where(m.max(0) > 0.004)[0]
        if len(ys) == 0:
            return None
        pad = int(pad)
        return slice(max(0, ys[0] - pad), min(self.h, ys[-1] + pad + 1)), slice(max(0, xs[0] - pad), min(self.w, xs[-1] + pad + 1))

    def wash(self, m, color, k=1.0, wob=2.5, soft=0.9, edge=0.5, ew=5.0, gran=0.35, flow=0.3, fs=140, bleed=0.0, dry=0.0,
             grad=None, body=0.0, lift=False, clip=None):
        """Lay one wash. m: mask 0..1. k: pigment load. Returns the final (shaped) mask."""
        seed = self._s()
        roi = self._roi(m, wob * 4 + bleed * 3.5 + ew * 3 + 8)
        if roi is None:
            return m
        m = m[roi].copy()
        hh, ww = m.shape
        if wob > 0:
            dx = (fbm(hh, ww, wob * 16 + 6, seed, 3) - 0.5) * wob * 5 + (fbm(hh, ww, wob * 3 + 3, seed + 1, 2) - 0.5) * wob * 1.4
            dy = (fbm(hh, ww, wob * 16 + 6, seed + 2, 3) - 0.5) * wob * 5 + (fbm(hh, ww, wob * 3 + 3, seed + 3, 2) - 0.5) * wob * 1.4
            m = cv2.remap(m, self.X[:hh, :ww] + dx, self.Y[:hh, :ww] + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if bleed > 0:      # wet-in-wet: the edge feathers and crawls along the damp paper
            b = cv2.GaussianBlur(m, (0, 0), bleed)
            n = fbm(hh, ww, bleed * 1.6 + 3, seed + 7, 4)
            m = sstep(0.12, 0.88, b + (n - 0.5) * 0.7 * np.minimum(1, 4 * b * (1 - b) + 0.3))
            m = cv2.GaussianBlur(m, (0, 0), max(0.7, bleed * 0.22))
        elif soft > 0:
            m = cv2.GaussianBlur(m, (0, 0), soft)
        if clip is not None:
            m = m * clip[roi]
        d = m.copy()
        if edge > 0:       # pigment pooling: darker rim just inside the edge
            ring = np.clip(m - cv2.GaussianBlur(m, (0, 0), ew), 0, 1) * 2
            line = np.clip(m - cv2.GaussianBlur(m, (0, 0), max(0.8, ew * 0.3)), 0, 1) * 2
            d = m * (1 - 0.3 * edge) + edge * (0.6 * ring + 0.7 * line) * m
        if flow > 0:
            d *= 1 + flow * (fbm(hh, ww, fs, seed + 5, 3) - 0.5) * 1.3
        P = self.P[roi]
        if gran > 0:
            d *= np.clip(1 + gran * (0.5 - self.G[roi]) * 1.5, 0, 3)
        if dry > 0:        # dry brush: the paint only catches the peaks of the tooth
            pd = 0.6 * P + 0.4 * fbm(hh, ww, 9, seed + 9, 2)
            d *= sstep(dry - 0.10, dry + 0.10, 1 - pd + (1 - dry) * 0.5)
        if grad is not None:
            d *= grad[roi]
        d = np.clip(d, 0, 4)
        if lift:
            self.A[roi] *= (1 - np.clip(d * k, 0, 1))[..., None]
        else:
            self.A[roi] += d[..., None] * (k * absorb(color))
        if body:
            self.body[roi] = np.maximum(self.body[roi], np.clip(m * body, 0, 1))
        out = np.zeros((self.h, self.w), np.float32)
        out[roi] = m
        return out

    def glaze(self, color, k=0.3, grad=None, flow=0.25, fs=300, gran=0.3):
        """A wash over the whole sheet (no edge)."""
        return self.wash(self.full(), color, k, wob=0, soft=0, edge=0, gran=gran, flow=flow, fs=fs, grad=grad)

    def lift(self, m, amt=0.8, soft=1.5, wob=1.5, bleed=0.0):
        """Lift pigment back toward paper (a damp brush, a tissue, masking fluid)."""
        return self.wash(m, None, amt * (1.25 if amt >= 0.9 else 1), wob=wob, soft=soft, edge=0, gran=0 if amt >= 0.9 else 0.15, flow=0, bleed=bleed, lift=True)

    def stroke(self, pts, w, color, k=1.0, taper=(0.2, 0.3), dry=0.0, edge=0.45, wob=1.5, profile=None, closed=False, **kw):
        return self.wash(self.ribbon(pts, w, taper, profile, closed=closed), color, k, wob=wob, edge=edge, ew=max(1.5, w * 0.18), dry=dry, **kw)

    def ink(self, pts, w=2.0, color=INK, k=1.7, taper=(0.10, 0.22), tremor=0.7, press=0.35, closed=False, smooth=True, skip=0.0, dry=0.25, clip=None):
        """A pen line: pressure variation, taper at both ends, slight tremor, occasional skips over the paper tooth."""
        seed = self._s()
        P = spline(pts, closed, 2.0) if smooth else np.asarray(pts, np.float64)
        P = resample(P, 0.7)
        n = len(P)
        if n < 3:
            return
        L = n * 0.7
        d = np.gradient(P, axis=0)
        nrm = np.stack([-d[:, 1], d[:, 0]], 1)
        nrm /= np.hypot(nrm[:, 0], nrm[:, 1])[:, None] + 1e-9
        off = tremor * (0.6 * noise1(n, 9 / 0.7, seed) + 1.1 * noise1(n, 70 / 0.7, seed + 1))
        P = P + nrm * off[:, None]
        t = np.linspace(0, 1, n)
        wd = w * 0.5 * (1 + press * noise1(n, 55 / 0.7, seed + 2) + 0.15 * press * noise1(n, 7 / 0.7, seed + 3))
        if not closed:
            ta, tb = min(taper[0], 40 / L + taper[0] * 0.3), min(taper[1], 60 / L + taper[1] * 0.3)
            wd *= (0.18 + 0.82 * sstep(0, max(ta, 1e-3), t)) * (0.10 + 0.90 * (1 - sstep(1 - max(tb, 1e-3), 1, t)))
        if skip > 0:
            wd *= sstep(-0.2, 0.1, noise1(n, 26 / 0.7, seed + 4) + 1 - 2 * skip)
        x0, y0 = int(max(0, P[:, 0].min() - w - 4)), int(max(0, P[:, 1].min() - w - 4))
        x1, y1 = int(min(self.w, P[:, 0].max() + w + 5)), int(min(self.h, P[:, 1].max() + w + 5))
        if x1 <= x0 or y1 <= y0:
            return
        m = np.zeros(((y1 - y0) * 2, (x1 - x0) * 2), np.uint8)
        for (x, y), r in zip(P, wd):
            if r > 0.12:
                cv2.circle(m, (int(round((x - x0) * 32)), int(round((y - y0) * 32))), max(3, int(round(r * 32))), 255, -1, cv2.LINE_AA, 4)
        m = cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        roi = (slice(y0, y1), slice(x0, x1))
        if dry > 0:
            m *= 1 - dry * sstep(0.52, 0.80, self.P[roi])
        if clip is not None:
            m *= clip[roi]
        self.A[roi] += m[..., None] * (k * absorb(color))

    def hatch(self, m, ang=0.6, gap=9.0, w=1.3, color=INK, k=1.2, jit=0.35, tremor=0.6, skip=0.15):
        """Pen hatching clipped to a mask."""
        r = rs(self._s())
        roi = self._roi(m, 2)
        if roi is None:
            return
        y0, y1, x0, x1 = roi[0].start, roi[0].stop, roi[1].start, roi[1].stop
        cx, cy, R = (x0 + x1) / 2, (y0 + y1) / 2, math.hypot(x1 - x0, y1 - y0) / 2
        ca, sa = math.cos(ang), math.sin(ang)
        o = -R
        mm = cv2.GaussianBlur(m, (0, 0), 1.0)
        while o < R:
            oo = o + (r.random() - 0.5) * gap * jit
            l0, l1 = -R * (0.9 + 0.1 * r.random()), R * (0.9 + 0.1 * r.random())
            a = (cx + ca * l0 - sa * oo, cy + sa * l0 + ca * oo)
            b = (cx + ca * l1 - sa * oo, cy + sa * l1 + ca * oo)
            self.ink([a, ((a[0] + b[0]) / 2 + (r.random() - 0.5) * 3, (a[1] + b[1]) / 2 + (r.random() - 0.5) * 3), b], w, color, k, tremor=tremor, skip=skip, clip=mm, taper=(0.05, 0.08))
            o += gap

    def spatter(self, n, color, k=0.8, box=None, r=(0.8, 3.0), seed=None, lift=False):
        g = rs(self._s() if seed is None else seed)
        x0, y0, x1, y1 = box or (0, 0, self.w, self.h)
        m = np.zeros((self.h, self.w), np.uint8)
        for _ in range(n):
            rad = r[0] + (r[1] - r[0]) * g.random() ** 2.5
            cv2.circle(m, (int((x0 + g.random() * (x1 - x0)) * 16), int((y0 + g.random() * (y1 - y0)) * 16)), max(2, int(rad * 16)), 255, -1, cv2.LINE_AA, 4)
        m = m.astype(np.float32) / 255
        self.wash(m, color, k, wob=0.6, soft=0.5, edge=0.7, ew=1.5, gran=0.2, flow=0, lift=lift)

    def salt(self, m, amt=0.5, size=2.2, density=0.004, seed=None):
        """Salt crystals thrown into a wet wash: small pale stars."""
        g = rs(self._s() if seed is None else seed)
        s = (g.random((self.h, self.w)) < density).astype(np.float32)
        s = cv2.GaussianBlur(s, (0, 0), size) * (size * size * 5)
        s = np.clip(s * (0.5 + fbm(self.h, self.w, 6, 5, 2)), 0, 1) * m
        self.A *= (1 - amt * s)[..., None]

    # ------------------------------------------------------------ output
    def rgba(self):
        T = np.exp(-self.A)
        a = np.maximum(self.body, 1 - T.min(2))
        a = np.clip(a, 0, 1)
        C = PAPER * (T - (1 - a[..., None])) / np.maximum(a[..., None], 1e-4)
        return np.clip(C, 0, 1), a

    def rgb(self):
        return np.clip(PAPER * np.exp(-self.A), 0, 1)

    def save(self, name, opaque=False, q=90, crop=False, meta=None, anchor=None):
        if opaque:
            im = (self.rgb()[..., ::-1] * 255 + 0.5).astype(np.uint8)
            ox = oy = 0
        else:
            C, a = self.rgba()
            ox = oy = 0
            if crop:
                roi = self._roi(a, 4)
                if roi is not None:
                    C, a = C[roi], a[roi]
                    oy, ox = roi[0].start, roi[1].start
            im = (np.dstack([C[..., ::-1], a]) * 255 + 0.5).astype(np.uint8)
        fn = name + ".webp"
        cv2.imwrite(os.path.join(ART, fn), im, [cv2.IMWRITE_WEBP_QUALITY, q])
        entry = {"f": fn, "w": int(im.shape[1]), "h": int(im.shape[0]), "ox": int(ox), "oy": int(oy), "sw": self.w, "sh": self.h}
        if anchor is not None:
            entry["ax"], entry["ay"] = float(anchor[0]), float(anchor[1])
        if meta:
            entry.update(meta)
        # one small file per layer, merged into manifest.json by build.py: several painters can run at once
        os.makedirs(META, exist_ok=True)
        json.dump(entry, open(os.path.join(META, name + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
        return entry

    def preview(self, path, bg=None):
        """Flattened look on paper (or over another image) for checking."""
        if bg is None:
            bg = np.broadcast_to(PAPER, (self.h, self.w, 3))
        C, a = self.rgba()
        out = bg * (1 - a[..., None]) + C * a[..., None]
        cv2.imwrite(path, (out[..., ::-1] * 255 + 0.5).astype(np.uint8))


def merge_manifest():
    man = {}
    for fn in sorted(os.listdir(META)):
        if fn.endswith(".json"):
            man[fn[:-5]] = json.load(open(os.path.join(META, fn), encoding="utf-8"))
    json.dump(man, open(MANIFEST, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    return man


def paper_tile(size=512, seed=3):
    """A tileable paper sheet for the page: tooth + fibres, as a multiply layer (mid-grey = neutral ≈ white)."""
    r = rs(seed)
    g = r.random((size, size)).astype(np.float32)
    def lp(s):
        return cv2.GaussianBlur(np.tile(g, (3, 3)), (0, 0), s)[size:2 * size, size:2 * size]
    t = (lp(1.1) - 0.5) * 2.2 + (lp(2.6) - 0.5) * 3.5 + (lp(22) - 0.5) * 9
    fib = np.zeros((size, size), np.float32)
    for _ in range(260):
        x, y, a, L = r.random() * size, r.random() * size, r.random() * 6.28, 8 + r.random() * 26
        pts = np.array([[x + math.cos(a + 0.3 * k) * L * k / 5, y + math.sin(a + 0.3 * k) * L * k / 5] for k in range(6)])
        for ox in (-size, 0, size):
            for oy in (-size, 0, size):
                cv2.polylines(fib, [np.round((pts + (ox, oy)) * 16).astype(np.int32)], False, float(r.random() * 0.5 + 0.2), 1, cv2.LINE_AA, 4)
    t = 1 - np.clip(0.035 + t * 0.045 + fib * 0.05, 0, 0.3)
    im = (np.dstack([t * 1.0, t * 0.995, t * 0.985]) * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(ART, "paper.webp"), im, [cv2.IMWRITE_WEBP_QUALITY, 92])
