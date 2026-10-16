"""Two old Taipei walk-ups across a narrow lane, at night (and at first light); and one of their windows, close.

BLOCK (sheet 3072x1728). Seen from down in the lane, looking up a little. Painted through a real camera (see CAM):
the two facades stand like an open book, the left one (5 storeys) receding to the right, the right one (4 storeys and a
rusty tin room on its roof) receding to the left; between their roofs a wedge of sky with wires across it; at the
bottom, between their far ends, the lane runs on to a cross street.
Layers, back to front (all p = 1 unless said):
  bk_sky        opaque  night sky, low rain cloud, sodium glow low down                 (suggest p 0.35)
  bk_sky_dawn   opaque  the same sky at first light                                      (suggest p 0.35)
  bk_far        sprite  the lane beyond the blocks (far end building, the cross street), the wet lane floor (p 0.9)
  bk_glass      sprite  the dark glass of every window (rooms with the light off)
  bk_glass_lit  sprite  every window as a lit room (warm, curtains, a lamp); clip it to one window's quad to light it
  bk_sil_you / bk_sil_me   sprites, put inside "yours" / "mine" (anchor = bottom centre of that window's rect)
  bk_left, bk_right  sprites  the buildings; window glass is TRANSPARENT so whatever is drawn behind shows through
  bk_wires      sprite  near wires, a wall-arm street lamp, the edge of a vertical shop sign (suggest p 1.3)
All sprites are saved with crop and anchor (0, 0): draw them with L() (their ox/oy place them on the sheet).
Meta on bk_left / bk_right: windows [[x0,y0,x1,y1],...] (bounding rects of the glass), quads [[[x,y]x4],...] (the exact
glass corners: bottom-left, bottom-right, top-right, top-left in the facade's own sense), floors [Taiwan floor number
per window, 2..5], mine / yours (index into that layer's windows), roof [[x,y],...] roofline.

WIN (sheet 2560x1440). One window of the block seen straight on from across the lane at night.
  wn_bg opaque: tiled wall, aluminium frame, the warm room inside. meta glass, lamp, desk.
  wn_grille sprite (anchor 0,0, sheet coords); wn_you, wn_you_up (anchor = desk-edge centre); wn_note (anchor centre).
"""
import math

import cv2
import numpy as np

from kit import *

W, H = 3072, 1728
FONT = "D:/github_repository/a-little-of-today-mv/assets/fonts/NotoSerifSC-VF.ttf"
HAND = "D:/github_repository/a-little-of-today-mv/assets/fonts/LXGWWenKai-Regular.ttf"
PINK = (0.92, 0.82, 0.76)          # the pale pink-beige tile
GREY = (0.78, 0.78, 0.76)          # bare cement
RUST = (0.66, 0.36, 0.22)


# ---------------------------------------------------------------------------------------------- a window onto a sheet
class Sub(Sheet):
    """A rectangular part of a sheet that paints in the parent's coordinates (same paper, same pigment arrays)."""

    def __init__(self, s, x0, y0, x1, y1):
        px, py = getattr(s, "ox", 0), getattr(s, "oy", 0)
        x0, y0 = max(px, int(x0)), max(py, int(y0))
        x1, y1 = min(px + s.w, int(x1)), min(py + s.h, int(y1))
        self.ox, self.oy = x0, y0
        self.w, self.h = max(1, x1 - x0), max(1, y1 - y0)
        sl = (slice(y0 - py, y0 - py + self.h), slice(x0 - px, x0 - px + self.w))
        self.A, self.body, self.P, self.G = s.A[sl], s.body[sl], s.P[sl], s.G[sl]
        self.seed, self.n = s.seed * 31 + x0 * 7 + y0 * 3 + 1, 0
        self.Y, self.X = s.Y[:self.h, :self.w], s.X[:self.h, :self.w]
        self._raw = False
        self._g = rs(self.seed + 977)

    def _p(self, pts):
        P = np.asarray(pts, np.float64)
        return P if self._raw else P - (self.ox, self.oy)

    def poly(self, pts, smooth=True, step=3.0):
        return Sheet.poly(self, self._p(pts), smooth, step)

    def ellipse(self, cx, cy, rx, ry, rot=0.0):
        return Sheet.ellipse(self, cx - self.ox, cy - self.oy, rx, ry, rot)

    def ribbon(self, pts, w, taper=(0.0, 0.0), profile=None, smooth=True, closed=False):
        return Sheet.ribbon(self, self._p(pts), w, taper, profile, smooth, closed)

    def vgrad(self, y0, y1, a0=0.0, a1=1.0):
        return Sheet.vgrad(self, y0 - self.oy, y1 - self.oy, a0, a1)

    def hgrad(self, x0, x1, a0=0.0, a1=1.0):
        return Sheet.hgrad(self, x0 - self.ox, x1 - self.ox, a0, a1)

    def rgrad(self, cx, cy, r0, r1, a0=1.0, a1=0.0, sy=1.0):
        return Sheet.rgrad(self, cx - self.ox, cy - self.oy, r0, r1, a0, a1, sy)

    def ink(self, pts, w=2.0, color=INK, k=1.7, taper=(0.10, 0.22), tremor=0.7, press=0.35, closed=False, smooth=True, skip=0.0, dry=0.25, clip=None):
        """The engine's pen line, drawing its hand-noise from one generator per sheet (cheap for thousands of strokes)."""
        g = self._g
        P = self._p(pts)
        P = spline(P, closed, 2.0) if smooth else P
        P = resample(P, 0.7)
        n = len(P)
        if n < 3:
            return

        def nz(wl):
            kk = max(4, int(n * 0.7 / max(1.0, wl)) + 4)
            gg = g.random(kk) * 2 - 1
            x = np.linspace(0, kk - 3, n)
            i = x.astype(int)
            f = x - i
            f = f * f * (3 - 2 * f)
            return gg[i] * (1 - f) + gg[i + 1] * f

        L = n * 0.7
        d = np.gradient(P, axis=0)
        nrm = np.stack([-d[:, 1], d[:, 0]], 1)
        nrm /= np.hypot(nrm[:, 0], nrm[:, 1])[:, None] + 1e-9
        P = P + nrm * (tremor * (0.6 * nz(9) + 1.1 * nz(70)))[:, None]
        t = np.linspace(0, 1, n)
        wd = w * 0.5 * (1 + press * nz(55) + 0.15 * press * nz(7))
        if not closed:
            ta, tb = min(taper[0], 40 / L + taper[0] * 0.3), min(taper[1], 60 / L + taper[1] * 0.3)
            wd *= (0.18 + 0.82 * sstep(0, max(ta, 1e-3), t)) * (0.10 + 0.90 * (1 - sstep(1 - max(tb, 1e-3), 1, t)))
        if skip > 0:
            wd *= sstep(-0.2, 0.1, nz(26) + 1 - 2 * skip)
        x0, y0 = int(max(0, P[:, 0].min() - w - 4)), int(max(0, P[:, 1].min() - w - 4))
        x1, y1 = int(min(self.w, P[:, 0].max() + w + 5)), int(min(self.h, P[:, 1].max() + w + 5))
        if x1 <= x0 or y1 <= y0:
            return
        m = np.zeros(((y1 - y0) * 2, (x1 - x0) * 2), np.uint8)
        X = np.round((P[:, 0] - x0) * 32).astype(int).tolist()
        Y = np.round((P[:, 1] - y0) * 32).astype(int).tolist()
        Rr = np.round(wd * 32).astype(int).tolist()
        for i in range(n):
            if wd[i] > 0.12:
                cv2.circle(m, (X[i], Y[i]), max(3, Rr[i]), 255, -1, cv2.LINE_AA, 4)
        m = cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        roi = (slice(y0, y1), slice(x0, x1))
        if dry > 0:
            m *= 1 - dry * sstep(0.52, 0.80, self.P[roi])
        if clip is not None:
            m *= clip[roi]
        self.A[roi] += m[..., None] * (k * absorb(color))

    def hatch(self, *a, **kw):
        self._raw = True
        try:
            Sheet.hatch(self, *a, **kw)
        finally:
            self._raw = False

    def spatter(self, n, color, k=0.8, box=None, **kw):
        if box is not None:
            box = (box[0] - self.ox, box[1] - self.oy, box[2] - self.ox, box[3] - self.oy)
        return Sheet.spatter(self, n, color, k, box, **kw)


class Off(Sub):
    """A sheet of its own that is painted in world coordinates: its top-left corner sits at (ox, oy) on the world."""

    def __init__(self, w, h, ox, oy, seed=0):
        Sheet.__init__(self, w, h, seed)
        self.ox, self.oy, self._raw = int(ox), int(oy), False
        self._g = rs(seed + 977)


def sub_of(S, pts, pad=40):
    P = np.asarray(pts, float).reshape(-1, 2)
    return Sub(S, P[:, 0].min() - pad, P[:, 1].min() - pad, P[:, 0].max() + pad, P[:, 1].max() + pad)


# ---------------------------------------------------------------------------------------------- small habits
def J(r, v=1.0):
    return (r.random() - 0.5) * 2 * v


def U(r, a, b):
    return a + (b - a) * r.random()


def box(s, r, x0, y0, x1, y1, w=2.0, c=SEPIA, k=1.5, over=5.0, dbl=0.3, j=1.2, **kw):
    """A pen rectangle the way a hand draws it: four separate strokes, corners crossing, now and then gone over twice."""
    quad(s, r, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], w, c, k, over, dbl, j, **kw)


def quad(s, r, P, w=2.0, c=SEPIA, k=1.5, over=5.0, dbl=0.3, j=1.2, closed=True, **kw):
    P = [(p[0] + J(r, j), p[1] + J(r, j)) for p in P]
    pairs = list(zip(P, P[1:] + P[:1])) if closed else list(zip(P, P[1:]))
    for a, b in pairs:
        seg(s, a, b, w, c, k, over=over * (0.2 + r.random()), r=r, **kw)
        if r.random() < dbl:
            d = (J(r, 2.2), J(r, 2.2))
            seg(s, (a[0] + d[0], a[1] + d[1]), (b[0] + d[0], b[1] + d[1]), w * 0.7, c, k * 0.6, over=over * r.random(), r=r, skip=0.25)


def line(s, r, a, b, w=2.0, c=SEPIA, k=1.5, over=4.0, dbl=0.0, **kw):
    seg(s, a, b, w, c, k, over=over * (0.3 + r.random()), r=r, **kw)
    if dbl and r.random() < dbl:
        d = (J(r, 2), J(r, 2))
        seg(s, (a[0] + d[0], a[1] + d[1]), (b[0] + d[0], b[1] + d[1]), w * 0.7, c, k * 0.6, r=r, skip=0.3)


def put(s, m, color, k=0.8, lift=0.9, wob=1.2, **kw):
    """An object over what is already painted: lift most of what is under it, lay its wash, keep it opaque."""
    if lift > 0:
        s.lift(m, lift, soft=0.7, wob=0)
    return s.wash(m, color, k, wob=wob, body=1, **kw)


def streaks(s, r, pts_top, length, n, color=PAYNE, k=0.2, w=(5, 16), clip=None, dry=0.45):
    """Rain stains: thin dry-ish vertical drags of dirty water from a ledge (pts_top: list of (x, y) to start from)."""
    for _ in range(n):
        x, ya = pts_top[int(r.integers(0, len(pts_top)))]
        x += J(r, 6)
        yb = ya + U(r, 0.3, 1.0) * length
        s.stroke([(x, ya), (x + J(r, 2), (ya + yb) / 2), (x + J(r, 3), yb)], U(r, *w), color, k * U(r, 0.5, 1.3), dry=dry * U(r, 0.6, 1.2),
                 edge=0.3, wob=1.2, taper=(0.02, 0.7), clip=clip)


def mottle(s, r, x0, y0, x1, y1, colors, n=5, k=0.2, size=(60, 160), clip=None, bleed=18):
    """Wet-in-wet touches of neighbouring colour dropped into a wall while it is damp."""
    for i in range(n):
        c = colors[int(r.integers(0, len(colors)))]
        rx, ry = U(r, *size), U(r, *size)
        s.wash(s.blob(U(r, x0, x1), U(r, y0, y1), rx, ry * 0.8, irr=0.35), c, k * U(r, 0.5, 1.4), wob=4, bleed=bleed * U(r, 0.7, 1.4), edge=0.25, clip=clip)


def hl(s, pts, w=2.0, amt=0.3, dry=0.0, clip=None):
    """A wet highlight: pigment lifted along a line."""
    s.wash(s.ribbon(pts, w, smooth=len(pts) > 2), None, amt, wob=0, soft=0.7, edge=0, gran=0.2, flow=0, lift=True, dry=dry, clip=clip)


def cable(s, r, a, b, sag, w=1.4, c=INKBLUE, k=1.2, **kw):
    mid = ((a[0] + b[0]) / 2 + J(r, 6), (a[1] + b[1]) / 2 + sag)
    q1 = ((a[0] * 3 + b[0]) / 4, (a[1] * 3 + b[1]) / 4 + sag * 0.75)
    q3 = ((a[0] + b[0] * 3) / 4, (a[1] + b[1] * 3) / 4 + sag * 0.75)
    kw.setdefault("tremor", 0.5)
    s.ink([a, q1, mid, q3, b], w, c, k, press=0.2, taper=(0.02, 0.02), dry=0.15, **kw)


def iron(s, m, k=1.0, c=SEPIA, c2=INKBLUE, wob=0.45):
    """Wet black ironwork up close: two pigments, not one flat black."""
    m = s.wash(m, c, 1.05 * k, wob=wob, soft=0.8, edge=0.4, ew=4, flow=0.45, fs=70, gran=0.4, body=1)
    s.wash(m, c2, 0.7 * k, wob=1.5, soft=1.5, edge=0.2, flow=0.8, fs=40, gran=0.4, clip=m)
    return m


def bump(s, y0, y1, f=0.5):
    ym = y0 + (y1 - y0) * f
    return s.vgrad(y0, ym, 0, 1) * s.vgrad(ym, y1, 1, 0)


def cut(s, hole):
    """Make glass: clear the pigment and the body inside a mask."""
    s.A *= (1 - hole)[..., None]
    s.body *= (1 - hole)


# ---------------------------------------------------------------------------------------------- the camera
class Cam:
    """A pinhole camera at `pos` (metres; x right, y up, z down the lane), pitched up by `pitch`, lens shifted to (cx, cy)."""

    def __init__(self, f, cx, cy, pos, pitch):
        self.f, self.cx, self.cy = f, cx, cy
        self.pos = np.array(pos, float)
        self.cp, self.sp = math.cos(pitch), math.sin(pitch)

    def __call__(self, P):
        P = np.asarray(P, float) - self.pos
        X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
        yc = Y * self.cp - Z * self.sp
        zc = Y * self.sp + Z * self.cp
        return np.stack([self.cx + self.f * X / zc, self.cy - self.f * yc / zc], -1)

    def depth(self, P):
        P = np.asarray(P, float) - self.pos
        return P[..., 1] * self.sp + P[..., 2] * self.cp


class Fac:
    """A facade plane: s runs along it from its near end A to its far end B, y up, d out of the wall into the lane."""

    def __init__(self, cam, A, B, side):
        self.cam, self.side = cam, side
        A, B = np.array(A, float), np.array(B, float)
        self.o = A
        self.len = float(np.linalg.norm(B - A))
        self.es = (B - A) / self.len
        self.ey = np.array([0, 1.0, 0])
        en = np.cross(self.ey, self.es) * side
        self.en = en / np.linalg.norm(en)
        src = np.float32([[0, 0], [10, 0], [10, 10], [0, 10]])
        dst = np.float32([self.p(*q) for q in src])
        self.Hm = cv2.getPerspectiveTransform(src, dst)
        self.Hi = np.linalg.inv(self.Hm)

    def w3(self, sv, y, d=0.0):
        return self.o + sv * self.es + y * self.ey + d * self.en

    def p(self, sv, y, d=0.0):
        return tuple(float(v) for v in self.cam(self.w3(sv, y, d)))

    def q(self, s0, y0, s1, y1, d=0.0):
        """Quad of a rectangle on a plane parallel to the wall: bottom-left, bottom-right, top-right, top-left."""
        return [self.p(s0, y0, d), self.p(s1, y0, d), self.p(s1, y1, d), self.p(s0, y1, d)]

    def qs(self, s, y0, y1, d0, d1):
        """Quad of a rectangle across the wall at s (a side face): wall-bottom, out-bottom, out-top, wall-top."""
        return [self.p(s, y0, d0), self.p(s, y0, d1), self.p(s, y1, d1), self.p(s, y1, d0)]

    def qh(self, s0, s1, y, d0, d1):
        """Quad of a horizontal face at height y (a soffit or a slab): wall-near, wall-far, out-far, out-near."""
        return [self.p(s0, y, d0), self.p(s1, y, d0), self.p(s1, y, d1), self.p(s0, y, d1)]

    def off(self, d):
        """The same facade moved d metres out (a parapet front, a tin room set back with d < 0)."""
        return Fac(self.cam, self.o + d * self.en, self.o + d * self.en + self.len * self.es, self.side)

    def ppm(self, sv, y, d=0.0):
        """Screen pixels per metre at a point of the wall."""
        return self.cam.f / float(self.cam.depth(self.w3(sv, y, d)))

    def sy(self, sub):
        """The wall coordinates (s, y) of every pixel of a Sub, by the inverse homography of the wall plane."""
        X = sub.X + sub.ox
        Y = sub.Y + sub.oy
        Hi = self.Hi
        w = Hi[2, 0] * X + Hi[2, 1] * Y + Hi[2, 2]
        sv = (Hi[0, 0] * X + Hi[0, 1] * Y + Hi[0, 2]) / w
        yv = (Hi[1, 0] * X + Hi[1, 1] * Y + Hi[1, 2]) / w
        return sv.astype(np.float32), yv.astype(np.float32)


CAM = Cam(3400.0, 1536.0, 1060.0, (0.0, 1.5, 0.0), math.radians(6.0))
FL_L = [0.0, 3.6, 6.7, 9.8, 12.9, 16.0]          # slab heights, left block (5 storeys)
FL_R = [0.0, 3.5, 6.6, 9.7, 12.8]                # right block (4 storeys; a tin room on its roof)
LF = Fac(CAM, (-13.43, 0, 27.1), (-1.7, 0, 52.0), 1)
RF = Fac(CAM, (12.96, 0, 25.6), (1.9, 0, 53.0), -1)
UL = [6.3, 5.6, 5.1, 5.5, 5.0]                   # house widths along each facade, near to far
UR = [7.0, 5.9, 5.2, 5.7, 5.7]
TIN = (7.6, 18.1, 2.9)                           # the tin room on the right block's roof: s0, s1, height


# ---------------------------------------------------------------------------------------------- sky
def _clouds(t, r, clouds, c1, c2, c3, kmul=1.0):
    for cx, cy, rx, ry, k in clouds:
        t.wash(t.blob(cx, cy, rx, ry, irr=0.42, n=15), c1, k * kmul, wob=9, bleed=34, edge=0.2, ew=10, flow=0.5, fs=200, gran=0.25)
        t.wash(t.blob(cx + J(r, 60), cy + ry * 0.55, rx * 0.72, ry * 0.42, irr=0.4, n=13), c2, k * 0.9 * kmul, wob=6, bleed=16, edge=0.5, ew=8, flow=0.4, fs=120, gran=0.3)
        t.wash(t.blob(cx + J(r, 100), cy + ry * 0.9, rx * 0.5, ry * 0.26, irr=0.4), c3, k * 0.35 * kmul, wob=6, bleed=22, edge=0.1, gran=0.2)


CLOUDS = [(380, 120, 640, 90, 0.36), (1500, 70, 780, 80, 0.32), (2660, 150, 600, 90, 0.36), (1040, 300, 520, 60, 0.26),
          (2080, 330, 600, 56, 0.26), (2900, 420, 360, 44, 0.2), (180, 400, 380, 44, 0.2), (1560, 520, 560, 50, 0.22),
          (700, 620, 420, 40, 0.16), (2400, 640, 460, 40, 0.16)]


def sky():
    s = Sheet(W, H, 51)
    r = rs(51)
    F = s.full()
    s.glaze(INDIGO, 0.62, flow=0.35, fs=700, gran=0.06)
    s.wash(F, PRUSSIAN, 0.55, wob=0, soft=0, edge=0, grad=s.vgrad(-200, 900, 1, 0.0), flow=0.45, fs=420, gran=0.06)
    s.wash(F, PAYNE, 0.5, wob=0, soft=0, edge=0, grad=s.vgrad(-100, 500, 1, 0), flow=0.5, fs=260, gran=0.06)
    s.wash(F, VIOLET, 0.18, wob=0, soft=0, edge=0, grad=bump(s, 450, 1250, 0.5), flow=0.7, fs=380, gran=0.06)
    # the city's sodium light from below: the indigo thinned out low down, then warmed
    s.wash(F, None, 0.5, wob=0, soft=0, edge=0, gran=0, flow=0.5, fs=300, grad=s.vgrad(850, 1600, 0, 1), lift=True)
    s.wash(F, ROSE, 0.12, wob=0, soft=0, edge=0, grad=bump(s, 750, 1500, 0.6), flow=0.7, fs=300, gran=0.05)
    s.wash(F, AMBER, 0.34, wob=0, soft=0, edge=0, grad=s.vgrad(1000, 1650, 0, 1), flow=0.6, fs=260, gran=0.05)
    s.wash(F, SIENNA, 0.12, wob=0, soft=0, edge=0, grad=s.vgrad(1250, 1728, 0, 1), flow=0.6, fs=200, gran=0.05)
    for cx, cy, rx, ry, k in CLOUDS:
        s.wash(s.blob(cx, cy, rx, ry, irr=0.42, n=15), PAYNE, k * 0.9, wob=9, bleed=34, edge=0.25, ew=10, flow=0.5, fs=200, gran=0.12)
        s.wash(s.blob(cx + J(r, 60), cy + ry * 0.55, rx * 0.72, ry * 0.42, irr=0.4, n=13), INDIGO, k * 0.7, wob=6, bleed=16, edge=0.5, ew=8, flow=0.4, fs=120, gran=0.12)
    for cx, cy, rx, ry, a in [(900, 210, 300, 40, 0.2), (2140, 240, 340, 40, 0.2), (1500, 420, 300, 30, 0.18), (2560, 520, 420, 34, 0.16),
                              (520, 520, 420, 34, 0.16)]:
        s.lift(s.blob(cx, cy, rx, ry, irr=0.4, n=13), a, bleed=26, wob=7)
    # the undersides of the low cloud, warmed by the street lights
    for cx, cy, rx, ry in [(1500, 780, 640, 50), (700, 840, 500, 40), (2350, 820, 560, 46), (1550, 1010, 700, 50)]:
        s.wash(s.blob(cx, cy - ry * 0.6, rx * 0.9, ry * 0.7, irr=0.45, n=14), PAYNE, 0.22, wob=8, bleed=22, edge=0.35, gran=0.1)
        s.wash(s.blob(cx, cy + ry * 0.3, rx * 0.8, ry * 0.5, irr=0.45, n=14), OCHRE, 0.16, wob=8, bleed=26, edge=0.1, gran=0.1)
    s.save("bk_sky", opaque=True, q=88)


def sky_dawn():
    s = Sheet(W, H, 52)
    r = rs(52)
    F = s.full()
    s.wash(F, RAIN, 0.62, wob=0, soft=0, edge=0, grad=s.vgrad(-200, 950, 1, 0.0), flow=0.45, fs=420, gran=0.08)
    s.wash(F, SLATE, 0.3, wob=0, soft=0, edge=0, grad=s.vgrad(-100, 500, 1, 0), flow=0.5, fs=260, gran=0.08)
    s.wash(F, DUSK, 0.16, wob=0, soft=0, edge=0, grad=bump(s, 350, 1050, 0.5), flow=0.7, fs=380, gran=0.08)
    s.wash(F, SKIN, 0.5, wob=0, soft=0, edge=0, grad=s.vgrad(800, 1500, 0, 1), flow=0.7, fs=260, gran=0.08)
    s.wash(F, ROSE, 0.1, wob=0, soft=0, edge=0, grad=bump(s, 750, 1400, 0.5), flow=0.7, fs=300, gran=0.08)
    s.wash(F, OCHRE, 0.16, wob=0, soft=0, edge=0, grad=s.vgrad(1150, 1728, 0, 1), flow=0.7, fs=260, gran=0.08)
    for cx, cy, rx, ry, k in CLOUDS[:7]:
        s.wash(s.blob(cx, cy, rx, ry, irr=0.42, n=15), SLATE, k * 0.75, wob=9, bleed=34, edge=0.25, ew=10, flow=0.5, fs=200, gran=0.12)
        s.wash(s.blob(cx + J(r, 60), cy + ry * 0.55, rx * 0.72, ry * 0.42, irr=0.4, n=13), DUSK, k * 0.6, wob=6, bleed=16, edge=0.5, ew=8, flow=0.4, fs=120, gran=0.12)
    for cx, cy, rx, ry in [(1500, 760, 640, 40), (700, 820, 500, 34), (2350, 800, 560, 38), (1550, 990, 700, 40)]:
        s.wash(s.blob(cx, cy - ry * 0.4, rx * 0.9, ry * 0.7, irr=0.45, n=14), DUSK, 0.16, wob=8, bleed=20, edge=0.35, gran=0.1)
        s.wash(s.blob(cx, cy + ry * 0.3, rx * 0.8, ry * 0.45, irr=0.45, n=14), ROSE, 0.12, wob=8, bleed=20, edge=0.3, gran=0.1)
    for cx, cy, rx, ry, a in [(900, 210, 300, 40, 0.3), (2140, 240, 340, 40, 0.3), (1500, 420, 300, 30, 0.26), (2560, 520, 420, 34, 0.22),
                              (520, 520, 420, 34, 0.22)]:
        s.lift(s.blob(cx, cy, rx, ry, irr=0.4, n=13), a, bleed=26, wob=7)
    s.wash(s.full(), None, 0.3, wob=0, soft=0, edge=0, gran=0, flow=0.6, fs=300, grad=s.vgrad(1200, 1728, 0, 1), lift=True)
    s.save("bk_sky_dawn", opaque=True, q=88)


def ctr(s, m, w=2.2, color=INK, k=1.5, level=0.5, step=5.0, **kw):
    """kit.contour for a Sub: pen the outline of a mask given in the Sub's own pixels."""
    cs, _ = cv2.findContours((m > level).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cs:
        P = c[:, 0, :].astype(np.float64)
        if len(P) < 12:
            continue
        P = P[::max(1, int(step))]
        P = np.vstack([P, P[:2]])
        Pm = (np.roll(P, 1, 0) + P + np.roll(P, -1, 0)) / 3
        Pm[0], Pm[-1] = P[0], P[-1]
        s.ink(Pm + (getattr(s, "ox", 0), getattr(s, "oy", 0)), w, color, k, smooth=False, taper=(0.02, 0.04), **kw)


# ---------------------------------------------------------------------------------------------- the plan of the blocks
GREY2 = (0.80, 0.80, 0.74)        # an older, greyer celadon
DOOR_RED = (0.72, 0.30, 0.26)
DOOR_GREEN = (0.36, 0.56, 0.46)
FRAMES = [MIST, MIST, (0.80, 0.80, 0.78), (0.62, 0.52, 0.42)]      # aluminium, mostly silver, now and then bronze


def plan(side):
    """Geometry only (no paint): every house, floor, window, balcony and cage of one block, in wall metres.
    Shared by the walls, the glass layers and the figures, so it must not depend on anything painted."""
    if side == "L":
        F, FL, widths, seed = LF, FL_L, UL, 61
        tiles = [PINK, CELADON, TILE, PINK, GREY2]
        kinds = [["win", "bal", "win", "win"], ["bal", "win", "win", "bal"], ["win", "mine", "bal", "win"],
                 ["win", "bal", "win", "win"], ["win", "win", "bal", "win"]]
        ground = ["shutter", "door", "shutter", "door", "shutter"]
    else:
        F, FL, widths, seed = RF, FL_R, UR, 62
        tiles = [TILE, PINK, CELADON, GREY2, PINK]
        kinds = [["win", "bal", "win"], ["bal", "win", "bal"], ["win", "yours", "win"], ["bal", "win", "win"], ["win", "win", "bal"]]
        ground = ["door", "shutter", "shutter", "door", "shutter"]
    r = rs(seed)
    units, wins = [], []
    s0 = 0.0
    for i, uw in enumerate(widths):
        s1 = s0 + uw
        u = dict(i=i, s0=s0, s1=s1, tile=tiles[i], ground=ground[i], floors=[], seed=seed * 100 + i)
        for f, kind in enumerate(kinds[i]):
            yf, yt = FL[f + 1], FL[f + 2]
            fl = dict(kind=kind, yf=yf, yt=yt, n=f + 2, wins=[])
            if kind == "bal":
                a, b = s0 + U(r, 0.3, 0.45), s1 - U(r, 0.3, 0.45)
                fl.update(a=a, b=b, dd=U(r, 0.85, 1.05), laundry=r.random() < 0.75, plants=int(r.integers(0, 4)), rail=r.random() < 0.5)
                ga = a + U(r, 0.25, 0.6)
                gb = min(b - 0.5, ga + U(r, 2.2, 2.7))
                fl["wins"].append(dict(a=ga, b=gb, y0=yf + 0.08, y1=yf + 2.15, frame=FRAMES[int(r.integers(0, 4))], tr=False, door=True))
                if b - gb > 1.3:
                    wa = gb + 0.35
                    fl["wins"].append(dict(a=wa, b=min(b - 0.3, wa + 1.1), y0=yf + 1.0, y1=yf + 2.1, frame=FRAMES[int(r.integers(0, 4))], tr=False, door=False))
            else:
                ww = 2.15 if kind != "win" else U(r, 1.7, 2.3)
                if kind == "mine":
                    a = s0 + 1.4
                elif kind == "yours":
                    a = s0 + 1.55
                else:
                    a = s0 + U(r, 0.6, max(0.65, uw - ww - 0.6))
                b = a + ww
                y0 = yf + U(r, 0.9, 1.0)
                y1 = y0 + U(r, 1.32, 1.5)
                fl["wins"].append(dict(a=a, b=b, y0=y0, y1=y1, frame=FRAMES[int(r.integers(0, 4))], tr=r.random() < 0.35, door=False))
                cage = kind != "win" or r.random() < 0.8
                fl.update(a=a, b=b, cage=cage, dd=U(r, 0.36, 0.48), plants=int(r.integers(0, 4)) if kind == "win" else 2,
                          vine=r.random() < 0.3 and kind == "win", roof=r.random() < 0.6, deco=int(r.integers(0, 3)), rust=r.random() < 0.5)
                ac = None
                if kind == "win":
                    q = r.random()
                    if q < 0.35 and s1 - b > 1.1:
                        ac = ("box", b + 0.3, y0 + 0.25)
                    elif q < 0.6 and a - s0 > 1.2:
                        ac = ("split", a - 1.05, yf + 0.25)
                    elif q < 0.75:
                        ac = ("under", (a + b) / 2 - 0.35, y0 - 0.7)
                fl["ac"] = ac
                # a bathroom's small louvred window at the other side, now and then
                room = (s1 - b) if (ac is None or ac[0] != "box") else 0
                if room > 1.2 and r.random() < 0.55:
                    ba = b + U(r, 0.45, room - 0.75)
                    fl["wins"].append(dict(a=ba, b=ba + 0.55, y0=yf + 1.7, y1=yf + 2.25, frame=MIST, tr=False, door=False, bath=True))
            for w in fl["wins"]:
                w.setdefault("bath", False)
                w["idx"] = len(wins)
                w["floor"] = fl["n"]
                w["unit"] = i
                wins.append(w)
            u["floors"].append(fl)
        units.append(u)
        s0 = s1
    return F, FL, units, wins


def panes(w):
    """The glass of one window in wall metres [(a, y0, b, y1), ...] and its bars."""
    a, b, y0, y1 = w["a"], w["b"], w["y0"], w["y1"]
    fw = 0.055
    if w["bath"]:
        return [(a + 0.04, y0 + 0.04, b - 0.04, y1 - 0.04)]
    m = (a + b) / 2
    ys = [(y0 + fw, y1 - fw)]
    if w["tr"]:
        ys = [(y0 + fw, y1 - 0.42), (y1 - 0.37, y1 - fw)]
    out = []
    for ya, yb in ys:
        out.append((a + fw, ya, m - 0.025, yb))
        out.append((m + 0.025, ya, b - fw, yb))
    return out


def glass_quad(F, w):
    """The whole glass opening of a window at the glass plane, as a screen quad."""
    a, b, y0, y1 = w["a"], w["b"], w["y0"], w["y1"]
    return F.q(a + 0.05, y0 + 0.05, b - 0.05, y1 - 0.05, -0.1)


def pw(F, sv, y, d=0.0, base=0.02, lo=1.0, hi=2.8):
    return float(np.clip(F.ppm(sv, y, d) * base, lo, hi))


def addhole(S, s, m):
    sl = (slice(s.oy - getattr(S, "oy", 0), s.oy - getattr(S, "oy", 0) + s.h), slice(s.ox - getattr(S, "ox", 0), s.ox - getattr(S, "ox", 0) + s.w))
    S.hole[sl] = np.maximum(S.hole[sl], m)


def tilework(s, r, F, m, sv, yv, ts=0.11, k=0.26, patch=(0.48, 0.66), seed=0, odd=(SLATE, UMBER, ROSE), wobble=0.4, lift=0.25, oddk=0.2):
    """Small square tiles: a grout grid said in patches, and single tiles touched lighter or darker, in perspective."""
    h, w = s.h, s.w
    sv = sv + (fbm(h, w, 34, seed + 1, 2) - 0.5) * ts * wobble
    yv = yv + (fbm(h, w, 34, seed + 2, 2) - 0.5) * ts * wobble
    gsx = np.hypot(*np.gradient(sv)) + 1e-7
    gsy = np.hypot(*np.gradient(yv)) + 1e-7
    a, b = sv / ts, yv / ts
    ds = np.abs(a - np.round(a)) * ts / gsx
    dy = np.abs(b - np.round(b)) * ts / gsy
    ln = np.maximum(np.clip(1 - ds / 0.7, 0, 1), np.clip(1 - dy / 0.7, 0, 1))
    fade = sstep(3.0, 6.5, ts / np.maximum(gsx, gsy))
    pm = sstep(patch[0], patch[1], fbm(h, w, 70, seed + 3, 3))
    pm = np.maximum(pm, 0.06)
    dry = sstep(0.35, 0.75, s.P + 0.3 * fbm(h, w, 6, seed + 4, 2))
    s.wash((ln * fade * pm * dry * m).astype(np.float32), SEPIA, k, wob=0, soft=0.0, edge=0, gran=0, flow=0.3, fs=30)
    i, j = np.floor(a).astype(np.int64), np.floor(b).astype(np.int64)
    hh = (((i * 73856093) ^ (j * 19349663) ^ (seed * 83492791)) & 1023) / 1023.0
    hh2 = (((i * 19349663) ^ (j * 83492791) ^ (seed * 73856093)) & 1023) / 1023.0
    cl = sstep(0.3, 0.7, fbm(h, w, 120, seed + 5, 2))
    inner = (1 - ln) * m * fade
    for c_i, c in enumerate(odd):
        sel = ((hh > 0.90) & (np.floor(hh2 * len(odd)) == c_i)).astype(np.float32)
        s.wash((sel * inner * cl).astype(np.float32), c, oddk, wob=0, soft=0.4, edge=0, gran=0.2, flow=0.4, fs=40)
    s.wash(((hh < 0.07).astype(np.float32) * inner * cl).astype(np.float32), None, lift, wob=0, soft=0.4, edge=0, gran=0, flow=0, lift=True)


def vline(s, r, F, sv, y0, y1, d=0.0, w=1.6, c=SEPIA, k=1.3, **kw):
    line(s, r, F.p(sv, y0, d), F.p(sv, y1, d), w, c, k, **kw)


def hline(s, r, F, s0, s1, y, d=0.0, w=1.6, c=SEPIA, k=1.3, **kw):
    line(s, r, F.p(s0, y, d), F.p(s1, y, d), w, c, k, **kw)


# ---------------------------------------------------------------------------------------------- pass 1: the walls
def p1_unit(S, F, FL, u, last):
    r = rs(u["seed"])
    s0, s1 = u["s0"], u["s1"]
    top = FL[-1]
    ext = F.q(s0 - 0.3, -0.8, s1 + 0.3, top + 1.3) + F.q(s0 - 0.3, -0.8, s1 + 0.3, top + 1.3, 1.2)
    s = sub_of(S, ext, 40)
    sv, yv = F.sy(s)
    col = u["tile"]
    m = Q(s, F.q(s0, -0.8, s1, top + 1.0))
    s.lift(m, 1.0, soft=0.5, wob=0)
    m = s.wash(m, col, 0.66, wob=0.8, soft=0.7, edge=0.35, ew=6, flow=0.5, fs=220, gran=0.32, body=1)
    yn = np.clip(yv / top, 0, 1)
    s.wash(m, DUSK, 0.36, wob=0, soft=0, edge=0, grad=(1 - 0.75 * yn).astype(np.float32), flow=0.6, fs=160, gran=0.25)     # the lane is dim, the top sees the sky
    s.wash(m, PAYNE, 0.2, wob=0, soft=0, edge=0, grad=sstep(FL[1] + 0.6, FL[1] - 1.2, yv).astype(np.float32), flow=0.6, fs=140, gran=0.3)
    s.wash(m, col, 0.2, wob=3, bleed=10, edge=0.15, flow=0.7, fs=120, gran=0.4, clip=m)
    bb = np.asarray(F.q(s0, 0, s1, top), float)
    mottle(s, r, bb[:, 0].min(), bb[:, 1].min(), bb[:, 0].max(), bb[:, 1].max(), [DUSK, SLATE, col, ROSE if col is PINK else TEAL], 6, 0.12, size=(50, 130), clip=m)
    tm = m * sstep(FL[1] + 0.1, FL[1] + 0.4, yv)
    tilework(s, r, F, tm, sv, yv, 0.105 if col is not PINK else 0.12, seed=u["seed"])
    # bare cement where tiles have fallen away
    for _ in range(int(r.integers(0, 3))):
        cs, cy = U(r, s0 + 0.5, s1 - 0.8), U(r, FL[1] + 0.6, top - 0.5)
        P = [F.p(cs + 0.35 * math.cos(t) * U(r, 0.6, 1.3), cy + 0.28 * math.sin(t) * U(r, 0.6, 1.3)) for t in np.linspace(0, 2 * math.pi, 9)[:-1]]
        pm = Q(s, P)
        s.wash(pm, GREY, 0.45, wob=1.5, soft=0.8, edge=0.5, ew=3, gran=0.5)
        s.wash(pm, SEPIA, 0.12, wob=2, soft=1, edge=0.3, flow=0.8, fs=20)
        quad(s, r, P, 1.0, SEPIA, 0.8, over=1, dbl=0, j=0.5, skip=0.3)
    # slab bands
    for y in FL[1:]:
        bm = Q(s, F.q(s0, y - 0.12, s1, y + 0.16))
        s.wash(bm, GREY, 0.3, wob=0.8, soft=0.8, edge=0.4, ew=3, flow=0.5, fs=60)
        s.wash(bm, SLATE, 0.12, wob=1, soft=1, edge=0.2)
        hline(s, r, F, s0, s1, y - 0.12, 0, pw(F, s0, y, 0, 0.016), SEPIA, 1.2, over=6, dbl=0.4)
        hline(s, r, F, s0, s1, y + 0.16, 0, pw(F, s0, y, 0, 0.011), SEPIA, 0.9, over=3, skip=0.25)
        s.wash(Q(s, F.q(s0, y - 0.5, s1, y - 0.12)), PAYNE, 0.1, wob=2, bleed=6, edge=0.1, grad=None)
    # the column strip at the party wall, and the edge of the house
    cm = Q(s, F.q(s0, 0, s0 + 0.24, top + 1.0))
    s.wash(cm, SLATE, 0.16, wob=1, soft=1, edge=0.3, ew=3)
    vline(s, r, F, s0, -0.5, top + 1.0, 0, pw(F, s0, 6, 0, 0.02), SEPIA, 1.4, dbl=0.4, over=8)
    vline(s, r, F, s0 + 0.24, 0, top + 1.0, 0, pw(F, s0, 6, 0, 0.012), SEPIA, 0.9, skip=0.3, over=2)
    if last:
        vline(s, r, F, s1, -0.5, top + 1.0, 0, pw(F, s1, 6, 0, 0.02), SEPIA, 1.5, dbl=0.5, over=8)
    # parapet and its coping
    cop = Q(s, F.q(s0, top + 0.92, s1, top + 1.08, 0.06))
    put(s, cop, GREY, 0.4, lift=0.8, wob=0.6, edge=0.4, ew=2)
    hl(s, [F.p(s0, top + 1.05, 0.06), F.p(s1, top + 1.05, 0.06)], 3.0, 0.3)
    hline(s, r, F, s0, s1, top + 1.08, 0.06, pw(F, s0, top, 0, 0.02), SEPIA, 1.4, dbl=0.4, over=6)
    hline(s, r, F, s0, s1, top + 0.92, 0.06, pw(F, s0, top, 0, 0.012), SEPIA, 1.0, skip=0.2)
    s.wash(Q(s, F.q(s0, top + 0.5, s1, top + 0.92)), PAYNE, 0.12, wob=2, bleed=6, edge=0.1)
    streaks(s, r, [F.p(U(r, s0, s1), top + 0.9) for _ in range(8)], 0.25 * F.ppm(s0, top) * 1.5, 6, PAYNE, 0.16, (4, 10), clip=m)
    # floors: windows set into the wall
    for fl in u["floors"]:
        if fl["kind"] == "bal":
            a, b = fl["a"], fl["b"]
            s.wash(Q(s, F.q(a, fl["yf"], b, fl["yt"])), PAYNE, 0.16, wob=1.5, bleed=8, edge=0.1, grad=s.vgrad(F.p(a, fl["yf"])[1], F.p(a, fl["yt"])[1], 0.3, 1))
        else:
            # shade under where the hood / cage roof will be
            a, b = fl["a"], fl["b"]
            y1 = fl["wins"][0]["y1"]
            s.wash(Q(s, F.q(a - 0.3, y1 - 0.2, b + 0.3, y1 + 0.35)), DUSK, 0.18, wob=2, bleed=8, edge=0.1)
        for w in fl["wins"]:
            p1_window(S, s, r, F, w, col)
    p1_ground(S, s, r, F, u, FL, m)
    return s


def p1_window(S, s, r, F, w, col):
    a, b, y0, y1 = w["a"], w["b"], w["y0"], w["y1"]
    dg = -0.1
    sc = F.ppm(a, y0)
    pen = float(np.clip(sc * 0.02, 1.0, 2.6))
    jam = Q(s, F.qs(b, y0, y1, 0, dg))
    head = Q(s, F.qh(a, b, y1, 0, dg))
    put(s, np.maximum(jam, head), col, 0.5, lift=0.9, wob=0.4, edge=0.4, ew=2)
    s.wash(jam, PAYNE, 0.3, wob=0.5, soft=0.8, edge=0.2)
    s.wash(head, PAYNE, 0.5, wob=0.5, soft=0.8, edge=0.2)
    fm = Q(s, F.q(a, y0, b, y1, dg))
    put(s, fm, w["frame"], 0.5, lift=0.9, wob=0.4, edge=0.45, ew=2)
    s.wash(fm, SLATE, 0.2, wob=0.5, soft=1, edge=0, grad=s.vgrad(F.p(a, y1)[1], F.p(a, y0)[1], 1, 0.3))
    hole = s.zeros()
    for (pa, py0, pb, py1) in panes(w):
        hole = np.maximum(hole, Q(s, F.q(pa, py0, pb, py1, dg)))
    if w["door"]:          # a sliding door's lower panels are solid
        sol = Q(s, F.q(a, y0, b, y0 + 0.45, dg))
        hole *= 1 - sol
    addhole(S, s, hole)
    # pen: the opening, the frame's inner edge, the meeting stiles
    quad(s, r, F.q(a, y0, b, y1, 0), pen, SEPIA, 1.4, over=4, dbl=0.3, j=0.6)
    if not w["bath"]:
        m = (a + b) / 2
        vline(s, r, F, m - 0.025, y0 + 0.05, y1 - 0.05, dg, pen * 0.8, SEPIA, 1.2, over=1)
        vline(s, r, F, m + 0.025, y0 + 0.05, y1 - 0.05, dg, pen * 0.6, SEPIA, 1.0, over=1, skip=0.2)
        if w["tr"]:
            hline(s, r, F, a, b, y1 - 0.4, dg, pen * 0.8, SEPIA, 1.1, over=1)
        if w["door"]:
            hline(s, r, F, a, b, y0 + 0.45, dg, pen * 0.8, SEPIA, 1.1, over=1)
    quad(s, r, F.q(a + 0.05, y0 + 0.05, b - 0.05, y1 - 0.05, dg), pen * 0.6, SEPIA, 0.9, over=1, dbl=0, j=0.5, skip=0.2)
    # sill
    if not w["door"]:
        sm = Q(s, F.q(a - 0.07, y0 - 0.08, b + 0.07, y0, 0.05))
        put(s, sm, GREY, 0.35, lift=0.8, wob=0.4, edge=0.4, ew=2)
        hline(s, r, F, a - 0.08, b + 0.08, y0 - 0.08, 0.05, pen, SEPIA, 1.3, over=2)
        st = [F.p(U(r, a, b), y0 - 0.08) for _ in range(5)]
        streaks(s, r, st, sc * U(r, 0.6, 1.6), int(r.integers(2, 5)), PAYNE, 0.2, (sc * 0.04, sc * 0.12), dry=0.4)


def shutter(s, r, F, a, b, h, col=SLATE, k=0.6):
    sc = F.ppm(a, 1.5)
    pen = float(np.clip(sc * 0.018, 1.0, 2.4))
    m = Q(s, F.q(a, 0, b, h, -0.05))
    put(s, m, col, k, lift=0.9, wob=0.8, edge=0.4, ew=4, flow=0.5, fs=60, gran=0.45)
    s.wash(m, PAYNE, 0.3, wob=1, soft=2, edge=0, grad=s.vgrad(F.p(a, h)[1], F.p(a, 0)[1], 0.2, 1.0), flow=0.6, fs=60)
    y = 0.1
    while y < h - 0.05:
        hline(s, r, F, a + 0.02, b - 0.02, y, -0.05, U(r, 0.6, 1.0) * pen * 0.6, SEPIA, U(r, 0.5, 0.9), over=0, skip=0.15)
        y += 0.09 * U(r, 0.9, 1.1)
    st = [F.p(U(r, a, b), h) for _ in range(6)]
    streaks(s, r, st, sc * 1.6, 5, PAYNE, 0.16, (sc * 0.05, sc * 0.14), clip=m)
    hm = Q(s, F.q(a - 0.08, h, b + 0.08, h + 0.32, 0.12))
    put(s, hm, col, k * 0.9, lift=0.85, wob=0.6, edge=0.5, ew=3)
    s.wash(hm, PAYNE, 0.3, wob=1, soft=1.5, edge=0)
    quad(s, r, F.q(a - 0.08, h, b + 0.08, h + 0.32, 0.12), pen, SEPIA, 1.4, over=3, dbl=0.3)
    quad(s, r, F.q(a, 0, b, h, -0.05), pen, SEPIA, 1.5, over=4, dbl=0.4)
    lk = (a + b) / 2 + J(r, 0.3)
    put(s, Q(s, F.q(lk - 0.06, 0.05, lk + 0.06, 0.18, -0.04)), SEPIA, 0.7, lift=0.3, wob=0.3)


def p1_ground(S, s, r, F, u, FL, m):
    s0, s1 = u["s0"], u["s1"]
    g = FL[1]
    sc = F.ppm(s0, 1.5)
    pen = float(np.clip(sc * 0.02, 1.0, 2.6))
    gm = Q(s, F.q(s0, -0.8, s1, g - 0.15))
    s.wash(gm, PAYNE, 0.22, wob=1.5, bleed=8, edge=0.15, flow=0.6, fs=120)
    s.wash(gm, SEPIA, 0.12, wob=2, bleed=10, edge=0.1, grad=s.vgrad(F.p(s0, 1.0)[1], F.p(s0, 0)[1], 0, 1), flow=0.7, fs=90)
    if u["ground"] == "door":
        da, db = s0 + 0.45, s0 + 1.45
        dcol = DOOR_RED if u["i"] % 2 else DOOR_GREEN
        dm = Q(s, F.q(da, 0, db, 2.25, -0.1))
        put(s, dm, dcol, 0.85, lift=0.9, wob=0.8, edge=0.5, ew=4, flow=0.5, fs=50, gran=0.45)
        s.wash(dm, PAYNE, 0.25, wob=1, soft=2, edge=0, grad=s.vgrad(F.p(da, 2.2)[1], F.p(da, 0)[1], 0.2, 1))
        jam = Q(s, F.qs(db, 0, 2.25, 0, -0.1))
        put(s, jam, GREY, 0.5, lift=0.8, wob=0.4)
        s.wash(jam, PAYNE, 0.35, wob=0.4, soft=0.8)
        quad(s, r, F.q(da, 0, db, 2.25, -0.1), pen, SEPIA, 1.5, over=4, dbl=0.4)
        gw = F.q(da + 0.2, 1.25, db - 0.2, 1.95, -0.1)                 # a small barred window in the door
        s.wash(Q(s, gw), INKBLUE, 0.6, wob=0.5, soft=0.8, edge=0.4, ew=2)
        quad(s, r, gw, pen * 0.8, SEPIA, 1.3, over=2)
        for q in np.linspace(0.12, 0.88, 6):
            vline(s, r, F, da + 0.2 + q * 0.6, 1.25, 1.95, -0.1, pen * 0.7, SEPIA, 1.2, over=0)
        quad(s, r, F.q(da + 0.15, 0.2, db - 0.15, 1.05, -0.1), pen * 0.7, SEPIA, 1.0, over=2, dbl=0.2)
        put(s, Q(s, F.q(db - 0.18, 1.0, db - 0.1, 1.12, -0.08)), OCHRE, 0.5, lift=0.6, wob=0.2)
        # the blue door-number plate and the mailboxes
        put(s, Q(s, F.q(db + 0.12, 2.0, db + 0.42, 2.18, 0.01)), PRUSSIAN, 0.8, lift=0.8, wob=0.3, edge=0.4, ew=1)
        hl(s, [F.p(db + 0.16, 2.09, 0.01), F.p(db + 0.38, 2.09, 0.01)], 1.2, 0.5)
        mx = db + 0.2
        for i in range(3):
            for j in range(2):
                q = F.q(mx + i * 0.26, 1.15 + j * 0.3, mx + i * 0.26 + 0.24, 1.15 + j * 0.3 + 0.27, 0.12)
                put(s, Q(s, q), (GREY, RUST, SLATE)[(i + j) % 3], 0.5, lift=0.8, wob=0.3, edge=0.5, ew=1)
                quad(s, r, q, pen * 0.6, SEPIA, 1.1, over=1, dbl=0, j=0.4)
                hline(s, r, F, mx + i * 0.26 + 0.04, mx + i * 0.26 + 0.2, 1.15 + j * 0.3 + 0.2, 0.12, pen * 0.5, SEPIA, 1.0, over=0)
        a = db + 1.1
        if s1 - 0.4 - a > 1.2:
            shutter(s, r, F, a, s1 - 0.4, 2.7, (SLATE, GREY, RAIN)[u["i"] % 3])
    else:
        shutter(s, r, F, s0 + 0.5, s1 - 0.45, 2.75, (SLATE, RUST, GREY, RAIN)[u["i"] % 4], 0.6 if u["i"] % 4 != 1 else 0.45)
    s.wash(Q(s, F.q(s0, -0.8, s1, 0.5)), SEPIA, 0.3, wob=3, bleed=10, edge=0.1, clip=m)            # splash at the foot


# ---------------------------------------------------------------------------------------------- pass 2: what stands out of the walls
def plant_pot(s, r, F, sv, y, d, size=1.0, trail=False, pen=SEPIA):
    """A pot of something leafy standing on a ledge at (sv, y, d)."""
    sc = F.ppm(sv, y, d)
    x, yy = F.p(sv, y, d)
    pw_ = 0.22 * size * sc
    ph = 0.2 * size * sc
    pot = [(x - pw_ / 2, yy - ph), (x + pw_ / 2, yy - ph), (x + pw_ * 0.38, yy), (x - pw_ * 0.38, yy)]
    put(s, Q(s, pot), TERRA, 0.75, lift=0.8, wob=0.4, edge=0.5, ew=2)
    s.wash(Q(s, pot), PAYNE, 0.3, wob=0.5, soft=1, grad=s.hgrad(x - pw_ / 2, x + pw_ / 2, 0.1, 1))
    quad(s, r, pot, max(1.0, sc * 0.012), pen, 1.2, over=1, dbl=0, j=0.4)
    lm = s.zeros()
    for _ in range(int(5 + 6 * size)):
        a = U(r, -2.6, -0.5)
        L = U(r, 0.12, 0.38) * size * sc
        cx, cy = x + math.cos(a) * L * 0.6, yy - ph + math.sin(a) * L * 0.7
        lm = np.maximum(lm, s.blob(cx, cy, L * 0.32, L * 0.2, irr=0.3, n=7, rot=a))
    if trail:
        for _ in range(3):
            xa = x + J(r, pw_ * 0.4)
            pts = [(xa, yy - ph * 0.8), (xa + J(r, 6), yy + sc * U(r, 0.2, 0.4)), (xa + J(r, 10), yy + sc * U(r, 0.5, 0.9))]
            for q in np.linspace(0.15, 1, 6):
                px_ = pts[0][0] + (pts[2][0] - pts[0][0]) * q + J(r, 3)
                py_ = pts[0][1] + (pts[2][1] - pts[0][1]) * q
                lm = np.maximum(lm, s.blob(px_, py_, sc * 0.05, sc * 0.035, irr=0.3, n=6))
            s.ink(pts, max(0.8, sc * 0.006), LEAF, 1.0, skip=0.2)
    lm = put(s, lm, MOSS, 0.85, lift=0.7, wob=0.5, edge=0.5, ew=2, flow=0.6, fs=20, gran=0.4)
    s.wash(lm, PRUSSIAN, 0.45, wob=1, soft=1.2, edge=0.2, grad=s.vgrad(yy - ph - sc * 0.4, yy, 0.2, 1), clip=lm)
    s.wash(lm, LEAF, 0.3, wob=1, soft=1.5, edge=0.2, flow=0.9, fs=12, clip=lm)
    ctr(s, lm, max(0.9, sc * 0.009), pen, 1.0, step=3, skip=0.3)


def acunit(s, r, F, kind, sv, y, pen=SEPIA):
    """An air conditioner: a window-type box through the wall, or a split unit on a bracket."""
    if kind == "split":
        a, b, y0, y1, dd = sv, sv + 0.8, y, y + 0.55, 0.32
    elif kind == "under":
        a, b, y0, y1, dd = sv, sv + 0.7, y, y + 0.45, 0.4
    else:
        a, b, y0, y1, dd = sv, sv + 0.62, y, y + 0.42, 0.38
    sc = F.ppm(a, y)
    pen_w = float(np.clip(sc * 0.016, 1.0, 2.2))
    front = F.q(a, y0, b, y1, dd)
    side = F.qs(a, y0, y1, 0, dd)
    bot = F.qh(a, b, y0, 0, dd)
    put(s, Q(s, side), MIST, 0.55, lift=0.85, wob=0.4, edge=0.4, ew=2)
    s.wash(Q(s, side), PAYNE, 0.3, wob=0.5, soft=1)
    put(s, Q(s, bot), SLATE, 0.5, lift=0.85, wob=0.4, edge=0.4, ew=2)
    s.wash(Q(s, bot), PAYNE, 0.45, wob=0.5, soft=1)
    fm = put(s, Q(s, front), (0.84, 0.83, 0.78), 0.55, lift=0.9, wob=0.4, edge=0.45, ew=2)
    s.wash(fm, RAIN, 0.2, wob=1, soft=1.5, edge=0, grad=s.hgrad(front[0][0], front[1][0], 0.2, 1))
    s.wash(fm, SEPIA, 0.12, wob=1, soft=1, edge=0.3, flow=0.9, fs=16)
    if kind == "split":
        cx, cy = F.p(a + 0.28, (y0 + y1) / 2, dd)
        rr = 0.2 * sc
        s.wash(s.ellipse(cx, cy, rr * 0.75, rr), PAYNE, 0.45, wob=0.4, soft=0.8, edge=0.4, ew=2)
        s.ink(ell(cx, cy, rr * 0.75, rr, 0.3, 0.3 + 2 * math.pi + 0.2, 30), pen_w * 0.8, pen, 1.2)
        for q in (0.25, 0.5, 0.75):
            hline(s, r, F, a + 0.55, b - 0.05, y0 + (y1 - y0) * q, dd, pen_w * 0.5, pen, 1.0, over=0)
        for sv2 in (a + 0.1, b - 0.1):          # the bracket
            line(s, r, F.p(sv2, y0, dd), F.p(sv2, y0 - 0.06, 0.0), pen_w * 0.8, INKBLUE, 1.2, over=1)
    else:
        for q in np.linspace(0.15, 0.85, 5):
            hline(s, r, F, a + 0.06, b - 0.06, y0 + (y1 - y0) * q, dd, pen_w * 0.5, pen, 1.0, over=0, skip=0.15)
    quad(s, r, front, pen_w, pen, 1.4, over=2, dbl=0.3, j=0.4)
    line(s, r, side[0], side[1], pen_w * 0.8, pen, 1.2, over=1)
    line(s, r, side[3], side[2], pen_w * 0.8, pen, 1.2, over=1)
    dx, dy = F.p((a + b) / 2 + J(r, 0.2), y0, dd * 0.5)
    s.stroke([(dx, dy), (dx + J(r, 2), dy + sc * 1.2)], max(3, sc * 0.05), PAYNE, 0.22, dry=0.3, edge=0.4, wob=1, taper=(0, 0.7))     # the drip


def cage(s, r, F, fl, pen=INKBLUE):
    """The iron window cage (鐵窗): a box of bars out over the window, its floor, its tin hat, the plants it keeps."""
    w = fl["wins"][0]
    a, b, y0, y1 = fl["a"], fl["b"], w["y0"], w["y1"]
    A_, B_, Y0, Y1, dd = a - 0.12, b + 0.12, y0 - 0.22, y1 + 0.18, fl["dd"]
    sc = F.ppm(a, y0, dd)
    bw = float(np.clip(sc * 0.016, 1.0, 2.2))
    # the floor of the cage, seen from below
    fq = F.qh(A_, B_, Y0, 0, dd)
    put(s, Q(s, fq), SEPIA, 0.4, lift=0.6, wob=0.4, edge=0.4, ew=2)
    s.wash(Q(s, fq), INKBLUE, 0.3, wob=0.5, soft=1)
    for sv in np.arange(A_ + 0.05, B_, 0.2):
        line(s, r, F.p(sv, Y0, 0), F.p(sv, Y0, dd), bw * 0.7, pen, 1.0, over=0)
    # plants on it, behind the front bars
    for i in range(fl.get("plants", 0)):
        plant_pot(s, r, F, U(r, A_ + 0.25, B_ - 0.25), Y0 + 0.02, dd * U(r, 0.35, 0.7), U(r, 0.7, 1.25), trail=fl.get("vine", False) and i == 0)
    # the hat: a sheet of tin or green corrugated plastic sloping out over the top
    if fl.get("roof"):
        hq = [F.p(A_ - 0.05, Y1 + 0.32, 0), F.p(B_ + 0.05, Y1 + 0.32, 0), F.p(B_ + 0.05, Y1 + 0.06, dd + 0.16), F.p(A_ - 0.05, Y1 + 0.06, dd + 0.16)]
        hm = Q(s, hq)
        c = RUST if fl.get("rust") else (0.62, 0.74, 0.66)
        put(s, hm, c, 0.55, lift=0.85, wob=0.5, edge=0.45, ew=2, flow=0.6, fs=30)
        s.wash(hm, PAYNE, 0.3, wob=0.6, soft=1, edge=0, grad=s.vgrad(hq[3][1], hq[0][1], 1, 0.3))
        for sv in np.arange(A_, B_ + 0.05, 0.09):
            line(s, r, F.p(sv, Y1 + 0.32, 0), F.p(sv, Y1 + 0.06, dd + 0.16), bw * 0.5, SEPIA, 0.7, over=0, skip=0.3)
        quad(s, r, hq, bw, SEPIA, 1.3, over=2, dbl=0.3, j=0.4)
        if fl.get("rust"):
            st = [F.p(U(r, A_, B_), Y1 + 0.06, dd + 0.16) for _ in range(4)]
            streaks(s, r, st, sc * 0.5, 3, RUST, 0.25, (2, 5))
    # bars: the near side, then the front
    for dv in np.arange(0.1, dd, 0.11):
        line(s, r, F.p(A_, Y0, dv), F.p(A_, Y1, dv), bw * 0.85, pen, 1.25, over=0.5)
    for yv_ in (Y0, Y1, Y0 + (Y1 - Y0) * 0.35):
        line(s, r, F.p(A_, yv_, 0), F.p(A_, yv_, dd), bw, pen, 1.35, over=1)
    sv = A_
    while sv <= B_ + 1e-3:
        line(s, r, F.p(sv, Y0, dd), F.p(sv, Y1, dd), bw * (1.25 if abs(sv - A_) < 1e-3 or sv > B_ - 0.05 else 0.9), pen, 1.35, over=0.5, skip=0.04)
        sv += 0.11
    for yv_ in (Y0, Y1):
        line(s, r, F.p(A_, yv_, dd), F.p(B_, yv_, dd), bw * 1.3, pen, 1.45, over=2, dbl=0.4)
    ym = Y0 + (Y1 - Y0) * 0.35
    deco = fl.get("deco", 0)
    if deco == 0:          # a band of little loops
        line(s, r, F.p(A_, ym, dd), F.p(B_, ym, dd), bw, pen, 1.3, over=1)
        line(s, r, F.p(A_, ym + 0.12, dd), F.p(B_, ym + 0.12, dd), bw, pen, 1.3, over=1)
        sv = A_ + 0.055
        while sv < B_:
            cx, cy = F.p(sv, ym + 0.06, dd)
            rr = 0.045 * sc
            s.ink(ell(cx, cy, rr * 0.7, rr, 0.3, 0.3 + 2 * math.pi + 0.15, 16), bw * 0.7, pen, 1.2, tremor=0.3)
            sv += 0.11
    elif deco == 1:        # zigzag
        line(s, r, F.p(A_, ym, dd), F.p(B_, ym, dd), bw, pen, 1.3, over=1)
        pts = []
        sv, up = A_, 0
        while sv <= B_ + 1e-3:
            pts.append(F.p(sv, ym + (0.14 if up else 0.0), dd))
            sv += 0.11
            up = 1 - up
        s.ink(pts, bw * 0.8, pen, 1.25, smooth=False, tremor=0.4)
        line(s, r, F.p(A_, ym + 0.14, dd), F.p(B_, ym + 0.14, dd), bw, pen, 1.3, over=1)
    else:
        line(s, r, F.p(A_, ym, dd), F.p(B_, ym, dd), bw, pen, 1.3, over=1)
        line(s, r, F.p(A_, Y1 - 0.25, dd), F.p(B_, Y1 - 0.25, dd), bw * 0.8, pen, 1.2, over=1, skip=0.1)
    # the near uprights, a touch of rust, wet light on the top rail
    line(s, r, F.p(A_, Y0, 0), F.p(A_, Y1, 0), bw, pen, 1.3, over=1)
    if fl.get("rust"):
        rm = s.ribbon([F.p(A_, Y0, dd), F.p(B_, Y0, dd)], bw * 3)
        s.wash(rm, RUST, 0.35, wob=1, soft=1, edge=0.3, dry=0.3)
    hl(s, [F.p(A_ + 0.05, Y1 + 0.01, dd), F.p(B_ - 0.05, Y1 + 0.01, dd)], bw * 1.2, 0.25, dry=0.3)


def hood(s, r, F, fl):
    """A concrete rain hood over a window with no cage."""
    w = fl["wins"][0]
    a, b, y1 = w["a"] - 0.18, w["b"] + 0.18, w["y1"] + 0.14
    dd = 0.38
    sc = F.ppm(a, y1)
    pen = float(np.clip(sc * 0.018, 1.0, 2.4))
    sof = F.qh(a, b, y1, 0, dd)
    fr = F.q(a, y1, b, y1 + 0.09, dd)
    sd = F.qs(a, y1, y1 + 0.09, 0, dd)
    put(s, Q(s, sof), GREY, 0.5, lift=0.85, wob=0.4, edge=0.4, ew=2)
    s.wash(Q(s, sof), PAYNE, 0.4, wob=0.5, soft=1)
    put(s, np.maximum(Q(s, fr), Q(s, sd)), GREY, 0.4, lift=0.85, wob=0.4, edge=0.4, ew=2)
    quad(s, r, fr, pen, SEPIA, 1.4, over=3, dbl=0.3, j=0.4)
    line(s, r, sof[0], sof[3], pen, SEPIA, 1.3, over=1)
    st = [F.p(U(r, a, b), y1, dd) for _ in range(4)]
    streaks(s, r, st, sc * 0.4, 3, PAYNE, 0.2, (2, 5))


def laundry(s, r, F, a, b, y, d, sc):
    """A pole across the balcony and what hangs on it."""
    pen = float(np.clip(sc * 0.014, 0.9, 2.0))
    line(s, r, F.p(a + 0.1, y, d), F.p(b - 0.1, y, d), pen * 1.2, INKBLUE, 1.3, over=1)
    sv = a + U(r, 0.25, 0.5)
    cols = [SOY, RAIN, ROSE, OCHRE, MIST, TEAL, CREAM, SLATE, NAVY]
    while sv < b - 0.5:
        kind = r.random()
        c = cols[int(r.integers(0, len(cols)))]
        if kind < 0.4:      # a shirt on a hanger
            wd, ht = U(r, 0.42, 0.55), U(r, 0.55, 0.7)
            P = [(sv, y - 0.1), (sv + wd * 0.3, y - 0.06), (sv + wd * 0.5, y - 0.1), (sv + wd * 0.7, y - 0.06), (sv + wd, y - 0.1),
                 (sv + wd * 1.08, y - 0.3), (sv + wd * 0.88, y - 0.33), (sv + wd * 0.86, y - ht), (sv + wd * 0.14, y - ht), (sv + wd * 0.12, y - 0.33), (sv - wd * 0.08, y - 0.3)]
            s.ink([F.p(sv + wd * 0.3, y - 0.06, d), F.p(sv + wd * 0.5, y + 0.02, d), F.p(sv + wd * 0.7, y - 0.06, d)], pen * 0.7, SEPIA, 1.0)
        elif kind < 0.7:    # a towel
            wd, ht = U(r, 0.3, 0.5), U(r, 0.5, 0.8)
            P = [(sv, y), (sv + wd, y), (sv + wd + 0.02, y - ht), (sv - 0.01, y - ht * 0.96)]
        else:               # trousers by the hem
            wd, ht = U(r, 0.34, 0.4), U(r, 0.75, 0.95)
            P = [(sv, y), (sv + wd, y), (sv + wd * 0.98, y - ht), (sv + wd * 0.56, y - ht), (sv + wd * 0.5, y - ht * 0.3), (sv + wd * 0.44, y - ht), (sv + wd * 0.02, y - ht)]
        pts = [F.p(p[0], p[1], d) for p in P]
        m = put(s, Q(s, pts), c, U(r, 0.45, 0.75), lift=0.85, wob=0.6, edge=0.5, ew=2, flow=0.5, fs=20)
        s.wash(m, PAYNE, 0.25, wob=1, soft=1.5, edge=0, grad=s.vgrad(pts[0][1], pts[0][1] + sc * 0.8, 0.2, 0.8), clip=m)
        ctr(s, m, pen * 0.8, SEPIA, 1.1, step=3, skip=0.15)
        sv += wd + U(r, 0.08, 0.3)


def balcony(s, r, F, fl, col, pen_c=INKBLUE):
    a, b, yf, yt, dd = fl["a"], fl["b"], fl["yf"], fl["yt"], fl["dd"]
    sc = F.ppm(a, yf, dd)
    bw = float(np.clip(sc * 0.016, 1.0, 2.2))
    pen = float(np.clip(sc * 0.02, 1.0, 2.6))
    # its roof: the slab above, seen from below
    sof = F.qh(a - 0.08, b + 0.08, yt - 0.16, 0, dd + 0.05)
    put(s, Q(s, sof), GREY, 0.45, lift=0.85, wob=0.5, edge=0.4, ew=3)
    s.wash(Q(s, sof), PAYNE, 0.42, wob=0.8, soft=1.5, grad=s.vgrad(sof[3][1], sof[0][1], 0.5, 1))
    if fl.get("laundry"):
        laundry(s, r, F, a, b, yt - 0.42, dd * 0.55, sc)
    # the cage over the parapet
    Y0, Y1 = yf + 1.0, yt - 0.16
    for dv in np.arange(0.12, dd, 0.12):
        line(s, r, F.p(a, Y0, dv), F.p(a, Y1, dv), bw * 0.85, pen_c, 1.25, over=0.5)
    sv = a
    while sv <= b + 1e-3:
        line(s, r, F.p(sv, Y0, dd), F.p(sv, Y1, dd), bw * 0.9, pen_c, 1.35, over=0.5, skip=0.04)
        sv += 0.12
    for yv_ in (Y1, Y0 + 0.9):
        line(s, r, F.p(a, yv_, dd), F.p(b, yv_, dd), bw * 1.2, pen_c, 1.4, over=2, dbl=0.3)
        line(s, r, F.p(a, yv_, 0), F.p(a, yv_, dd), bw, pen_c, 1.3, over=1)
    # the parapet wall (tiled) and the slab under it
    pf = F.q(a, yf, b, Y0, dd)
    ps = F.qs(a, yf, Y0, 0, dd)
    pm = put(s, Q(s, pf), col, 0.66, lift=0.95, wob=0.5, edge=0.4, ew=3, flow=0.5, fs=60, gran=0.35)
    s.wash(pm, DUSK, 0.34, wob=1, soft=2, edge=0.1, grad=s.vgrad(pf[3][1], pf[0][1], 0.4, 1.0))
    s.wash(pm, col, 0.2, wob=3, bleed=8, edge=0.2, flow=0.8, fs=40, gran=0.4, clip=pm)
    s.wash(pm, PAYNE, 0.12, wob=2, bleed=6, edge=0.2, flow=0.9, fs=25, clip=pm)
    sm = put(s, Q(s, ps), col, 0.6, lift=0.95, wob=0.5, edge=0.4, ew=3)
    s.wash(sm, PAYNE, 0.32, wob=1, soft=1.5)
    Fo = F.off(dd)
    sv_, yv_ = Fo.sy(s)
    tilework(s, r, Fo, pm, sv_, yv_, 0.11, k=0.28, seed=int(a * 100) + 7)
    st = [F.p(U(r, a, b), yf, dd) for _ in range(6)]
    streaks(s, r, st, sc * 0.9, 4, PAYNE, 0.18, (sc * 0.03, sc * 0.09))
    bf = F.q(a - 0.08, yf - 0.18, b + 0.08, yf, dd + 0.05)
    bsof = F.qh(a - 0.08, b + 0.08, yf - 0.18, 0, dd + 0.05)
    put(s, Q(s, bsof), GREY, 0.45, lift=0.9, wob=0.5, edge=0.4, ew=3)
    s.wash(Q(s, bsof), PAYNE, 0.5, wob=0.8, soft=1.5)
    put(s, Q(s, bf), GREY, 0.42, lift=0.9, wob=0.5, edge=0.4, ew=2)
    s.wash(Q(s, bf), SLATE, 0.22, wob=1, soft=1.5, edge=0.2)
    quad(s, r, pf, pen, SEPIA, 1.5, over=4, dbl=0.4, j=0.5)
    line(s, r, ps[0], ps[1], pen, SEPIA, 1.3, over=2)
    line(s, r, ps[3], ps[2], pen, SEPIA, 1.3, over=2)
    quad(s, r, bf, pen, SEPIA, 1.4, over=4, dbl=0.3, j=0.5)
    line(s, r, bsof[0], bsof[3], pen * 0.8, SEPIA, 1.2, over=1)
    hl(s, [F.p(a + 0.05, Y0 - 0.02, dd), F.p(b - 0.05, Y0 - 0.02, dd)], bw * 1.5, 0.3, dry=0.2)
    for _ in range(fl.get("plants", 0)):
        plant_pot(s, r, F, U(r, a + 0.3, b - 0.3), Y0, dd * 0.6, U(r, 0.8, 1.3))


def canopy(s, r, F, a, b, y=2.95, dd=0.95):
    """A little tin canopy over a ground-floor door or shutter; a tube light under it. Returns the light's sheet point."""
    sc = F.ppm(a, y, dd)
    pen = float(np.clip(sc * 0.018, 1.0, 2.4))
    hq = [F.p(a, y + 0.25, 0), F.p(b, y + 0.25, 0), F.p(b, y, dd), F.p(a, y, dd)]
    hm = Q(s, hq)
    put(s, hm, RUST, 0.5, lift=0.85, wob=0.5, edge=0.4, ew=3, flow=0.6, fs=30)
    s.wash(hm, PAYNE, 0.45, wob=0.6, soft=1.5, edge=0)
    for sv in np.arange(a, b, 0.1):
        line(s, r, F.p(sv, y + 0.25, 0), F.p(sv, y, dd), pen * 0.5, SEPIA, 0.7, over=0, skip=0.3)
    quad(s, r, hq, pen, SEPIA, 1.4, over=3, dbl=0.3, j=0.5)
    lp = F.p((a + b) / 2, y + 0.08, 0.15)
    tube = s.ribbon([F.p((a + b) / 2 - 0.3, y + 0.1, 0.1), F.p((a + b) / 2 + 0.3, y + 0.1, 0.1)], max(2.0, sc * 0.04))
    s.lift(tube, 1.0, soft=0.5, wob=0)
    s.wash(tube, CREAM, 0.15, wob=0, soft=0.5, body=1)
    st = [F.p(U(r, a, b), y, dd) for _ in range(4)]
    streaks(s, r, st, sc * 0.6, 3, RUST, 0.2, (2, 5))
    return [round(lp[0]), round(lp[1])]


def downpipe(s, r, F, sv, y0, y1, d=0.1):
    sc = F.ppm(sv, (y0 + y1) / 2)
    w = 0.1 * sc
    a, b = F.p(sv, y0, d), F.p(sv, y1, d)
    m = s.ribbon([a, b], w, smooth=False)
    put(s, m, GREY, 0.45, lift=0.8, wob=0.4, edge=0.4, ew=2)
    s.wash(m, PAYNE, 0.3, wob=0.5, soft=1, grad=s.hgrad(a[0] - w / 2, a[0] + w / 2, 0.1, 1))
    pen = float(np.clip(sc * 0.016, 1.0, 2.2))
    line(s, r, (a[0] - w / 2, a[1]), (b[0] - w / 2, b[1]), pen * 0.7, SEPIA, 1.1, over=2)
    line(s, r, (a[0] + w / 2, a[1]), (b[0] + w / 2, b[1]), pen, SEPIA, 1.3, over=2)
    y = y0 + U(r, 0.5, 1.5)
    while y < y1:
        p = F.p(sv, y, d)
        line(s, r, (p[0] - w * 0.7, p[1]), (p[0] + w * 0.7, p[1]), pen, SEPIA, 1.3, over=0)
        y += U(r, 1.8, 2.6)


def p2_unit(S, F, FL, u, lamps):
    r = rs(u["seed"] + 5)
    s0, s1 = u["s0"], u["s1"]
    top = FL[-1]
    ext = F.q(s0 - 0.4, -0.8, s1 + 0.4, top + 2) + F.q(s0 - 0.4, -0.8, s1 + 0.4, top + 2, 1.3)
    s = sub_of(S, ext, 40)
    if u["i"] % 2 == 0:
        downpipe(s, r, F, s0 + 0.12, 0.2, top + 0.9)
    for fl in u["floors"]:
        if fl["kind"] == "bal":
            balcony(s, r, F, fl, u["tile"])
        else:
            if fl.get("ac") and fl["ac"][0] == "under":
                acunit(s, r, F, "under", fl["ac"][1], fl["ac"][2])
            if fl["cage"]:
                cage(s, r, F, fl)
            else:
                hood(s, r, F, fl)
            if fl.get("ac") and fl["ac"][0] != "under":
                acunit(s, r, F, fl["ac"][0], fl["ac"][1], fl["ac"][2])
        for w in fl["wins"]:
            if w["bath"]:       # louvred glass and a little grille, flush
                pen = pw(F, w["a"], w["y0"], 0, 0.014)
                for q in np.linspace(0.15, 0.85, 5):
                    hline(s, r, F, w["a"] + 0.04, w["b"] - 0.04, w["y0"] + (w["y1"] - w["y0"]) * q, -0.06, pen * 0.7, SEPIA, 1.0, over=0)
                for q in (0.33, 0.66):
                    vline(s, r, F, w["a"] + (w["b"] - w["a"]) * q, w["y0"], w["y1"], 0.05, pen, INKBLUE, 1.3, over=1)
    if u["ground"] == "door":
        lamps.append(canopy(s, r, F, s0 + 0.25, s0 + 1.7))
    elif r.random() < 0.6:
        lamps.append(canopy(s, r, F, s0 + 0.4, s1 - 0.4, 3.05, 0.8))


# ---------------------------------------------------------------------------------------------- the roofs
def tank(S, r, F, sv, d, y, rad=0.62, h=1.35, stand=1.5, c=MIST, steel=True):
    """A water tank on a steel stand, standing on the roof at (sv, d) behind the parapet."""
    s = sub_of(S, [F.p(sv - 1.2, y, d), F.p(sv + 1.2, y + stand + h + 0.6, d)], 60)
    sc = F.ppm(sv, y + stand, d)
    pen = float(np.clip(sc * 0.02, 1.0, 2.6))
    for ds in (-rad * 0.8, rad * 0.8):
        line(s, r, F.p(sv + ds, y, d), F.p(sv + ds * 0.9, y + stand, d), pen * 1.2, INKBLUE, 1.4, over=1)
    line(s, r, F.p(sv - rad * 0.8, y + 0.2, d), F.p(sv + rad * 0.75, y + stand - 0.1, d), pen * 0.8, INKBLUE, 1.2, over=0)
    line(s, r, F.p(sv + rad * 0.8, y + 0.2, d), F.p(sv - rad * 0.75, y + stand - 0.1, d), pen * 0.8, INKBLUE, 1.2, over=0)
    line(s, r, F.p(sv - rad, y + stand, d), F.p(sv + rad, y + stand, d), pen * 1.3, INKBLUE, 1.4, over=3)
    xl, yb = F.p(sv - rad, y + stand + 0.05, d)
    xr, _ = F.p(sv + rad, y + stand + 0.05, d)
    _, yt = F.p(sv, y + stand + h, d)
    cx, wr = (xl + xr) / 2, (xr - xl) / 2
    eh = wr * 0.18
    body = np.maximum(s.rect(xl, yt, xr, yb), np.maximum(s.ellipse(cx, yb, wr, eh), s.ellipse(cx, yt, wr, eh)))
    bm = put(s, body, c, 0.5, lift=0.95, wob=0.5, edge=0.45, ew=3, flow=0.5, fs=30)
    s.wash(bm, SLATE if steel else SIENNA, 0.4, wob=1, soft=2, edge=0, grad=s.hgrad(xl, xr, 0.1, 1), clip=bm)
    if steel:
        for q, a in ((0.22, 0.5), (0.3, 0.3), (0.62, 0.2)):
            hl(s, [(xl + (xr - xl) * q, yt + eh), (xl + (xr - xl) * q, yb + eh * 0.6)], max(2, wr * 0.08), a)
        for q in (0.3, 0.6):
            yy = yt + (yb - yt) * q
            s.ink(ell(cx, yy, wr, eh, 0.05, math.pi - 0.05, 24), pen * 0.6, SEPIA, 0.9, skip=0.2)
    else:
        for q in np.linspace(0.15, 0.85, 4):
            yy = yt + (yb - yt) * q
            s.ink(ell(cx, yy, wr, eh, 0.05, math.pi - 0.05, 24), pen * 0.8, SEPIA, 1.0, skip=0.1)
    st = [(U(r, xl, xr), yb) for _ in range(4)]
    streaks(s, r, st, sc * 0.7, 3, RUST, 0.25, (2, 6))
    s.ink([(xl, yt), (xl, yb)], pen, SEPIA, 1.4)
    s.ink([(xr, yt), (xr, yb)], pen * 1.2, SEPIA, 1.5)
    s.ink(ell(cx, yb, wr, eh, 0.0, math.pi, 30), pen, SEPIA, 1.4)
    s.ink(ell(cx, yt, wr, eh, -0.1, math.pi + 0.1, 30), pen, SEPIA, 1.3)
    s.ink(ell(cx, yt, wr, eh, math.pi + 0.3, 2 * math.pi - 0.3, 20), pen * 0.7, SEPIA, 0.9, skip=0.3)
    put(s, s.ellipse(cx, yt - eh * 0.7, wr * 0.25, eh * 0.5), c, 0.5, lift=0.6, wob=0.3)
    s.ink(ell(cx, yt - eh * 0.7, wr * 0.25, eh * 0.5, 0, 2 * math.pi, 16), pen * 0.7, SEPIA, 1.2)
    # a pipe down to the roof
    s.ink([F.p(sv + rad * 0.5, y + stand + 0.1, d), F.p(sv + rad * 0.55, y + 0.3, d), F.p(sv + rad + 0.6, y + 0.2, d)], pen * 1.2, SEPIA, 1.3)


def aerial(S, r, F, sv, d, y, h=2.2, bars=3):
    s = sub_of(S, [F.p(sv - 0.8, y, d), F.p(sv + 0.8, y + h + 0.3, d)], 40)
    sc = F.ppm(sv, y + h, d)
    pen = float(np.clip(sc * 0.014, 1.0, 2.0))
    a, t = F.p(sv, y, d), F.p(sv + J(r, 0.05), y + h, d)
    s.ink([a, ((a[0] + t[0]) / 2 + J(r, 1), (a[1] + t[1]) / 2), t], pen * 1.2, SEPIA, 1.3, taper=(0.02, 0.1))
    for i in range(bars):
        yy = y + h - 0.12 - i * U(r, 0.2, 0.3)
        hw = U(r, 0.3, 0.55) * (1 - i * 0.15)
        line(s, r, F.p(sv - hw, yy + J(r, 0.03), d - hw * 0.3), F.p(sv + hw, yy + J(r, 0.03), d + hw * 0.3), pen, SEPIA, 1.2, over=1)


def roof_rail(S, r, F, s0, s1, y, pen_c=INKBLUE):
    s = sub_of(S, [F.p(s0, y, 0), F.p(s1, y + 1.1, 0.1)], 30)
    sc = F.ppm(s0, y)
    bw = float(np.clip(sc * 0.014, 1.0, 2.0))
    sv = s0 + 0.05
    while sv < s1:
        line(s, r, F.p(sv, y, 0.03), F.p(sv, y + 0.95, 0.03), bw, pen_c, 1.3, over=0.5, skip=0.05)
        sv += U(r, 0.9, 1.2)
    line(s, r, F.p(s0, y + 0.95, 0.03), F.p(s1, y + 0.95, 0.03), bw * 1.3, pen_c, 1.4, over=2, dbl=0.4)
    line(s, r, F.p(s0, y + 0.5, 0.03), F.p(s1, y + 0.5, 0.03), bw * 0.8, pen_c, 1.1, over=1, skip=0.15)


def tin_room(S, r, F, s0, s1, y, h, recs, wins_out):
    """頂樓加蓋: a rusty corrugated-iron room on the roof, set back a little, with a window and an overhanging roof."""
    Fo = F.off(-0.55)
    top = y + h
    ext = Fo.q(s0 - 0.3, y, s1 + 0.3, top + 0.4) + Fo.q(s0 - 0.3, y, s1 + 0.3, top + 0.4, 0.8)
    s = sub_of(S, ext, 40)
    sc = Fo.ppm(s0, top)
    pen = float(np.clip(sc * 0.02, 1.0, 2.6))
    # its near side wall
    sd = Q(s, Fo.qs(s0, y, top, 0, -2.4))
    put(s, sd, RUST, 0.55, lift=1.0, wob=0.8, edge=0.4, ew=4, flow=0.6, fs=40, gran=0.5)
    s.wash(sd, PAYNE, 0.4, wob=1, soft=2)
    for dv in np.arange(-0.1, -2.4, -0.1):
        line(s, r, Fo.p(s0, y, dv), Fo.p(s0, top, dv), pen * 0.5, SEPIA, 0.8, over=0, skip=0.3)
    # the front wall
    fm = Q(s, Fo.q(s0, y, s1, top))
    fm = put(s, fm, SIENNA, 0.6, lift=1.0, wob=0.8, edge=0.4, ew=5, flow=0.6, fs=60, gran=0.5)
    s.wash(fm, RUST, 0.45, wob=3, bleed=10, edge=0.2, flow=0.8, fs=50, gran=0.5, clip=fm)
    s.wash(fm, UMBER, 0.25, wob=3, bleed=12, edge=0.2, flow=0.9, fs=40, grad=s.vgrad(Fo.p(s0, top)[1], Fo.p(s0, y)[1], 0.2, 1), clip=fm)
    mottle(s, r, *Fo.p(s0, top), *Fo.p(s1, y), [BURNT, PAYNE, SLATE, OCHRE], 7, 0.18, size=(30, 90), clip=fm)
    sv = s0 + 0.04
    while sv < s1:
        k = 0.5 + 0.4 * r.random()
        line(s, r, Fo.p(sv, y, 0), Fo.p(sv, top, 0), pen * 0.55, SEPIA, k, over=0, skip=0.2)
        if r.random() < 0.3:
            hl(s, [Fo.p(sv + 0.03, y + 0.1, 0), Fo.p(sv + 0.03, top - 0.1, 0)], pen * 1.2, 0.18, dry=0.4)
        sv += 0.075
    st = [Fo.p(U(r, s0, s1), top - 0.1) for _ in range(10)]
    streaks(s, r, st, sc * 1.6, 9, BURNT, 0.3, (3, 9), clip=fm)
    # a window in it (glass: a hole, like the rest)
    w = tin_window()
    p1_window(S, s, r, Fo, w, SIENNA)
    w["idx"], w["floor"], w["unit"], w["fac"] = len(wins_out), 5, -1, "tin"
    wins_out.append(w)
    # a door at the far end
    da = s1 - 1.6
    dm = Q(s, Fo.q(da, y, da + 0.9, y + 2.0, -0.04))
    put(s, dm, SLATE, 0.6, lift=0.9, wob=0.6, edge=0.45, ew=3)
    s.wash(dm, PAYNE, 0.3, wob=1, soft=2)
    quad(s, r, Fo.q(da, y, da + 0.9, y + 2.0, -0.04), pen, SEPIA, 1.4, over=3, dbl=0.3)
    # the overhanging roof sheet
    rq = [Fo.p(s0 - 0.25, top + 0.05, 0), Fo.p(s1 + 0.25, top + 0.05, 0), Fo.p(s1 + 0.25, top - 0.12, 0.45), Fo.p(s0 - 0.25, top - 0.12, 0.45)]
    rm = put(s, Q(s, rq), RUST, 0.6, lift=0.95, wob=0.6, edge=0.45, ew=3)
    s.wash(rm, PAYNE, 0.55, wob=0.8, soft=1.5)
    for sv in np.arange(s0 - 0.25, s1 + 0.25, 0.12):
        line(s, r, Fo.p(sv, top + 0.05, 0), Fo.p(sv, top - 0.12, 0.45), pen * 0.5, SEPIA, 0.7, over=0, skip=0.3)
    edge = Q(s, Fo.q(s0 - 0.25, top - 0.12, s1 + 0.25, top - 0.02, 0.45))
    put(s, edge, RUST, 0.7, lift=0.9, wob=0.5, edge=0.5, ew=2)
    quad(s, r, rq, pen, SEPIA, 1.5, over=5, dbl=0.4)
    hline(s, r, Fo, s0 - 0.25, s1 + 0.25, top - 0.02, 0.45, pen, SEPIA, 1.4, over=4)
    quad(s, r, Fo.q(s0, y, s1, top), pen, SEPIA, 1.4, over=5, dbl=0.4)
    line(s, r, Fo.p(s0, top, -2.4), Fo.p(s0, top, 0), pen, SEPIA, 1.3, over=2)
    return s


def block(side):
    S = Sheet(W, H, 70 if side == "L" else 71)
    S.hole = np.zeros((H, W), np.float32)
    F, FL, units, wins = plan(side)
    r = rs(72 if side == "L" else 73)
    top = FL[-1]
    lamps = []
    extra = []
    # things on the roofs (behind the parapets: painted first, the walls go over them)
    if side == "L":
        tank(S, r, F, 19.6, -1.3, top, 0.66, 1.4, 1.6)
        tank(S, r, F, 25.4, -1.0, top, 0.5, 1.1, 0.9, c=SOY, steel=False)
        aerial(S, r, F, 13.5, -0.8, top + 0.8, 2.4, 3)
        aerial(S, r, F, 23.3, -1.5, top + 0.8, 1.9, 2)
    else:
        tank(S, r, F, 21.1, -1.2, top, 0.62, 1.3, 1.4)
        aerial(S, r, F, 26.0, -1.0, top + 0.8, 2.2, 3)
    if side == "R":
        tin_room(S, r, F, TIN[0], TIN[1], top, TIN[2], None, extra)
        aerial(S, r, F, 10.5, -1.4, top + 2.95, 1.8, 3)
    for i, u in enumerate(units):
        p1_unit(S, F, FL, u, i == len(units) - 1)
    cut(S, S.hole)
    for u in units[::-1]:
        p2_unit(S, F, FL, u, lamps)
    if side == "L":
        roof_rail(S, r, F, 16.9, 27.4, top + 1.08)
    else:
        roof_rail(S, r, F, 0.0, 7.5, top + 1.08)
        roof_rail(S, r, F, 18.2, 29.5, top + 1.08)
    # cable bundles along the wall, drooping between their hooks
    s = Sub(S, 0, 0, W, H)
    for yb, nn in ((FL[1] + 0.35, 4), (FL[2] + 0.25, 2)):
        hooks = np.arange(0.0, F.len + 0.1, 4.2)
        for j in range(nn):
            for h0, h1 in zip(hooks, hooks[1:]):
                a, b = F.p(h0, yb - j * 0.07, 0.15), F.p(h1, yb - j * 0.07, 0.15)
                sag = F.ppm((h0 + h1) / 2, yb) * U(r, 0.08, 0.2) * (1 + j * 0.3)
                cable(s, r, a, b, sag, pw(F, h0, yb, 0, 0.012, 0.9, 1.8), INKBLUE, U(r, 1.0, 1.3))
    allw = wins + extra
    q = [[[round(v, 1) for v in p] for p in glass_quad(F.off(-0.55) if w.get("fac") == "tin" else F, w)] for w in allw]
    rects = [[int(min(p[0] for p in qq)), int(min(p[1] for p in qq)), int(math.ceil(max(p[0] for p in qq))), int(math.ceil(max(p[1] for p in qq)))] for qq in q]
    mi = [w["idx"] for w in wins if w["unit"] == 2 and w["floor"] == 3][0]
    roofline = [[round(v) for v in F.p(sv, top + 1.08, 0.06)] for sv in np.linspace(0, F.len, 6)]
    meta = {"windows": rects, "quads": q, "floors": [w["floor"] for w in allw], "lamps": lamps, "roof": roofline}
    meta["mine" if side == "L" else "yours"] = mi
    S.save("bk_left" if side == "L" else "bk_right", crop=True, anchor=(0, 0), q=90, meta=meta)
    return S


def left():
    block("L")


def right():
    block("R")


# ---------------------------------------------------------------------------------------------- the lane beyond, the floor
def far():
    """The lane running on past the two blocks to a cross street, and the wet lane floor under everything."""
    S = Sheet(W, H, 74)
    r = rs(74)
    hz = CAM((0, 1.5, 1e5))[1]
    # the lane floor
    s = Sub(S, 0, 1380, W, H)
    fl = s.rect(-10, 1440, W + 10, H + 10)
    put(s, fl, SLATE, 0.75, lift=0, wob=1, edge=0.3, ew=6, flow=0.6, fs=200, gran=0.45)
    s.wash(fl, PAYNE, 0.5, wob=0, soft=0, edge=0, grad=s.vgrad(1460, H, 0.3, 1), flow=0.6, fs=160, gran=0.4)
    s.wash(fl, SEPIA, 0.18, wob=3, bleed=14, edge=0.1, flow=0.9, fs=90, gran=0.4)
    # puddles: lifted, flattened by the distance, holding the sky
    for cx, cy, rx in ((1500, 1530, 70), (1330, 1610, 140), (1760, 1660, 170), (1120, 1700, 190), (2060, 1580, 90), (1580, 1490, 40),
                       (900, 1650, 110), (2330, 1700, 160)):
        ry = (cy - hz) * 0.12
        pm = s.blob(cx, cy, rx, ry, irr=0.4, n=12)
        s.lift(pm, 0.45, soft=2.0, wob=3)
        s.wash(pm, PAYNE, 0.12, wob=2, soft=1.5, edge=0.7, ew=3)
        s.ink(ell(cx, cy, rx * 0.98, ry * 0.95, 3.4, 6.0, 20), 1.1, SEPIA, 0.5, skip=0.45)
    sx, sy = CAM((-0.4, 0, 92))
    refl = s.ribbon([(sx, sy + 4), (sx + 6, sy + 90), (sx - 4, H + 10)], 120, profile=lambda q: 1 - 0.5 * q)
    s.wash(refl, None, 0.45, wob=3, soft=6, edge=0, gran=0, flow=0.5, fs=30, lift=True, dry=0.25)
    s.wash(refl, OCHRE, 0.25, wob=3, soft=6, edge=0, dry=0.25)
    # the red no-parking lines along the walls, a drain, a manhole
    for F in (LF, RF):
        for dd in (0.45, 0.6):
            a, b = F.p(-1.0, 0, dd), F.p(F.len + 25, 0, dd + (0 if F is LF else 0))
            s.ink([a, b], 3.0 if dd < 0.5 else 1.6, (0.78, 0.30, 0.26), 0.9, skip=0.35, dry=0.4, tremor=1.2)
    cx, cy = 1560, 1610
    s.wash(s.ellipse(cx, cy, 70, 12), PAYNE, 0.6, wob=0.6, soft=0.8, edge=0.5, ew=2)
    s.ink(ell(cx, cy, 70, 12, 0.2, 6.6, 30), 1.6, SEPIA, 1.2)
    for q in np.linspace(-0.7, 0.7, 6):
        s.ink([(cx + q * 70, cy - 9 * math.sqrt(1 - q * q)), (cx + q * 70, cy + 9 * math.sqrt(1 - q * q))], 1.0, SEPIA, 0.9)
    s.wash(s.full(), None, 0.2, wob=0, soft=0, edge=0, gran=0, flow=0.7, fs=160, grad=s.vgrad(1440, 1520, 1, 0), lift=True)
    # the lane going on: slivers of wall on both sides, and the building across the end
    t = Sub(S, 1300, 600, 1800, 1500)
    def strip(xa, za, zb, n_st, seed):
        g = rs(seed)
        P = lambda z, y, x=xa: CAM((x, y, z))
        top = n_st * 3.1 + 0.6
        q = [P(za, 0), P(zb, 0), P(zb, top), P(za, top)]
        m = put(t, Q(t, q), SLATE, 0.75, lift=0, wob=0.6, edge=0.4, ew=3, flow=0.6, fs=60, gran=0.4)
        t.wash(m, PAYNE, 0.35, wob=1, soft=2, edge=0, flow=0.7, fs=30)
        quad(t, g, q, 1.2, SEPIA, 1.2, over=3, dbl=0.2)
        for k in range(1, n_st):
            line(t, g, P(za, k * 3.1 + 0.4), P(zb, k * 3.1 + 0.4), 1.0, SEPIA, 0.9, over=2, skip=0.2)
        for z in np.arange(za + 2, zb - 1, U(g, 3.5, 5)):
            for k in range(1, n_st):
                wq = [P(z, k * 3.1 + 0.9), P(z + 1.6, k * 3.1 + 0.9), P(z + 1.6, k * 3.1 + 2.3), P(z, k * 3.1 + 2.3)]
                t.wash(Q(t, wq), INKBLUE, 0.6, wob=0.4, soft=0.6, edge=0.4, ew=1)
                if g.random() < 0.6:
                    quad(t, g, [CAM((xa + (0.4 if xa < 0 else -0.4), y_, z_)) for (z_, y_) in ((z - 0.1, k * 3.1 + 0.7), (z + 1.7, k * 3.1 + 0.7), (z + 1.7, k * 3.1 + 2.5), (z - 0.1, k * 3.1 + 2.5))], 1.0, INKBLUE, 1.1, over=1, dbl=0, j=0.3)
        return top
    strip(-1.7, 52, 92, 5, 741)
    strip(1.9, 53, 92, 4, 742)
    # vertical shop signs sticking out into the lane, one after another
    for z, x, hgt, c in ((60, -1.7, 4.5, (0.80, 0.34, 0.30)), (71, -1.7, 3.8, (0.94, 0.86, 0.60)), (66, 1.9, 4.2, (0.54, 0.70, 0.78))):
        sx = 0.9 if x < 0 else -0.9
        q = [CAM((x, hgt, z)), CAM((x + sx, hgt, z)), CAM((x + sx, hgt + 2.4, z)), CAM((x, hgt + 2.4, z))]
        put(t, Q(t, q), c, 0.65, lift=0.9, wob=0.4, edge=0.5, ew=2)
        quad(t, r, q, 1.1, SEPIA, 1.2, over=1, dbl=0, j=0.3)
    # the building across the end of the lane, a lit shop at its foot
    zc = 92
    P = lambda x, y: CAM((x, y, zc))
    q = [P(-6, 0), P(6, 0), P(6, 13.2), P(-6, 13.2)]
    m = put(t, Q(t, q), MIST, 0.6, lift=0, wob=0.6, edge=0.4, ew=3, flow=0.5, fs=60, gran=0.4)
    t.wash(m, SLATE, 0.7, wob=1, soft=2, edge=0)
    t.wash(m, DUSK, 0.4, wob=2, bleed=8, edge=0.1)
    t.wash(m, PAYNE, 0.3, wob=2, bleed=10, edge=0.1, grad=t.vgrad(q[3][1], q[0][1], 1, 0.2))
    quad(t, r, q, 1.3, SEPIA, 1.3, over=3, dbl=0.3)
    for k in range(1, 4):
        line(t, r, P(-6, k * 3.2 + 0.3), P(6, k * 3.2 + 0.3), 1.0, SEPIA, 1.0, over=2, skip=0.15)
        for x in (-4.8, -2.3, 0.4, 2.9):
            wq = [P(x, k * 3.2 + 1.0), P(x + 1.6, k * 3.2 + 1.0), P(x + 1.6, k * 3.2 + 2.3), P(x, k * 3.2 + 2.3)]
            t.wash(Q(t, wq), INKBLUE, U(r, 0.5, 0.9), wob=0.8, soft=0.8, edge=0.4, ew=1)
            if r.random() < 0.6:
                box(t, r, wq[0][0] - 1, wq[2][1] - 1, wq[1][0] + 1, wq[0][1] + 1, 1.0, INKBLUE, 1.0, over=1, dbl=0, j=0.5, skip=0.2)
    shop = [P(-3.2, 0), P(2.4, 0), P(2.4, 2.8), P(-3.2, 2.8)]
    t.lift(Q(t, shop), 1.0, soft=0.6, wob=0.4)
    t.wash(Q(t, shop), CREAM, 0.3, wob=0.4, soft=0.8, edge=0.4, ew=2)
    sign = [P(-3.4, 2.8), P(2.6, 2.8), P(2.6, 3.4), P(-3.4, 3.4)]
    put(t, Q(t, sign), (0.92, 0.62, 0.30), 0.6, lift=0.9, wob=0.3)
    put(t, Q(t, [P(-0.6, 2.85), P(0.5, 2.85), P(0.5, 3.35), P(-0.6, 3.35)]), (0.40, 0.66, 0.52), 0.7, lift=0.6, wob=0.2)
    quad(t, r, shop, 1.1, SEPIA, 1.1, over=2, dbl=0.2)
    for x in (-1.6, 0.8):
        line(t, r, P(x, 0), P(x, 2.8), 1.0, SEPIA, 1.0, over=0)
    line(t, r, P(-6, 3.0), P(-3.4, 3.0), 1.0, SEPIA, 0.9)
    # a figure-less doorway glow, wires across the far end
    for k in range(4):
        cable(t, r, CAM((-1.7, 6.5 + k * 0.4, 58 + k * 6)), CAM((1.9, 6.2 + k * 0.5, 60 + k * 6)), 4 + k * 2, 1.0, INKBLUE, 1.0)
    S.save("bk_far", crop=True, anchor=(0, 0), q=90, meta={"shop_glow": [round(v) for v in P(-0.4, 1.4)], "horizon": round(hz), "p": 0.9})


# ---------------------------------------------------------------------------------------------- the glass, dark and lit
def all_windows():
    """Every window of both blocks with its facade and its screen quad (glass plane)."""
    out = []
    for side in ("L", "R"):
        F, FL, units, wins = plan(side)
        for w in wins:
            out.append((side, F, w, glass_quad(F, w)))
    w = tin_window()
    Fo = RF.off(-0.55)
    out.append(("R", Fo, w, glass_quad(Fo, w)))
    return out


def tin_window():
    s0, s1, y = TIN[0], TIN[1], FL_R[-1]
    wa = s0 + (s1 - s0) * 0.32
    return dict(a=wa, b=wa + 1.6, y0=y + 1.35, y1=y + 2.3, frame=MIST, tr=False, door=False, bath=False, floor=5, unit=-1)


def glass(lit=False):
    S = Sheet(W, H, 75 + lit)
    for i, (side, F, w, q) in enumerate(all_windows()):
        r = rs(800 + i)
        P = np.asarray(q, float)
        c = P.mean(0)
        P = c + (P - c) * 1.06                                     # a little larger than the hole
        x0, y0 = P.min(0)
        x1, y1 = P.max(0)
        s = Sub(S, x0 - 12, y0 - 12, x1 + 12, y1 + 12)
        m = Q(s, P)
        gw, gh = x1 - x0, y1 - y0
        bl = P[0] if side == "L" else P[1]                         # the corner nearest the camera, low
        if not lit:
            m = s.wash(m, INKBLUE, U(r, 1.0, 1.3), wob=0, soft=0.6, edge=0.4, ew=3, flow=0.5, fs=40, gran=0.3, body=1)
            s.wash(m, PAYNE, 0.45, wob=1, soft=2, edge=0, grad=s.vgrad(y0, y1, 1, 0.3), flow=0.7, fs=30, clip=m)
            if r.random() < 0.6:       # a curtain, just paler than the dark
                cw = gw * U(r, 0.25, 0.5)
                xa = x0 if r.random() < 0.5 else x1 - cw
                s.wash(s.rect(xa, y0, xa + cw, y1) * m, SLATE, 0.3, wob=1, soft=1.5, edge=0.4, ew=2)
                for qq in np.linspace(0.2, 0.8, 3):
                    xx = xa + cw * qq
                    s.ink([(xx, y0 + 2), (xx + J(r, 2), (y0 + y1) / 2), (xx + J(r, 2), y1 - 2)], 1.0, INKBLUE, 0.6, skip=0.3, clip=m)
            # the glass catches a little of the sky
            a = U(r, 0.2, 0.6)
            band = Q(s, [(x0 + gw * a, y0), (x0 + gw * (a + 0.18), y0), (x0 + gw * (a - 0.15), y1), (x0 + gw * (a - 0.33), y1)])
            s.wash(band * m, None, U(r, 0.2, 0.35), wob=1, soft=3, edge=0, gran=0, flow=0, lift=True)
            s.wash(band * m, RAIN, 0.12, wob=1, soft=3, edge=0)
        else:
            m = s.wash(m, CREAM, 0.35, wob=0, soft=0.6, edge=0.4, ew=3, flow=0.5, fs=40, gran=0.3, body=1)
            lx, ly = x0 + gw * U(r, 0.25, 0.75), y0 - gh * 0.1
            kind = r.random()
            s.wash(m, OCHRE, 0.55, wob=0, soft=0, edge=0, grad=s.rgrad(lx, ly, gh * 0.2, gh * 1.3, 0.35, 1.0), flow=0.6, fs=30, gran=0.3)
            s.wash(m, AMBER, 0.32, wob=0, soft=0, edge=0, grad=s.rgrad(lx, ly, gh * 0.4, gh * 1.4, 0.0, 1.0), flow=0.6, fs=30)
            s.wash(s.rect(x0 - 5, y0 - 5, x1 + 5, y0 + gh * 0.16) * m, SIENNA, 0.35, wob=1, soft=2, edge=0.2)       # the ceiling
            if kind < 0.45:            # drapes, half drawn
                for sd in (0, 1):
                    if r.random() < 0.3:
                        continue
                    cw = gw * U(r, 0.18, 0.34)
                    xa = x0 if sd == 0 else x1 - cw
                    col = [SIENNA, ROSE, TEAL, BURNT, OCHRE][int(r.integers(0, 5))]
                    cm = s.rect(xa, y0, xa + cw, y1 + 4) * m
                    s.wash(cm, col, 0.42, wob=1, soft=1.2, edge=0.5, ew=2)
                    for qq in np.linspace(0.2, 0.85, 3):
                        xx = xa + cw * qq
                        s.ink([(xx, y0 + 2), (xx + J(r, 2), (y0 + y1) / 2), (xx + J(r, 2), y1 - 2)], 1.0, BURNT, 0.6, skip=0.2, clip=m)
            elif kind < 0.7:           # a blind, half down
                s.wash(s.rect(x0 - 5, y0 - 5, x1 + 5, y0 + gh * U(r, 0.3, 0.55)) * m, SIENNA, 0.4, wob=0.8, soft=1, edge=0.5, ew=2)
                yy = y0 + 4
                while yy < y0 + gh * 0.5:
                    s.ink([(x0 + 2, yy), (x1 - 2, yy + J(r, 1))], 0.8, BURNT, 0.4, skip=0.3, clip=m)
                    yy += 5
            else:                      # a shelf and its things, a hanging lamp
                sy_ = y0 + gh * U(r, 0.35, 0.5)
                s.wash(s.rect(x0 + gw * 0.1, sy_, x0 + gw * 0.6, sy_ + 3) * m, UMBER, 0.6, wob=0.4, soft=0.6)
                for k in range(int(r.integers(2, 5))):
                    bx = x0 + gw * U(r, 0.12, 0.55)
                    bh = gh * U(r, 0.06, 0.14)
                    s.wash(s.rect(bx, sy_ - bh, bx + gw * U(r, 0.03, 0.07), sy_) * m, [UMBER, BURNT, TEAL, SIENNA][k % 4], 0.5, wob=0.3, soft=0.6)
                s.ink([(lx, y0), (lx, y0 + gh * 0.18)], 1.0, UMBER, 0.8, clip=m)
                s.wash(s.ellipse(lx, y0 + gh * 0.2, gw * 0.06, gh * 0.04) * m, UMBER, 0.6, wob=0.3, soft=0.6)
            if r.random() < 0.4:       # a plant on the sill inside
                px = x0 + gw * U(r, 0.2, 0.8)
                for _ in range(6):
                    s.stroke([(px + J(r, 2), y1 - 2), (px + J(r, gw * 0.08), y1 - gh * U(r, 0.15, 0.3))], max(2, gw * 0.025), MOSS, 0.6, taper=(0, 0.7), wob=0.4, edge=0.3, clip=m)
            s.wash(m, AMBER, 0.12, wob=1, soft=2, edge=0.3, ew=4, flow=0.6, fs=20)
    S.save("bk_glass_lit" if lit else "bk_glass", crop=True, anchor=(0, 0), q=90)


def glass_dark():
    glass(False)


def glass_lit():
    glass(True)


# ---------------------------------------------------------------------------------------------- the two of us, in our windows
def silhouette(which):
    side = "R" if which == "you" else "L"
    F, FL, units, wins = plan(side)
    w = [w for w in wins if w["unit"] == 2 and w["floor"] == 3][0]
    q = np.asarray(glass_quad(F, w), float)
    x0, y0 = q.min(0)
    x1, y1 = q.max(0)
    bx, by = (x0 + x1) / 2, y1
    gw, gh = x1 - x0, y1 - y0
    S = Sheet(W, H, 77 if which == "you" else 78)
    r = rs(77 if which == "you" else 78)
    s = Sub(S, x0 - 20, y0 - 20, x1 + 20, y1 + 20)
    DK = INKBLUE
    if which == "you":
        # the desk edge, the lamp at one end, you asleep on your folded arms
        dy = by - gh * 0.16
        desk = s.rect(x0 + gw * 0.02, dy, x1 - gw * 0.02, dy + gh * 0.05)
        s.wash(desk, SEPIA, 1.1, wob=0.5, soft=0.8, edge=0.4, ew=2, body=1)
        lx = x0 + gw * 0.8
        s.stroke([(lx, dy), (lx - gw * 0.06, dy - gh * 0.28), (lx + gw * 0.02, dy - gh * 0.42)], max(2, gw * 0.02), DK, 1.4, taper=(0, 0), wob=0.3, edge=0.3, body=1)
        shade = Q(s, [(lx - gw * 0.06, dy - gh * 0.45), (lx + gw * 0.06, dy - gh * 0.47), (lx + gw * 0.12, dy - gh * 0.34), (lx - gw * 0.02, dy - gh * 0.32)])
        s.wash(shade, DK, 1.4, wob=0.4, soft=0.7, edge=0.4, ew=2, body=1)
        s.wash(s.ellipse(lx, dy - gh * 0.02, gw * 0.06, gh * 0.02), DK, 1.2, wob=0.3, soft=0.6, body=1)
        cx = x0 + gw * 0.42
        arms = s.blob(cx, dy - gh * 0.07, gw * 0.22, gh * 0.08, irr=0.12, n=12)
        back = Q(s, [(cx - gw * 0.24, dy + 2), (cx - gw * 0.22, dy - gh * 0.16), (cx - gw * 0.1, dy - gh * 0.3), (cx + gw * 0.08, dy - gh * 0.32),
                     (cx + gw * 0.2, dy - gh * 0.2), (cx + gw * 0.26, dy + 2)])
        head = s.blob(cx - gw * 0.02, dy - gh * 0.22, gw * 0.1, gh * 0.11, irr=0.1, n=11)
        body = np.maximum(np.maximum(arms, back), head)
        bm = s.wash(body, NAVY, 1.4, wob=0.6, soft=0.8, edge=0.5, ew=3, body=1, gran=0.3)
        s.wash(bm, DK, 0.9, wob=1, soft=1.5, edge=0.2, clip=bm)
        hair = s.blob(cx - gw * 0.02, dy - gh * 0.25, gw * 0.1, gh * 0.08, irr=0.15, n=10)
        s.wash(hair, INK, 0.9, wob=0.5, soft=0.8, edge=0.3)
        ctr(s, bm, 1.3, INK, 1.2, step=3, skip=0.1)
        meta = {"lamp": [round(lx - bx), round(dy - gh * 0.36 - by)], "fits": [round(x0), round(y0), round(x1), round(y1)]}
    else:
        # sitting up in profile by the window, low ponytail
        dy = by - gh * 0.12
        s.wash(s.rect(x0 + gw * 0.02, dy, x1 - gw * 0.02, dy + gh * 0.05), SEPIA, 1.0, wob=0.5, soft=0.8, edge=0.4, ew=2, body=1)
        cx = x0 + gw * 0.38
        hx, hy = cx + gw * 0.03, dy - gh * 0.5
        hr = gh * 0.095
        torso = Q(s, [(cx - gw * 0.13, dy + 2), (cx - gw * 0.12, dy - gh * 0.22), (cx - gw * 0.05, dy - gh * 0.36), (cx + gw * 0.05, dy - gh * 0.37),
                      (cx + gw * 0.11, dy - gh * 0.25), (cx + gw * 0.14, dy + 2)])
        neck = s.rect(hx - hr * 0.35, hy, hx + hr * 0.35, dy - gh * 0.33)
        head = s.blob(hx, hy, hr * 0.92, hr, irr=0.06, n=12)
        tail = s.ribbon([(hx - hr * 0.8, hy + hr * 0.2), (hx - hr * 1.15, hy + hr * 0.9), (hx - hr * 1.0, hy + hr * 1.6)], hr * 0.55, taper=(0, 0.7))
        knot = s.blob(hx - hr * 0.86, hy + hr * 0.45, hr * 0.3, hr * 0.26, irr=0.1, n=8)
        arm = s.ribbon([(cx + gw * 0.04, dy - gh * 0.25), (cx + gw * 0.12, dy - gh * 0.08), (cx + gw * 0.24, dy - gh * 0.02)], gw * 0.05)
        body = np.maximum.reduce([torso, neck, head, tail, knot, arm])
        bm = s.wash(body, SEPIA, 1.3, wob=0.5, soft=0.8, edge=0.5, ew=3, body=1, gran=0.3)
        s.wash(bm, DK, 0.8, wob=1, soft=1.5, edge=0.2, clip=bm)
        ctr(s, bm, 1.3, INK, 1.2, step=3, skip=0.1)
        cup = s.rect(cx + gw * 0.3, dy - gh * 0.08, cx + gw * 0.36, dy)
        s.wash(cup, DK, 1.2, wob=0.3, soft=0.6, body=1)
        meta = {"fits": [round(x0), round(y0), round(x1), round(y1)]}
    S.save("bk_sil_you" if which == "you" else "bk_sil_me", crop=True, anchor=(bx, by), q=92, meta=meta)


def sil_you():
    silhouette("you")


def sil_me():
    silhouette("me")


# ---------------------------------------------------------------------------------------------- near things
def wires():
    """Wires across the slice of sky, a street lamp on its pole at the right, the edge of a vertical shop sign at the left."""
    S = Sheet(W, H, 79)
    r = rs(79)
    s = Sub(S, 0, 0, W, H)
    # the lamp: a steel pole up the right edge, an arm reaching over the lane, a cobra head
    PX = 2930
    pole = s.ribbon([(PX + 6, H + 20), (PX + 2, 800), (PX - 3, 380)], 44, profile=lambda t: 1 - 0.3 * t)
    iron(s, pole, 0.9, SEPIA, INKBLUE)
    hl(s, [(PX - 10, H), (PX - 12, 800), (PX - 15, 400)], 5, 0.3, dry=0.3)
    arm = [(PX - 2, 420), (PX - 20, 320), (PX - 110, 268), (PX - 260, 262), (PX - 330, 275)]
    am = s.ribbon(arm, 18, profile=lambda t: 1 - 0.35 * t)
    iron(s, am, 0.9)
    head = [(PX - 300, 262), (PX - 395, 266), (PX - 430, 286), (PX - 405, 300), (PX - 300, 292)]
    iron(s, s.poly(head), 1.0)
    lens = s.poly([(PX - 400, 296), (PX - 312, 292), (PX - 318, 302), (PX - 392, 306)])
    s.lift(lens, 1.0, soft=0.5, wob=0.2)
    s.wash(lens, CREAM, 0.25, wob=0.3, soft=0.6, body=1)
    hl(s, [(PX - 30, 330), (PX - 110, 276), (PX - 250, 270)], 3, 0.3)
    s.ink(arm, 2.2, INK, 1.4)
    s.ink(head + [head[0]], 2.0, INK, 1.4)
    line(s, r, (PX - 18, H), (PX - 21, 390), 2.4, INK, 1.4, dbl=0.4)
    line(s, r, (PX + 26, H), (PX + 18, 390), 2.6, INK, 1.5, dbl=0.4)
    for y in (1200, 900, 640):                     # bands and a junction box strapped to the pole
        line(s, r, (PX - 22, y), (PX + 26, y + 2), 2.0, INK, 1.4, over=2)
    jb = s.rect(PX - 40, 980, PX + 20, 1090)
    iron(s, jb, 0.7, SLATE, PAYNE)
    box(s, r, PX - 40, 980, PX + 20, 1090, 2.0, INK, 1.4, over=3)
    lamp = [PX - 356, 300]
    # wires: across the gap, slung from the pole and from both walls
    for a, b, sag, w in (((-20, 60), (PX, 330), 160, 2.6), ((-20, 120), (PX, 350), 210, 2.4), ((-20, 210), (PX - 10, 372), 140, 2.0),
                         ((300, -20), (PX, 400), 240, 2.2), ((-20, 330), (W + 20, 180), 260, 2.0), ((600, -20), (W + 20, 250), 300, 1.8),
                         ((-20, 20), (W + 20, 40), 330, 2.8), ((1100, -20), (W + 20, 120), 180, 1.6)):
        cable(s, r, a, b, sag, w, INKBLUE, 1.5)
    for x, y in ((PX, 340), (PX - 6, 365)):
        s.ink([(x - 22, y), (x + 26, y)], 3, INK, 1.4)
    # the sign: a tall light box on its bracket at the left edge, mostly out of frame
    sx0, sx1, sy0, sy1 = -60, 150, 380, 1500
    sm = s.rect(sx0, sy0, sx1, sy1)
    put(s, sm, (0.92, 0.90, 0.84), 0.5, lift=0, wob=0.6, edge=0.4, ew=4)
    s.wash(sm, OCHRE, 0.12, wob=2, bleed=10, edge=0.1)
    # set the characters one under another
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("L", (s.w, s.h), 0)
    d = ImageDraw.Draw(im)
    fnt = ImageFont.truetype(FONT, 170)
    for i, ch in enumerate("自助洗衣"):
        d.text((60 - s.ox, 560 + i * 230 - s.oy), ch, 255, fnt, anchor="mm")
    tm = np.asarray(im, np.float32) / 255
    s.wash(tm * sm, (0.80, 0.24, 0.22), 1.1, wob=1.0, soft=0.7, edge=0.5, ew=2, gran=0.3)
    s.wash(sm, PAYNE, 0.2, wob=0, soft=0, edge=0, grad=s.vgrad(sy0, sy1, 0.0, 1))
    frame = np.maximum(s.rect(sx0, sy0 - 26, sx1 + 18, sy0), s.rect(sx0, sy1, sx1 + 18, sy1 + 26))
    frame = np.maximum(frame, s.rect(sx1, sy0 - 26, sx1 + 18, sy1 + 26))
    iron(s, frame, 0.8, SLATE, INKBLUE)
    box(s, r, sx0, sy0, sx1, sy1, 2.4, INK, 1.4, over=6, dbl=0.4)
    for y in (sy0 + 60, sy1 - 60):
        iron(s, s.rect(sx1 + 10, y - 8, sx1 + 140, y + 8), 0.9)
        line(s, r, (sx1 + 10, y + 8), (sx1 + 140, y + 8), 1.6, INK, 1.3)
    S.save("bk_wires", crop=True, anchor=(0, 0), q=90, meta={"lamp": lamp, "sign": [0, sy0, sx1, sy1], "p": 1.3})


# ============================================================================================== WIN: one window, close
WW, WH = 2560, 1440
WGL = (720, 380, 1840, 1030)          # the glass, inside the aluminium frame
WFR = (684, 344, 1876, 1066)          # the frame's outside edge = the hole in the wall
WMID = 1280                           # the meeting stiles of the two sliding panes
WLAMP = (1585, 742)                   # the desk lamp's bulb
WDESK = (1300, 905)                   # the front edge of the desk, where you sit
WBACK = (850, 470, 1710, 905)         # the room's back wall as seen through the glass


def win_wall(s, r):
    """The tiled wall at night round the window: small square tiles, grime, the warm spill from the glass."""
    F = s.full()
    s.wash(F, CELADON, 0.7, wob=0, soft=0, edge=0, flow=0.5, fs=300, gran=0.35)
    s.wash(F, TILE, 0.3, wob=0, soft=0, edge=0, flow=0.7, fs=180, gran=0.4)
    s.wash(F, PAYNE, 0.9, wob=0, soft=0, edge=0, flow=0.6, fs=260, gran=0.3)
    s.wash(F, INDIGO, 0.35, wob=0, soft=0, edge=0, flow=0.7, fs=200, gran=0.3, grad=s.vgrad(0, WH, 1, 0.5))
    mottle(s, r, 100, 100, WW - 100, WH - 100, [DUSK, SLATE, PRUSSIAN, VIOLET], 10, 0.14, size=(90, 220))
    X, Y = s.X.astype(np.float32), s.Y.astype(np.float32)
    tilework(s, r, None, F, X, Y, 27.0, k=0.5, patch=(0.36, 0.56), seed=901, odd=(PAYNE, SLATE, PRUSSIAN), wobble=0.12, lift=0.08)
    # the light from the window, falling on the wall round it and on the sill
    sp = np.clip(s.rgrad(1280, 720, 350, 1300, 1, 0, sy=0.62), 0, 1)
    s.wash(F, None, 0.78, wob=0, soft=0, edge=0, gran=0, flow=0.4, fs=200, grad=sp, lift=True)
    s.wash(F, AMBER, 0.42, wob=0, soft=0, edge=0, grad=sp * sp, flow=0.5, fs=140, gran=0.3)
    s.wash(F, OCHRE, 0.2, wob=0, soft=0, edge=0, grad=sp, flow=0.5, fs=140, gran=0.3)
    s.wash(F, ROSE, 0.2, wob=0, soft=0, edge=0, grad=sp, flow=0.5, fs=140, gran=0.3)
    s.wash(F, OCHRE, 0.2, wob=0, soft=0, edge=0, grad=np.clip(s.rgrad(1280, 1120, 100, 700, 1, 0, sy=0.4), 0, 1), flow=0.5, fs=120)
    # grime: black water stains from the sill and the AC, a patch where tiles fell off
    streaks(s, r, [(U(r, 700, 1860), 1110) for _ in range(20)], 300, 16, PAYNE, 0.28, (6, 22))
    streaks(s, r, [(U(r, 2070, 2330), 640) for _ in range(8)], 500, 7, PAYNE, 0.3, (6, 20))
    streaks(s, r, [(U(r, 600, 1960), 250) for _ in range(12)], 120, 8, PAYNE, 0.22, (5, 14))


def win_room(s, r):
    """The room through the glass: back wall in lamp light, a shelf, a cork board with notes, the desk, a half-drawn curtain."""
    x0, y0, x1, y1 = WGL
    bx0, by0, bx1, by1 = WBACK
    gm = s.rect(x0 - 6, y0 - 6, x1 + 6, y1 + 6)
    s.lift(gm, 1.0, soft=0.5, wob=0)
    s.wash(gm, CREAM, 0.4, wob=0, soft=0.5, edge=0, flow=0.5, fs=200, gran=0.3)
    lx, ly = WLAMP
    far_ = s.rgrad(lx, ly + 40, 80, 820, 0, 1, sy=0.8)
    s.wash(gm, OCHRE, 0.75, wob=0, soft=0, edge=0, grad=0.35 + 0.65 * far_, flow=0.6, fs=160, gran=0.35)
    s.wash(gm, SIENNA, 0.55, wob=0, soft=0, edge=0, grad=far_ ** 1.4, flow=0.6, fs=120, gran=0.35)
    s.wash(gm, UMBER, 0.35, wob=0, soft=0, edge=0, grad=far_ ** 2.5, flow=0.7, fs=100, gran=0.3)
    # ceiling and side walls of the room
    ceil = Q(s, [(x0 - 6, y0 - 6), (x1 + 6, y0 - 6), (bx1, by0), (bx0, by0)])
    s.wash(ceil, SIENNA, 0.45, wob=1.5, bleed=8, edge=0.2, grad=s.vgrad(y0, by0, 1, 0.5))
    s.wash(ceil, UMBER, 0.25, wob=1.5, bleed=8, edge=0.2)
    lw = Q(s, [(x0 - 6, y0 - 6), (bx0, by0), (bx0, by1), (x0 - 6, y1 + 6)])
    s.wash(lw, SIENNA, 0.35, wob=1.5, bleed=10, edge=0.2)
    rw = Q(s, [(x1 + 6, y0 - 6), (bx1, by0), (bx1, by1), (x1 + 6, y1 + 6)])
    s.wash(rw, OCHRE, 0.25, wob=1.5, bleed=10, edge=0.2)
    for a, b in (((bx0, by0), (x0, y0)), ((bx1, by0), (x1, y0)), ((bx0, by0), (bx0, by1)), ((bx1, by0), (bx1, by1))):
        s.ink([a, ((a[0] + b[0]) / 2 + J(r, 2), (a[1] + b[1]) / 2 + J(r, 2)), b], 1.4, BURNT, 0.6, skip=0.3, tremor=1.0)
    s.ink([(bx0, by0), (bx1, by0)], 1.4, BURNT, 0.55, skip=0.3, tremor=1.0)
    # a ceiling light, off
    s.wash(s.ellipse(1280, y0 + 30, 70, 12), CREAM, 0.3, wob=0.6, soft=1, edge=0.4)
    s.ink(ell(1280, y0 + 30, 70, 12, 0, 2 * math.pi, 30), 1.4, BURNT, 0.7)
    # the shelf over the desk, its books and a plant
    sy_ = 600
    s.stroke([(1180, sy_), (1460, sy_ - 2), (1700, sy_)], 9, UMBER, 0.75, taper=(0, 0), wob=1, edge=0.4, soft=1.2)
    s.wash(s.rect(1180, sy_ + 4, 1700, sy_ + 24), SIENNA, 0.35, wob=2, bleed=6, edge=0)
    xx = 1200
    for i in range(14):
        bw_ = U(r, 14, 30)
        bh = U(r, 60, 105)
        if r.random() < 0.15:
            xx += U(r, 20, 50)
            continue
        c = [BURNT, TEAL, SIENNA, ROSE, UMBER, OCHRE, MOSS, PRUSSIAN][int(r.integers(0, 8))]
        lean = J(r, 6) if r.random() < 0.3 else 0
        bm = Q(s, [(xx, sy_), (xx + bw_, sy_), (xx + bw_ + lean, sy_ - bh), (xx + lean, sy_ - bh)])
        s.wash(bm, c, U(r, 0.45, 0.8), wob=0.8, soft=1.2, edge=0.45, ew=3, flow=0.6, fs=20)
        if r.random() < 0.5:
            s.ink([(xx + 3, sy_ - bh * 0.7), (xx + bw_ - 3, sy_ - bh * 0.7)], 1.0, UMBER, 0.6)
        xx += bw_ + U(r, 0, 4)
        if xx > 1560:
            break
    pp = s.rect(1620, sy_ - 50, 1668, sy_)
    s.wash(pp, TERRA, 0.7, wob=0.6, soft=1, edge=0.4)
    for _ in range(9):
        a = U(r, -2.8, -0.3)
        s.stroke([(1644, sy_ - 48), (1644 + math.cos(a) * U(r, 40, 80), sy_ - 48 + math.sin(a) * U(r, 30, 70))], 12, MOSS, 0.6, taper=(0.1, 0.8), wob=0.8, edge=0.4, soft=1.2)
    # the cork board, pinned with yellow notes
    cb = s.rect(905, 520, 1120, 735)
    s.wash(cb, SIENNA, 0.6, wob=1, soft=1.4, edge=0.5, ew=4, flow=0.8, fs=16, gran=0.6)
    s.ink([(905, 520), (1120, 522), (1121, 735), (904, 733), (905, 520)], 2.0, UMBER, 0.8, smooth=False)
    for nx, ny, rot in ((940, 560, 0.05), (1010, 600, -0.08), (1060, 545, 0.1), (960, 650, -0.04), (1050, 670, 0.06)):
        P = np.array([(-26, -26), (26, -26), (26, 26), (-26, 26)], float)
        c_, s_ = math.cos(rot), math.sin(rot)
        P = np.stack([nx + P[:, 0] * c_ - P[:, 1] * s_, ny + P[:, 0] * s_ + P[:, 1] * c_], 1)
        nm = Q(s, P)
        s.lift(nm, 0.8, soft=0.8, wob=0)
        s.wash(nm, NOTE, 0.55, wob=0.6, soft=1, edge=0.4, ew=2)
        s.wash(s.ellipse(nx, ny - 20, 4, 4), ROSE, 0.9, wob=0.2, soft=0.6)
    # a calendar beside it
    s.wash(s.rect(1135, 640, 1180, 720), CREAM, 0.15, wob=0.5, soft=1, edge=0.4)
    s.wash(s.rect(1135, 640, 1180, 655), ROSE, 0.4, wob=0.5, soft=1)
    # the desk along the back wall: its top, seen a little from above, and its front
    dy = WDESK[1]
    top = Q(s, [(bx0 - 40, dy - 34), (bx1 + 40, dy - 34), (bx1 + 70, dy), (bx0 - 70, dy)])
    s.wash(top, SIENNA, 0.6, wob=0.8, soft=1, edge=0.4, ew=3, flow=0.6, fs=50, gran=0.4)
    s.wash(top, UMBER, 0.3, wob=1, soft=2, edge=0, grad=s.hgrad(bx0, lx, 1, 0.1))
    front = s.rect(bx0 - 70, dy, bx1 + 70, y1 + 10)
    s.wash(front, UMBER, 0.85, wob=0.8, soft=1, edge=0.4, ew=4, flow=0.6, fs=60, gran=0.4)
    s.wash(front, SEPIA, 0.5, wob=1, soft=2, edge=0, grad=s.vgrad(dy, y1, 0.3, 1))
    s.ink([(bx0 - 70, dy), (1280, dy + 2), (bx1 + 70, dy)], 2.2, SEPIA, 1.2)
    # things on the desk: a mug, a phone face down, a stack of paper
    s.wash(s.rect(1470, dy - 64, 1506, dy - 22), CREAM, 0.35, wob=0.4, soft=0.8, edge=0.4, ew=2)
    s.wash(s.rect(1490, dy - 64, 1506, dy - 22), SIENNA, 0.3, wob=0.4, soft=0.8)
    s.ink(ell(1512, dy - 44, 10, 12, -1.4, 1.4, 12), 1.6, UMBER, 0.9)
    s.wash(Q(s, [(1000, dy - 30), (1100, dy - 32), (1110, dy - 18), (995, dy - 16)]), CREAM, 0.12, wob=0.5, soft=0.8, edge=0.5, ew=2)
    s.ink([(995, dy - 16), (1110, dy - 18)], 1.2, UMBER, 0.7)
    # the lamp: an arm lamp at the right, its pool on the desk and the wall
    pool = np.clip(s.rgrad(lx, dy - 20, 30, 260, 1, 0, sy=0.35), 0, 1)
    s.wash(pool, None, 0.6, wob=0, soft=0, edge=0, gran=0, flow=0, grad=None, lift=True)
    s.wash(pool, OCHRE, 0.25, wob=0, soft=0, edge=0)
    halo = np.clip(s.rgrad(lx, ly, 20, 300, 1, 0), 0, 1)
    s.wash(halo, None, 0.55, wob=0, soft=0, edge=0, gran=0, flow=0, lift=True)
    s.wash(halo, AMBER, 0.12, wob=0, soft=0, edge=0)
    arm = [(lx + 60, dy - 30), (lx + 40, dy - 150), (lx + 90, dy - 230), (lx + 30, ly - 20)]
    s.stroke(arm, 7, SEPIA, 1.1, taper=(0, 0), wob=0.3, edge=0.3)
    s.wash(s.ellipse(lx + 60, dy - 28, 40, 8), SEPIA, 1.0, wob=0.3, soft=0.8)
    shade = Q(s, [(lx - 40, ly - 50), (lx + 40, ly - 62), (lx + 62, ly + 2), (lx - 30, ly + 14)])
    s.wash(shade, PRUSSIAN, 0.9, wob=0.5, soft=0.8, edge=0.5, ew=3)
    s.wash(shade, SEPIA, 0.5, wob=0.5, soft=1)
    s.lift(s.ellipse(lx + 12, ly + 6, 30, 7), 1.0, soft=2)
    # the curtain at the left, half drawn, the lamp coming through it
    cx1 = 930
    cm = Q(s, [(x0 - 6, y0 + 4), (cx1, y0 + 4), (cx1 - 20, y0 + 300), (cx1 + 15, y1 + 6), (x0 - 6, y1 + 6)])
    s.lift(cm, 0.85, soft=1.2, wob=1.5)
    cm = s.wash(cm, CREAM, 0.5, wob=1.5, soft=1.2, edge=0.5, ew=4, flow=0.6, fs=40)
    s.wash(cm, ROSE, 0.22, wob=2, bleed=8, edge=0.2, clip=cm)
    s.wash(cm, OCHRE, 0.3, wob=2, bleed=8, edge=0.2, grad=s.hgrad(x0, cx1, 0.3, 1), clip=cm)
    s.wash(cm, SIENNA, 0.3, wob=1, soft=2, edge=0, grad=s.hgrad(x0, cx1, 1, 0.0), clip=cm)
    for q in np.linspace(0.08, 0.92, 9):
        xx = x0 + (cx1 - x0) * q
        s.stroke([(xx, y0 + 10), (xx + J(r, 4), (y0 + y1) / 2), (xx + J(r, 8) + q * 12, y1)], U(r, 6, 16), SIENNA, U(r, 0.2, 0.4), dry=0.3, edge=0.3, wob=1, taper=(0, 0.1), clip=cm)
        s.ink([(xx + 4, y0 + 12), (xx + 4 + J(r, 3), (y0 + y1) / 2), (xx + J(r, 6) + q * 12, y1 - 4)], 1.1, BURNT, 0.5, skip=0.3, clip=cm)
    s.stroke([(x0 - 6, y0 + 8), (x1 + 6, y0 + 8)], 8, UMBER, 0.8, taper=(0, 0), wob=0.5, edge=0.4)            # the curtain rail
    # the glass: a cool sheen and the night's reflection in its corners
    sheen = Q(s, [(1350, y0), (1520, y0), (1300, y1), (1130, y1)])
    s.wash(sheen, None, 0.2, wob=1, soft=6, edge=0, gran=0, flow=0, lift=True)
    s.wash(s.rect(x1 - 300, y0, x1, y0 + 120) * s.rgrad(x1, y0, 0, 300, 1, 0), RAIN, 0.3, wob=1, soft=4, edge=0)


def win_frame(s, r, body=0.0):
    """The aluminium window: frame, two sliding panes, the meeting stiles, a latch."""
    fx0, fy0, fx1, fy1 = WFR
    x0, y0, x1, y1 = WGL
    frm = np.clip(s.rect(fx0, fy0, fx1, fy1) - s.rect(x0, y0, x1, y1), 0, 1)
    sti = s.rect(WMID - 18, y0, WMID + 18, y1)
    m = np.maximum(frm, sti)
    for xa, xb in ((x0 - 2, x0 + 12), (x1 - 12, x1 + 2)):
        m = np.maximum(m, s.rect(xa, y0, xb, y1))
    m = np.maximum(m, s.rect(x0, y0 - 2, x1, y0 + 12))
    m = np.maximum(m, s.rect(x0, y1 - 12, x1, y1 + 2))
    put(s, m, (0.86, 0.85, 0.82), 0.5, lift=1.0, wob=0.5, edge=0.4, ew=3, flow=0.5, fs=60)
    if body:
        s.body[:] = np.maximum(s.body, m)
    s.wash(m, SLATE, 0.45, wob=1, soft=2, edge=0, grad=s.vgrad(fy0, fy1, 1, 0.4))
    s.wash(m, AMBER, 0.35, wob=1, soft=2, edge=0, grad=np.clip(s.rgrad(1280, 760, 200, 700, 1, 0), 0, 1))
    hl(s, [(WMID - 12, y0 + 6), (WMID - 12, y1 - 6)], 4, 0.4)
    hl(s, [(x0 + 4, y1 - 4), (x1 - 4, y1 - 4)], 3, 0.3)
    box(s, r, fx0, fy0, fx1, fy1, 2.6, SEPIA, 1.5, over=10, dbl=0.5)
    box(s, r, x0, y0, x1, y1, 2.0, SEPIA, 1.4, over=6, dbl=0.4)
    for xx in (WMID - 18, WMID + 18):
        line(s, r, (xx, y0), (xx, y1), 2.0, SEPIA, 1.4, over=2)
    line(s, r, (WMID, y0 + 4), (WMID, y1 - 4), 1.2, SEPIA, 1.0, over=0, skip=0.2)
    for xx in (x0 + 12, x1 - 12):
        line(s, r, (xx, y0 + 12), (xx, y1 - 12), 1.3, SEPIA, 1.1, over=0, skip=0.15)
    put(s, s.rect(WMID - 12, 690, WMID + 12, 740), SLATE, 0.6, lift=0.5, wob=0.3)
    box(s, r, WMID - 12, 690, WMID + 12, 740, 1.4, SEPIA, 1.3, over=1)


def win_bg():
    s = Sheet(WW, WH, 90)
    r = rs(90)
    win_wall(s, r)
    fx0, fy0, fx1, fy1 = WFR
    # the window's reveal in the wall: the top and the right jamb catch a little of the room's light
    rev = Q(s, [(fx0 - 40, fy0 - 40), (fx1 + 40, fy0 - 40), (fx1, fy0), (fx0, fy0)])
    put(s, rev, GREY, 0.6, lift=1.0, wob=0.6, edge=0.4, ew=3)
    s.wash(rev, PAYNE, 0.8, wob=1, soft=2, edge=0.2)
    for sd in (0, 1):
        xa, xb = (fx0 - 40, fx0) if sd == 0 else (fx1, fx1 + 40)
        jm = Q(s, [(xa if sd == 0 else xb, fy0 - 40), (xb if sd == 0 else xa, fy0), (xb if sd == 0 else xa, fy1), (xa if sd == 0 else xb, fy1 + 40)])
        put(s, jm, GREY, 0.55, lift=1.0, wob=0.6, edge=0.4, ew=3)
        s.wash(jm, PAYNE, 0.6, wob=1, soft=2, edge=0.2)
        s.wash(jm, AMBER, 0.25, wob=1, soft=3, edge=0)
    win_room(s, r)
    win_frame(s, r)
    # the sill outside, wet, catching the light
    sm = s.rect(fx0 - 60, fy1 + 40, fx1 + 60, fy1 + 70)
    put(s, sm, GREY, 0.55, lift=1.0, wob=0.6, edge=0.45, ew=3)
    s.wash(sm, PAYNE, 0.4, wob=1, soft=2)
    s.wash(s.rect(fx0 - 60, fy1 + 40, fx1 + 60, fy1 + 48), OCHRE, 0.4, wob=0.6, soft=1)
    box(s, r, fx0 - 60, fy1 + 40, fx1 + 60, fy1 + 70, 2.2, SEPIA, 1.5, over=8, dbl=0.4)
    s.wash(s.rect(fx0 - 50, fy1 + 70, fx1 + 50, fy1 + 110), PAYNE, 0.5, wob=2, bleed=8, edge=0.1)
    # a concrete hood over it
    hm = s.rect(fx0 - 120, fy0 - 150, fx1 + 120, fy0 - 110)
    put(s, hm, GREY, 0.5, lift=1.0, wob=0.6, edge=0.45, ew=3)
    s.wash(hm, PAYNE, 0.75, wob=1, soft=2)
    sof = Q(s, [(fx0 - 120, fy0 - 110), (fx1 + 120, fy0 - 110), (fx1 + 60, fy0 - 70), (fx0 - 60, fy0 - 70)])
    put(s, sof, GREY, 0.5, lift=1.0, wob=0.6, edge=0.4, ew=3)
    s.wash(sof, PAYNE, 0.5, wob=1, soft=2)
    s.wash(sof, AMBER, 0.3, wob=1, soft=3, grad=s.vgrad(fy0 - 110, fy0 - 70, 0, 1))
    box(s, r, fx0 - 120, fy0 - 150, fx1 + 120, fy0 - 110, 2.4, SEPIA, 1.5, over=8, dbl=0.4)
    line(s, r, (fx0 - 60, fy0 - 70), (fx1 + 60, fy0 - 70), 1.6, SEPIA, 1.2, over=4)
    streaks(s, r, [(U(r, fx0 - 100, fx1 + 100), fy0 - 110) for _ in range(10)], 120, 8, PAYNE, 0.3, (5, 14))
    # an AC box through the wall at the right, its drip; a downpipe at the left; cables
    ax0, ay0, ax1, ay1 = 2060, 470, 2340, 650
    am = s.rect(ax0, ay0, ax1, ay1)
    put(s, am, (0.84, 0.83, 0.78), 0.55, lift=1.0, wob=0.6, edge=0.45, ew=3)
    s.wash(am, PAYNE, 0.9, wob=1, soft=2, edge=0, grad=s.hgrad(ax0, ax1, 0.6, 1))
    s.wash(am, AMBER, 0.3, wob=1, soft=3, edge=0, grad=s.hgrad(ax0, ax1, 1, 0))
    side = Q(s, [(ax0, ay0), (ax0 - 40, ay0 - 30), (ax0 - 40, ay1 - 30), (ax0, ay1)])
    put(s, side, GREY, 0.5, lift=1.0, wob=0.5)
    s.wash(side, AMBER, 0.3, wob=1, soft=2)
    s.wash(side, PAYNE, 0.4, wob=1, soft=2)
    for yy in np.arange(ay0 + 20, ay1 - 10, 14):
        line(s, r, (ax0 + 14, yy), (ax1 - 14, yy), 1.3, SEPIA, 1.1, over=0, skip=0.1)
    box(s, r, ax0, ay0, ax1, ay1, 2.4, SEPIA, 1.5, over=5, dbl=0.4)
    path(s, [(ax0, ay0), (ax0 - 40, ay0 - 30), (ax0 - 40, ay1 - 30), (ax0, ay1)], 2.0, SEPIA, 1.4, over=3, r=r)
    s.stroke([(2200, ay1), (2204, ay1 + 300), (2198, ay1 + 600)], 12, PAYNE, 0.35, dry=0.3, edge=0.4, wob=1.2, taper=(0, 0.7))
    pm = s.ribbon([(250, -10), (252, WH + 10)], 46, smooth=False)
    put(s, pm, GREY, 0.5, lift=1.0, wob=0.5, edge=0.4, ew=3)
    s.wash(pm, PAYNE, 0.75, wob=1, soft=2, grad=s.hgrad(228, 274, 0.5, 1))
    line(s, r, (228, -10), (230, WH + 10), 1.8, SEPIA, 1.3)
    line(s, r, (274, -10), (276, WH + 10), 2.4, SEPIA, 1.5)
    for yy in (180, 700, 1220):
        line(s, r, (218, yy), (284, yy + 2), 3, SEPIA, 1.5, over=2)
    for k_ in range(4):
        cable(s, r, (-20, 1260 + k_ * 16), (WW + 20, 1180 + k_ * 20), 70 + k_ * 12, 2.6, INKBLUE, 1.4)
    cable(s, r, (300, -20), (1100, 260), 30, 2.0, INKBLUE, 1.2)
    # the cage's tin hat just showing at the top, the next window's cage bottom at the top edge
    for xx in np.arange(fx0 - 160, fx1 + 160, 46):
        line(s, r, (xx, -10), (xx + J(r, 2), 70), 2.6, INKBLUE, 1.5, over=0)
    line(s, r, (fx0 - 170, 70), (fx1 + 170, 72), 3.4, INKBLUE, 1.6, dbl=0.5)
    s.save("wn_bg", opaque=True, q=90, meta={"glass": list(WGL), "lamp": list(WLAMP), "desk": list(WDESK), "frame": list(WFR), "mid": WMID})


def win_grille():
    """The aluminium frame again (so the figures and the note sit behind it) and the iron cage in front, crisp."""
    s = Sheet(WW, WH, 91)
    r = rs(91)
    win_frame(s, r, body=1.0)
    fx0, fy0, fx1, fy1 = WFR
    gx0, gx1, gy0, gy1 = fx0 - 70, fx1 + 70, fy0 - 70, fy1 + 30
    bw = 4.0
    # the cage floor (flat bars) and a pot on it at the right
    for xx in np.arange(gx0, gx1, 60):
        iron(s, s.ribbon([(xx, gy1), (xx + 30, gy1 + 34)], bw * 1.4, smooth=False), 0.9)
    iron(s, s.rect(gx0 - 6, gy1 - 4, gx1 + 36, gy1 + 6), 0.9)
    iron(s, s.rect(gx0 + 24, gy1 + 30, gx1 + 36, gy1 + 38), 0.9)
    for (px, sz) in ((1700, 1.0), (1780, 0.75)):
        pot = [(px - 50 * sz, gy1 - 70 * sz), (px + 50 * sz, gy1 - 70 * sz), (px + 38 * sz, gy1), (px - 38 * sz, gy1)]
        pm = put(s, Q(s, pot), TERRA, 0.8, lift=1.0, wob=0.5, edge=0.5, ew=3)
        s.wash(pm, PAYNE, 0.7, wob=1, soft=2, grad=s.hgrad(px - 50, px + 50, 0.6, 1))
        s.wash(pm, AMBER, 0.25, wob=1, soft=2, grad=s.hgrad(px - 50, px + 50, 1, 0))
        path(s, pot, 2.2, INK, 1.4, closed=True, over=3, r=r)
        lm = s.zeros()
        for _ in range(int(14 * sz)):
            a = U(r, -2.7, -0.4)
            L = U(r, 50, 120) * sz
            cx, cy = px + math.cos(a) * L * 0.6, gy1 - 70 * sz + math.sin(a) * L * 0.8
            lm = np.maximum(lm, s.blob(cx, cy, L * 0.34, L * 0.17, irr=0.3, n=8, rot=a))
        lm = put(s, lm, MOSS, 1.0, lift=1.0, wob=0.6, edge=0.5, ew=3, flow=0.6, fs=30, gran=0.4)
        s.wash(lm, INDIGO, 0.8, wob=1, soft=2, edge=0.2, clip=lm)
        s.wash(lm, AMBER, 0.3, wob=1, soft=3, grad=s.vgrad(gy1 - 200, gy1, 1, 0), clip=lm)
        ctr(s, lm, 1.8, INK, 1.3, step=3, skip=0.15)
    # bars: uprights, rails, a band of loops
    xx = gx0
    while xx <= gx1 + 1:
        iron(s, s.ribbon([(xx, gy0), (xx + J(r, 1), gy1)], bw * (1.6 if xx in (gx0,) or xx > gx1 - 40 else 1.0), smooth=False), 0.95)
        xx += 52
    for yy, w_ in ((gy0, 9), (gy1, 9), (gy0 + 150, 6), (gy0 + 250, 6)):
        iron(s, s.ribbon([(gx0 - 8, yy), (gx1 + 8, yy + J(r, 1.5))], w_, smooth=False), 0.95)
    xx = gx0 + 26
    while xx < gx1:
        lp = ell(xx, gy0 + 200, 22, 46, 0, 2 * math.pi, 32)
        iron(s, s.ribbon(lp, 3.2, closed=True), 0.9)
        xx += 52
    # the warm light along the bars' inner edges, the wet sky along their tops
    xx = gx0
    while xx <= gx1 + 1:
        hl(s, [(xx + 1.5, gy0 + 10), (xx + 1.5, gy1 - 10)], 1.6, 0.4)
        s.wash(s.ribbon([(xx + 1.5, gy0 + 10), (xx + 1.5, gy1 - 10)], 1.6, smooth=False), AMBER, 0.4, wob=0, soft=0.5)
        line(s, r, (xx - 2.5, gy0), (xx - 2.5 + J(r, 1), gy1), 1.6, INK, 1.4, over=0)
        xx += 52
    hl(s, [(gx0, gy0 - 3), (gx1, gy0 - 3)], 2.6, 0.4)
    line(s, r, (gx0 - 10, gy0 + 5), (gx1 + 10, gy0 + 6), 2.0, INK, 1.4, over=3)
    line(s, r, (gx0 - 10, gy1 + 5), (gx1 + 10, gy1 + 6), 2.0, INK, 1.4, over=3)
    s.save("wn_grille", crop=True, anchor=(0, 0), q=92)


def win_figure(up):
    """You at the desk, from behind, dark against the lamp: head down on folded arms; or sitting up, a hand at the neck."""
    s = Sheet(WW, WH, 92 + up)
    r = rs(92 + up)
    dx, dy = WDESK
    gb = WGL[3] + 20

    def dress(bm, rx0, rx1):
        bm = s.wash(bm, NAVY, 1.25, wob=0.8, soft=1, edge=0.55, ew=6, flow=0.5, fs=60, gran=0.35, body=1)
        s.wash(bm, SEPIA, 0.45, wob=2, bleed=8, edge=0.2, flow=0.8, fs=40, clip=bm)
        s.wash(bm, INDIGO, 0.35, wob=2, bleed=10, edge=0.2, flow=0.8, fs=50, grad=s.vgrad(dy - 250, gb, 0.2, 1), clip=bm)
        # the lamp is at the right: it catches the rims on that side
        rim = np.clip(bm - cv2.GaussianBlur(bm, (0, 0), 5), 0, 1) * 2.4 * s.hgrad(rx0, rx1, 0, 1)
        s.wash(np.clip(rim, 0, 1), None, 0.55, wob=0, soft=1, edge=0, gran=0, flow=0, lift=True)
        s.wash(np.clip(rim, 0, 1), AMBER, 0.4, wob=0, soft=1, edge=0)
        return bm

    if not up:
        # from behind, slumped over the desk: elbows out on it, shoulders up, the head sunk between them on the arms
        torso = s.poly([(dx - 150, gb), (dx - 158, dy - 40), (dx - 176, dy - 100), (dx - 140, dy - 136), (dx - 60, dy - 152), (dx + 60, dy - 154),
                        (dx + 146, dy - 138), (dx + 180, dy - 100), (dx + 166, dy - 40), (dx + 154, gb)])
        el = s.blob(dx - 205, dy - 20, 62, 30, irr=0.08, n=12)
        er = s.blob(dx + 208, dy - 20, 62, 30, irr=0.08, n=12)
        head = s.blob(dx + 8, dy - 146, 60, 54, irr=0.05, n=14)
        bm = dress(np.maximum.reduce([torso, el, er, head]), dx - 60, dx + 240)
        hair = s.blob(dx + 8, dy - 150, 60, 52, irr=0.06, n=14) * s.vgrad(dy - 120, dy - 140, 0, 1)
        s.wash(hair, INK, 1.3, wob=0.8, soft=1, edge=0.4, ew=4, gran=0.3, body=1)
        s.ink(ell(dx + 8, dy - 150, 54, 46, 3.5, 5.9, 20), 1.6, SEPIA, 0.7, skip=0.3)
        s.ink([(dx - 120, dy - 130), (dx - 40, dy - 112), (dx + 60, dy - 112), (dx + 130, dy - 132)], 1.6, INK, 0.8, skip=0.25)     # the collar under the head
        s.ink([(dx + 4, dy - 100), (dx + 6, gb)], 1.4, INK, 0.6, skip=0.35)                                                   # the back seam
        s.ink([(dx - 168, dy - 60), (dx - 230, dy - 40), (dx - 262, dy - 18)], 1.4, INK, 0.8, skip=0.2)
        s.ink([(dx + 172, dy - 60), (dx + 232, dy - 40), (dx + 266, dy - 18)], 1.4, INK, 0.8, skip=0.2)
        ctr(s, bm, 2.2, INK, 1.4, step=3)
        name = "wn_you"
    else:
        # sitting up, the head raised and tipped, a hand rubbing the back of the neck, the elbow out to the right
        torso = s.poly([(dx - 150, gb), (dx - 156, dy - 120), (dx - 132, dy - 196), (dx - 62, dy - 220), (dx + 62, dy - 222),
                        (dx + 132, dy - 198), (dx + 156, dy - 120), (dx + 150, gb)])
        neck = s.rect(dx - 26, dy - 262, dx + 26, dy - 210)
        head = s.blob(dx + 6, dy - 306, 56, 64, irr=0.04, n=14, rot=0.14)
        upper = s.ribbon([(dx + 118, dy - 200), (dx + 196, dy - 262), (dx + 184, dy - 330)], 48, taper=(0, 0.15))
        fore = s.ribbon([(dx + 184, dy - 330), (dx + 110, dy - 290), (dx + 46, dy - 262)], 36, taper=(0, 0.25))
        bm = dress(np.maximum.reduce([torso, neck, head, upper, fore]), dx - 60, dx + 240)
        hair = s.blob(dx + 6, dy - 314, 58, 58, irr=0.05, n=14, rot=0.14) * s.vgrad(dy - 250, dy - 272, 0, 1)
        s.wash(hair, INK, 1.3, wob=0.8, soft=1, edge=0.4, ew=4, gran=0.3, body=1)
        hand = s.blob(dx + 40, dy - 262, 32, 22, irr=0.15, n=10, rot=0.5)
        hm = s.wash(hand, SKIN, 0.6, wob=0.6, soft=1, edge=0.4, ew=3, body=1)
        s.wash(hm, SEPIA, 0.7, wob=1, soft=2, clip=hm)
        s.wash(hm, AMBER, 0.25, wob=1, soft=2, grad=s.hgrad(dx + 10, dx + 70, 0, 1), clip=hm)
        s.ink([(dx - 24, dy - 214), (dx, dy - 206), (dx + 26, dy - 214)], 1.6, INK, 0.9)
        s.ink([(dx + 2, dy - 200), (dx + 4, gb)], 1.4, INK, 0.6, skip=0.35)
        s.ink([(dx + 136, dy - 190), (dx + 150, dy - 110)], 1.4, INK, 0.6, skip=0.3)
        ctr(s, bm, 2.2, INK, 1.4, step=3)
        ctr(s, hm, 1.6, INK, 1.2, step=3)
        name = "wn_you_up"
    s.save(name, crop=True, anchor=(dx, dy), q=92)


def win_you():
    win_figure(0)


def win_you_up():
    win_figure(1)


def win_note():
    """A blank yellow sticky note stuck inside the glass, seen from outside, lit from behind. Painted large (300 px);
    on the window it is about 0.08 m = 45 px at wn_bg's scale, so draw it at s 0.15-0.3 there, or larger for a close."""
    s = Sheet(420, 420, 93)
    r = rs(93)
    c = 210
    P = np.array([(-150, -150), (150, -146), (152, 150), (-148, 152)], float)
    rot = -0.04
    cr, sr = math.cos(rot), math.sin(rot)
    P = np.stack([c + P[:, 0] * cr - P[:, 1] * sr, c + P[:, 0] * sr + P[:, 1] * cr], 1)
    m = Q(s, P)
    m = s.wash(m, NOTE, 0.75, wob=1.2, soft=1, edge=0.55, ew=8, flow=0.5, fs=60, gran=0.35, body=1)
    s.wash(m, OCHRE, 0.3, wob=2, bleed=12, edge=0.2, grad=s.rgrad(c, c, 40, 220, 0, 1), clip=m)
    s.wash(m, None, 0.3, wob=0, soft=0, edge=0, gran=0, flow=0, grad=s.rgrad(c - 10, c - 10, 10, 160, 1, 0), lift=True)
    # the gummed strip at the top shows as a band, a little denser; the bottom edge lifts off the glass
    s.wash(Q(s, [P[0], P[1], P[1] + (0, 60), P[0] + (0, 60)]), OCHRE, 0.25, wob=1, soft=2, edge=0.3)
    s.wash(Q(s, [P[3] + (0, -40), P[2] + (0, -40), P[2], P[3]]), AMBER, 0.2, wob=1, soft=3, edge=0)
    # the writing on the other side shows through, reversed and unreadable: a few soft grey marks
    for i in range(3):
        y = c - 50 + i * 48 + J(r, 6)
        xa = c - 100 + J(r, 20)
        pts = [(xa + q * U(r, 140, 190), y + 6 * math.sin(q * 9 + i) + J(r, 3)) for q in np.linspace(0, 1, 8)]
        s.stroke(pts, 9, UMBER, 0.12, taper=(0.1, 0.3), wob=2, edge=0.1, soft=3, gran=0.2)
    path(s, [tuple(p) for p in P], 2.4, SEPIA, 1.2, closed=True, over=3, r=r)
    s.save("wn_note", crop=True, anchor=(c, c), q=92, meta={"true_px": 45})


ALL = {"block_sky": sky, "block_sky_dawn": sky_dawn, "block_far": far, "block_glass": glass_dark, "block_glass_lit": glass_lit,
       "block_left": left, "block_right": right, "block_sil_you": sil_you, "block_sil_me": sil_me, "block_wires": wires,
       "win_bg": win_bg, "win_grille": win_grille, "win_you": win_you, "win_you_up": win_you_up, "win_note": win_note}
