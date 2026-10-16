"""The two of us, never a face: always from behind. 我 (me): dark-brown hair in a low loose ponytail, a cream knit
cardigan, a mint scarf at night, light-blue jeans, white sneakers. 你 (you): a little taller, short black hair, a navy
jacket over a pale grey tee, dark trousers, dark shoes. Light comes from the upper left throughout.

Walking away from the camera, side by side (me left, you right, shoulders ~40 px apart), sheet 1000x1300, anchor = the
ground midpoint between our feet (sheet 490, 1230). Two drawings each, w0 = both left feet forward, w1 = both right
feet forward (we walk in step); the page alternates them. The forward foot sits ~55 px higher (farther away), the back
foot shows the sole of its shoe.
  fig_pair_w0 / _w1   dusk/night: scarf on
  fig_umb_w0 / _w1    under your one navy umbrella, held in your left hand and tipped toward me; the bent rib is the
                      right-hand one (its panel sags); beads of rain on the cloth, a drop under every tip, your right
                      shoulder darkened by rain
  fig_furl_w0 / _w1   daylight, no scarf, you carry the furled umbrella by its crook in your right hand
  meta: h_me, h_you (standing heights px; `h` itself is the image height in the manifest), head_me / head_you [dx, dy]
        centre of each head from the anchor, feet [[dx, dy] x4] ground contacts (mine fwd/back, then yours),
        umb (fig_umb only) {apex [dx, dy] the top of the canopy, rim [dx, dy] centre of the rim, r canopy radius px,
        tips [[dx, dy], ...] the visible rib tips, where drips fall from}.
Sitting on a step, from behind and a little above, knees up, arms round the knees; sheet 760x760, anchor (380, 690) =
the seat point (the middle of where the hips meet the step):
  sp_me (scarf, ~540 px), sp_you (jacket on, ~580 px), sp_you_tee (in the grey tee, same geometry as sp_you),
  sp_coat_on (your jacket over my shoulders: lies exactly on sp_me at the same anchor; my hair and scarf stay on top),
  meta shoulders [dx, dy] (middle of the shoulder line), head [dx, dy].
  sp_coat_fold: the jacket in mid-air folded over itself, sheet 600x520, anchor its centre (300, 250); meta fold
  [[dx, dy], [dx, dy]] the two ends of the fold along its top (where a hand would hold it).
"""
import math

import cv2
import numpy as np

from kit import *

HAIR_ME = (0.34, 0.22, 0.16)      # dark brown
HAIR_YOU = (0.13, 0.12, 0.14)
DENIM = (0.62, 0.74, 0.87)        # light-blue jeans
TEE = (0.82, 0.83, 0.84)
TROU = (0.27, 0.28, 0.32)         # dark trousers
SHOE = (0.22, 0.18, 0.17)

AX, AY = 490, 1230                # the pair sheets' anchor
ME_DX, YOU_DX = -146, 154         # where each of us stands, from the anchor
H_ME, H_YOU = 880, 940


# ------------------------------------------------------------------------------------------------ pen and wash habits
def pen(s, m, w=2.3, color=INK, k=1.4, seed=1, again=2, **kw):
    """Outline a mask, then go over a stretch or two again slightly off and a little past the end (a quick hand)."""
    contour(s, m, w, color, k, **kw)
    r = rs(seed)
    cs, _ = cv2.findContours((m > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cs:
        P = c[:, 0, :].astype(np.float64)
        if len(P) < 80:
            continue
        for _ in range(again):
            n = len(P)
            a = int(r.random() * n)
            L = int(n * (0.06 + 0.10 * r.random()))
            Q = np.array([P[(a + i) % n] for i in range(0, L, 4)])
            if len(Q) < 4:
                continue
            d = Q[-1] - Q[-2]
            d = d / (np.hypot(*d) + 1e-6)
            Q = np.vstack([Q, Q[-1] + d * (4 + 8 * r.random())]) + (r.random(2) - 0.5) * 3.0
            s.ink(Q, w * 0.75, color, k * 0.55, taper=(0.3, 0.4), smooth=True)


def cloth(s, m, col, shade, x0, x1, k=0.95, deep=0.5, light=0.18, fs=120, gran=0.45, flow=0.5, y0=None, y1=None, low=None):
    """A garment: first wash, the shadow side wet into it (light from the left), the lit side lifted back."""
    clip = cv2.GaussianBlur(m, (0, 0), 2.5)
    base(s, m, col, k, wob=2, edge=0.45, flow=flow, fs=fs, gran=gran)
    s.wash(m, shade, deep, wob=2, bleed=10, grad=s.hgrad(x0, x1, 0.06, 1), edge=0.2, gran=0.4, clip=clip)
    if y0 is not None:
        s.wash(m, low or shade, deep * 0.6, wob=2, bleed=12, grad=s.vgrad(y0, y1, 0, 1), edge=0.15, clip=clip)
    if light:
        s.lift(m * s.hgrad(x0 - (x1 - x0) * 0.1, x0 + (x1 - x0) * 0.42, 1, 0), light, soft=8, wob=2)


def crease(s, pts, w, color, k=0.3, clip=None):
    """A fold: a soft shadow stroke with a wet edge."""
    s.stroke(pts, w, color, k, wob=2, bleed=max(2, w * 0.18), clip=clip, taper=(0.35, 0.35), edge=0.35)


class Fig:
    """Local coordinates: u from the person's centre line (+ right), v up from the ground (negative), plus a sway of
    the upper body (`lean` px at the head, nothing at the hips)."""

    def __init__(self, s, cx, gy, H, lean=0.0):
        self.s, self.cx, self.gy, self.H, self.lean = s, cx, gy, H, lean

    def p(self, u, v):
        k = min(1.0, max(0.0, (-v - 0.48 * self.H) / (0.52 * self.H)))
        return (self.cx + u + self.lean * k, self.gy + v)

    def P(self, pts):
        return [self.p(u, v) for (u, v) in pts]


# ------------------------------------------------------------------------------------------------ legs and shoes
def halfplane(s, p, d, off):
    """1 up to `off` px past the line through p across direction d, 0 beyond: cuts a ribbon straight."""
    return ((s.X - p[0]) * d[0] + (s.Y - p[1]) * d[1] < off).astype(np.float32)


def unit(a, b):
    d = np.array([b[0] - a[0], b[1] - a[1]], np.float64)
    return d / (np.hypot(*d) + 1e-9)


def shoe(F, u, vb, w, h, raised, white=True, seed=1):
    """A shoe from behind: the heel counter and the sole edge. `raised` (the back foot pushing off): the heel is up and
    the tread of the sole shows below it, down to the toe still on the ground."""
    s = F.s
    x, y = F.p(u, vb)
    tl = h * 1.15 if raised else 0
    # the shadow on the ground
    s.wash(s.ellipse(x + 8, y + tl + 4, w * 0.78, h * 0.2), PAYNE, 0.32, wob=2, bleed=7, edge=0.1)
    heel = s.poly([(x - w * 0.5, y - h * 0.16), (x - w * 0.48, y - h * 0.62), (x - w * 0.34, y - h * 0.92), (x - w * 0.1, y - h * 1.0),
                   (x, y - h * 1.08), (x + w * 0.1, y - h * 1.0), (x + w * 0.34, y - h * 0.92), (x + w * 0.48, y - h * 0.62),
                   (x + w * 0.5, y - h * 0.16), (x + w * 0.42, y + 1), (x - w * 0.42, y + 1)])
    if raised:
        tr = s.poly([(x - w * 0.47, y - 4), (x + w * 0.47, y - 4), (x + w * 0.46, y + tl * 0.5), (x + w * 0.36, y + tl * 0.92), (x + w * 0.1, y + tl),
                     (x - w * 0.1, y + tl), (x - w * 0.36, y + tl * 0.92), (x - w * 0.46, y + tl * 0.5)])
        base(s, tr, MIST if white else UMBER, 0.75 if white else 0.9, wob=1, edge=0.45, gran=0.5)
        s.wash(tr, SLATE if white else SEPIA, 0.45, wob=1, bleed=4, grad=s.hgrad(x - w * 0.3, x + w * 0.5, 0.2, 1))
        hb = tr * s.vgrad(y + 2, y + 6) * (1 - s.vgrad(y + tl * 0.3, y + tl * 0.34))
        s.wash(hb, PAYNE, 0.35, wob=0.5, soft=1)                # the heel block, then the waist of the sole, then the ball
        for i in range(5):        # tread
            yy = y + tl * (0.45 + 0.1 * i)
            s.ink([(x - w * 0.36, yy), (x - w * 0.12, yy + 3), (x + w * 0.12, yy - 1), (x + w * 0.36, yy + 2)], 1.1, PAYNE, 0.8, skip=0.25, clip=tr)
        for i in range(2):
            yy = y + tl * (0.08 + 0.1 * i)
            s.ink([(x - w * 0.36, yy), (x + w * 0.36, yy + 1)], 1.0, PAYNE, 0.6, skip=0.3, clip=tr)
        pen(s, tr, 1.6, INK, 1.2, seed=seed + 3, again=0)
    if white:
        base(s, heel, CREAM, 0.10, wob=1, edge=0.3)
        s.wash(heel, SLATE, 0.32, wob=1.5, bleed=5, grad=s.hgrad(x - w * 0.2, x + w * 0.55, 0.1, 1))
        s.wash(heel, DUSK, 0.20, wob=1, bleed=4, grad=s.vgrad(y - h * 0.6, y, 0.1, 1))
        band = heel * s.vgrad(y - h * 0.24, y - h * 0.20)
        s.wash(band, SLATE, 0.45, wob=1, soft=1.5)
        tab = s.poly([(x - w * 0.11, y - h * 1.06), (x + w * 0.11, y - h * 1.06), (x + w * 0.10, y - h * 0.58), (x - w * 0.10, y - h * 0.58)])
        s.wash(tab, MINT, 0.55, wob=1, edge=0.4)
        pen(s, heel, 1.8, SEPIA, 1.25, seed=seed, again=1)
        s.ink([(x - w * 0.47, y - h * 0.22), (x, y - h * 0.19), (x + w * 0.47, y - h * 0.22)], 1.3, SEPIA, 0.8)
    else:
        base(s, heel, SHOE, 1.05, wob=1, edge=0.5)
        s.wash(heel, INK, 0.35, wob=1, bleed=5, grad=s.hgrad(x - w * 0.2, x + w * 0.5, 0, 1))
        s.lift(heel * s.hgrad(x - w * 0.5, x - w * 0.05, 1, 0) * s.vgrad(y - h, y - h * 0.4, 1, 0.2), 0.28, soft=3, wob=1)
        band = heel * s.vgrad(y - h * 0.22, y - h * 0.18)
        s.wash(band, UMBER, 0.5, wob=1, soft=1)
        pen(s, heel, 1.8, INK, 1.4, seed=seed, again=1)
        s.ink([(x - w * 0.47, y - h * 0.20), (x, y - h * 0.17), (x + w * 0.47, y - h * 0.20)], 1.3, SEPIA, 0.9)
    return (x, y + tl)


def legs(F, f, me, seed):
    """Both legs, the shoes and the seat. f = +1: the left foot forward (frame 0); -1: the right one.
    The forward foot has just landed farther from us, so it sits higher; the back one is pushing off: heel up and the
    sole showing, the knee a little bent. Returns the ground contacts."""
    s = F.s
    sc = F.H / 880
    r = rs(seed)
    if me:
        hipu, hipv, w, sw, sh = 44, -440, 80, 60, 44
    else:
        hipu, hipv, w, sw, sh = 52, -478, 92, 70, 48
    L = []
    for side in (-1, 1):
        fwd = (side == -1) == (f == 1)
        if fwd:
            L.append(dict(fwd=True, side=side, u=side * 24 * sc, vb=-54 * sc, knee=(side * 36 * sc, -264 * sc), hv=hipv))
        else:
            L.append(dict(fwd=False, side=side, u=side * 40 * sc, vb=-2 * sc, knee=(side * 46 * sc, -228 * sc), hv=hipv + 7))
    L.sort(key=lambda o: not o["fwd"])           # the far (forward) leg first
    feet = []
    for o in L:              # shoes under the hems
        g = shoe(F, o["u"], o["vb"], sw * (0.97 if o["fwd"] else 1.03), sh, not o["fwd"], white=me, seed=seed + 11 + o["side"])
        feet.append([round(g[0] - AX, 1), round(g[1] - AY, 1)])
    allm = s.zeros()
    for o in L:
        side = o["side"]
        ank = (o["u"] + side * 1, o["vb"] - sh * 0.86)
        hip = (side * hipu, o["hv"])
        pts = F.P([hip, ((hip[0] + o["knee"][0]) / 2, (hip[1] + o["knee"][1]) / 2), o["knee"],
                   (o["knee"][0] * 0.5 + ank[0] * 0.5, o["knee"][1] * 0.5 + ank[1] * 0.5), ank])
        m = s.ribbon(pts, w * (1.0 if o["fwd"] else 1.03), profile=lambda t: np.interp(t, [0, 0.4, 0.55, 0.72, 1], [1.0, 0.74, 0.66, 0.66, 0.60]))
        d = unit(pts[-2], pts[-1])
        m *= halfplane(s, pts[-1], d, 2.0)             # the hem cut straight
        o["m"], o["pts"] = m, pts
        allm = np.maximum(allm, m)
    seat = s.poly(F.P([(-hipu - 46 * sc, hipv - 40), (hipu + 46 * sc, hipv - 40), (hipu + 50 * sc, hipv + 8), (hipu + 22 * sc, hipv + 40 * sc),
                       (0, hipv + 30 * sc), (-hipu - 22 * sc, hipv + 40 * sc), (-hipu - 50 * sc, hipv + 8)]))
    allm = np.maximum(allm, seat)
    x0, x1 = F.p(-hipu - 56, 0)[0], F.p(hipu + 56, 0)[0]
    gy = F.gy
    if me:
        cloth(s, allm, DENIM, SLATE, x0, x1, k=0.9, deep=0.40, light=0.22, fs=90, y0=gy - 480, y1=gy - 40, low=INDIGO)
        s.wash(allm, PRUSSIAN, 0.08, wob=2, bleed=14, grad=s.vgrad(gy - 300, gy, 0, 1))
        pc, pk, fc, fk = INKBLUE, 1.25, SLATE, 0.32
    else:
        cloth(s, allm, TROU, INK, x0, x1, k=1.0, deep=0.35, light=0.16, fs=90, y0=gy - 500, y1=gy - 40, low=INDIGO)
        pc, pk, fc, fk = INK, 1.3, INK, 0.4
    for o in L:
        m, pts, side = o["m"], o["pts"], o["side"]
        kx, ky = pts[2]
        ax_, ay_ = pts[-1]
        if o["fwd"]:
            s.wash(m, INDIGO, 0.10, wob=2, bleed=6, clip=m)
        # the back of the knee: deeper creases on the bent, near leg
        n = 4 if not o["fwd"] else 2
        for i in range(n):
            yy = ky + (i - n / 2 + 0.5) * 12 * sc + r.random() * 4
            ww = w * 0.30 * (1 - 0.12 * abs(i - n / 2))
            crease(s, [(kx - ww, yy - 3), (kx, yy + 4 + (3 if not o["fwd"] else 0)), (kx + ww, yy - 2)], 7, fc, fk, clip=m)
            s.ink([(kx - ww * 0.8, yy - 2), (kx + side * 3, yy + 4), (kx + ww * 0.7, yy - 1)], 1.3, pc, 0.55, skip=0.35, clip=m)
        for i in range(3):         # the slack stacking at the hem
            yy = ay_ - 12 - i * 15
            s.ink([(ax_ - w * 0.27, yy + 3), (ax_ + side * 4 + r.random() * 4, yy - 3), (ax_ + w * 0.27, yy + 2)], 1.2, pc, 0.55, skip=0.35, clip=m)
            crease(s, [(ax_ - w * 0.25, yy + 4), (ax_, yy - 2), (ax_ + w * 0.25, yy + 3)], 6, fc, fk * 0.8, clip=m)
        # the outer seam and a long fold down the shaded side of the leg
        hx, hy = pts[0][0] + side * w * 0.44, pts[0][1] + 10
        s.ink([(hx, hy), (kx + side * w * 0.35, ky), (ax_ + side * w * 0.30, ay_ - 6)], 1.3, pc, 0.45, skip=0.4, clip=m)
        crease(s, [(pts[1][0] + w * 0.2, pts[1][1]), (kx + w * 0.22, ky - 30), (kx + w * 0.17, ky + 70 * sc)], 16, fc, 0.6 * fk, clip=m)
    for side in (-1, 1):        # the seat under the hem, the inseam
        a = F.p(side * 5, hipv + 20 * sc)
        b = F.p(side * hipu * 0.85, hipv + 36 * sc)
        c = F.p(side * (hipu + 34 * sc), hipv + 24 * sc)
        s.ink([a, b, c], 1.5, pc, 0.75, skip=0.2, clip=allm)
        crease(s, [a, b, c], 10, fc, 0.6, clip=allm)
    pen(s, allm, 2.2, pc, pk, seed=seed + 5)
    return feet


# ------------------------------------------------------------------------------------------------ arms and hands
def hand(F, x, y, d, w, L, side, seed):
    """A hand hanging at the side, seen from behind: the back of the hand and the little-finger edge, the fingers a
    little curled in toward the thigh. (x, y) sheet point of the wrist, d the forearm's direction."""
    s = F.s
    n = np.array([-d[1], d[0]])
    o = -side       # the fingers curl in toward the body

    def q(a, b):     # a across (in units of w), b along (in units of L)
        return (x + n[0] * a * w + d[0] * b * L, y + n[1] * a * w + d[1] * b * L)
    pts = [q(-0.48, -0.05), q(0.48, -0.05), q(0.55, 0.32), q(0.48 + o * 0.12, 0.66), q(0.26 + o * 0.22, 0.93), q(o * 0.28, 1.0),
           q(-0.22 + o * 0.22, 0.94), q(-0.42 + o * 0.12, 0.68), q(-0.52, 0.33)]
    m = s.poly(pts)
    base(s, m, SKIN, 0.62, wob=1.2, edge=0.35, flow=0.3)
    s.wash(m, BURNT, 0.30, wob=1.5, bleed=6, grad=s.hgrad(x - w * 0.5, x + w * 0.6, 0.1, 1) * s.vgrad(y, y + L, 0.4, 1), edge=0.2, clip=cv2.GaussianBlur(m, (0, 0), 2))
    s.wash(m, ROSE, 0.16, wob=1.5, bleed=6, grad=s.vgrad(y + L * 0.5, y + L, 0, 1), clip=cv2.GaussianBlur(m, (0, 0), 2))
    s.lift(m * s.hgrad(x - w * 0.5, x, 1, 0), 0.15, soft=3, wob=1)
    pen(s, m, 1.7, SEPIA, 1.2, seed=seed, again=1)
    for i in range(3):           # the gaps between the curled fingers, the knuckle line
        a = -0.3 + 0.3 * i + o * 0.08
        s.ink([q(a, 0.58), q(a + o * 0.05, 0.76), q(a + o * 0.1, 0.88 + 0.04 * (1 - abs(a)))], 1.1, SEPIA, 0.6, taper=(0.4, 0.2))
    s.ink([q(-0.36, 0.48), q(0, 0.52), q(0.36, 0.48)], 1.1, BURNT, 0.55, skip=0.3)
    return m


def arm(F, sh, el, wr, w, paint, pen_c, cuff, hand_wl, side, seed, under=False):
    """A sleeve from the shoulder seam to a cuff cut straight across, the hand coming out of it."""
    s = F.s
    P = F.P([sh, ((sh[0] + el[0]) / 2 + side * 2, (sh[1] + el[1]) / 2), el, ((el[0] + wr[0]) / 2, (el[1] + wr[1]) / 2), wr])
    d = unit(P[2], P[4])
    hand(F, P[4][0] - d[0] * 6, P[4][1] - d[1] * 6, d, hand_wl[0], hand_wl[1], side, seed + 40)
    sl = s.ribbon(P, w, profile=lambda t: np.interp(t, [0, 0.3, 0.6, 1], [1.0, 0.95, 0.9, 0.84]))
    sl *= halfplane(s, P[4], d, 3.0)
    x0, x1 = P[0][0] - w * 0.6, P[0][0] + w * 0.6
    paint(sl, x0, x1)
    if under:
        s.wash(sl, DUSK, 0.14, wob=1, bleed=6, clip=sl)
    ex_, ey_ = P[2]
    for i in range(3):            # the slack at the elbow
        yy = ey_ - 14 + i * 13
        crease(s, [(ex_ - w * 0.4, yy - 4 + i), (ex_ + side * 3, yy + 5), (ex_ + w * 0.4, yy - 2)], 7, cuff[2], 0.28, clip=sl)
        s.ink([(ex_ - w * 0.3, yy - 2), (ex_ + side * 2, yy + 5), (ex_ + w * 0.26, yy)], 1.2, pen_c, 0.5, skip=0.4, clip=sl)
    crease(s, [(P[0][0] - side * 4, P[0][1] + 30), P[1], (ex_ - side * 6, ey_ - 18)], w * 0.22, cuff[2], 0.22, clip=sl)
    cf = sl * (1 - halfplane(s, P[4], d, -cuff[0]))
    base(s, cf, cuff[1], 0.9, wob=0.8, edge=0.5)
    cuff[3](cf, P[4], d)
    s.ink([(P[4][0] - d[0] * cuff[0] - d[1] * w * 0.42, P[4][1] - d[1] * cuff[0] + d[0] * w * 0.42),
           (P[4][0] - d[0] * cuff[0] + d[1] * w * 0.42, P[4][1] - d[1] * cuff[0] - d[0] * w * 0.42)], 1.3, pen_c, 0.8)
    pen(s, sl, 2.1, pen_c, 1.3, seed=seed + 60)
    return sl, P


# ------------------------------------------------------------------------------------------------ me
def me_knit(s, m, x0, x1, gy=None, H=None):
    clip = cv2.GaussianBlur(m, (0, 0), 2.5)
    base(s, m, CREAM, 0.50, wob=2, edge=0.45, flow=0.15, fs=200, gran=0.5)
    s.wash(m, SOY, 0.30, wob=2, bleed=8, grad=s.hgrad(x0, x1, 1, 0.3), clip=clip)
    s.wash(m, OCHRE, 0.15, wob=2, bleed=10, grad=s.hgrad(x0, x1, 0.1, 1), edge=0.2, gran=0.5, flow=0.1, clip=clip)
    s.wash(m, DUSK, 0.24, wob=2, bleed=10, grad=s.hgrad(x0 + (x1 - x0) * 0.45, x1, 0, 1), edge=0.2, clip=clip)
    if gy is not None:
        s.wash(m, UMBER, 0.08, wob=2, bleed=12, grad=s.vgrad(gy - H * 0.60, gy - H * 0.47, 0, 1), clip=clip)
    s.lift(m * s.hgrad(x0 - (x1 - x0) * 0.1, x0 + (x1 - x0) * 0.4, 1, 0), 0.20, soft=8, wob=2)


def knit_cuff(s):
    def f(cf, p, d):
        s.wash(cf, OCHRE, 0.25, wob=1, soft=2)
        n = np.array([-d[1], d[0]])
        for q in np.linspace(-20, 20, 9):
            a = (p[0] + n[0] * q - d[0] * 22, p[1] + n[1] * q - d[1] * 22)
            b = (p[0] + n[0] * q * 1.05 + d[0] * 2, p[1] + n[1] * q * 1.05 + d[1] * 2)
            s.ink([a, b], 1.0, SIENNA, 0.6, clip=cf)
    return f


def ribs(s, m, xs, ytop, ybot, color=SIENNA, k=0.34, w=1.1, skip=0.45, seed=1, cx=0.0, spread=0.0, bow=5):
    """Knit ribs: soft vertical pen lines that follow the body, broken where the yarn catches the light."""
    r = rs(seed)
    mm = cv2.GaussianBlur(m, (0, 0), 1.2)
    for x in xs:
        x = x + (r.random() - 0.5) * 2.5
        yt = ytop(x)
        dx = (x - cx) * spread
        s.ink([(x, yt), (x + dx * 0.5 + bow * (r.random() - 0.5), (yt + ybot) / 2), (x + dx, ybot)], w, color, k * (0.6 + 0.6 * r.random()), skip=skip, clip=mm, tremor=1.0)


def rot(c, a):
    ca, sa = math.cos(a), math.sin(a)
    return lambda u, v: (c[0] + (u - c[0]) * ca - (v - c[1]) * sa, c[1] + (u - c[0]) * sa + (v - c[1]) * ca)


def me_walk(s, cx, gy, f, scarf=True, lean=0.0, seed=10, tilt=4.0):
    """我, walking away. f = +1 left foot forward. tilt: the head leans this many degrees toward you."""
    F = Fig(s, cx, gy, H_ME, lean)
    r = rs(seed)
    # arm swing: the arm opposite the forward leg swings forward (away from us: under the body, the hand half hidden
    # by the hip); the other swings back toward us. The inner (right) arm swings less: we walk close.
    arms = {}
    for side in (-1, 1):
        back = (side == -1) == (f == 1)
        amp = 0.5 if side == 1 else 1.0
        inner = 7 if side == 1 else 0
        sh = (side * 84, -692)
        if back:
            el = (side * (96 + 5 * amp), -566 + 4 * amp)
            wr = (side * (100 + 14 * amp + inner), -440 + 8 * amp)
        else:
            el = (side * (92 - 4 * amp), -568 + 6 * amp)
            wr = (side * (90 - 10 * amp + inner), -434 + 14 * amp)
        arms[side] = (back, sh, el, wr)
    cuff = (24, CREAM, OCHRE, knit_cuff(s))

    def do_arm(side):
        back, sh, el, wr = arms[side]
        sl, P = arm(F, sh, el, wr, 52, lambda m, a, b: me_knit(s, m, a, b), SEPIA, cuff, (31, 88), side, seed + side, under=not back)
        for q in np.linspace(-0.34, 0.34, 6):       # ribs along the sleeve
            n = np.array([-(P[4][1] - P[0][1]), P[4][0] - P[0][0]])
            n = n / np.hypot(*n) * q * 50
            s.ink([(p[0] + n[0] * (1 - 0.12 * i), p[1] + n[1] * (1 - 0.12 * i)) for i, p in enumerate(P[:4])], 1.0, SIENNA, 0.38, skip=0.5, clip=cv2.GaussianBlur(sl, (0, 0), 1.2))
        return sl

    for side in (-1, 1):
        if not arms[side][0]:
            do_arm(side)
    feet = legs(F, f, True, seed + 100)
    T = [(-28, -762), (-56, -754), (-80, -742), (-96, -724), (-104, -700), (-102, -670), (-95, -630), (-89, -580), (-86, -540), (-89, -495),
         (-95, -452), (-99, -420), (-52, -412), (0, -416), (52, -412), (99, -420), (95, -452), (89, -495), (86, -540), (89, -580),
         (95, -630), (102, -670), (104, -700), (96, -724), (80, -742), (56, -754), (28, -762), (0, -764)]
    tor = s.poly(F.P(T))
    x0, x1 = F.p(-104, -600)[0], F.p(104, -600)[0]
    me_knit(s, tor, x0, x1, gy, H_ME)
    for side in (-1, 1):       # folds: the knit pulls toward the arm that swings back; a soft dip at the small of the back
        back = arms[side][0]
        crease(s, [F.p(side * 64, -650), F.p(side * 38, -566 + (14 if back else 0)), F.p(side * 10, -505)], 22, OCHRE, 0.30 if back else 0.18, clip=tor)
        crease(s, [F.p(side * 92, -470), F.p(side * 58, -472), F.p(side * 28, -488)], 12, SIENNA, 0.16, clip=tor)
        crease(s, [F.p(side * 28, -700), F.p(side * 52, -660), F.p(side * 58, -618)], 24, DUSK, 0.12, clip=tor)      # shoulder blades
    crease(s, [F.p(-30, -540), F.p(0, -530), F.p(30, -540)], 30, OCHRE, 0.15, clip=tor)
    s.lift(tor * s.vgrad(gy - 770, gy - 700, 1, 0) * s.hgrad(x0, x1, 1, 0.3), 0.18, soft=6, wob=2)
    ribs(s, tor, np.arange(x0 + 8, x1 - 4, 9.5), lambda x: gy - 752 + 0.0024 * (x - cx) ** 2 + 8, gy - 452, seed=seed + 7, cx=cx, spread=0.05)
    hb = tor * s.vgrad(gy - 452, gy - 446)          # the ribbed hem band
    s.wash(hb, OCHRE, 0.22, wob=1.5, soft=2, edge=0.4)
    for x in np.arange(x0 - 2, x1 + 2, 5.0):
        s.ink([(x, gy - 448), (x + (x - cx) * 0.02, gy - 413 - 0.0006 * (x - cx) ** 2)], 0.95, SIENNA, 0.5, skip=0.25, clip=hb)
    s.ink(F.P([(-96, -450), (-50, -447), (0, -448), (50, -447), (96, -450)]), 1.3, SEPIA, 0.7, skip=0.2)
    pen(s, tor, 2.3, SEPIA, 1.35, seed=seed + 8)
    for side in (-1, 1):
        if arms[side][0]:
            do_arm(side)
    # the head turns a little toward you: everything above the collar rotates about the neck
    hr = rot((0, -768), math.radians(tilt))
    HQ = lambda pts: F.P([hr(u, v) for (u, v) in pts])
    nk = s.poly(HQ([(-22, -800), (22, -800), (26, -752), (-26, -752)]))
    base(s, nk, SKIN, 0.6, wob=1, edge=0.3)
    s.wash(nk, UMBER, 0.38, wob=1, bleed=5, grad=s.vgrad(gy - 800, gy - 770, 1, 0.2))
    hx, hy = F.p(*hr(0, -818))
    for side in (-1, 1):            # ears, just showing under the hair
        ex_, ey_ = F.p(*hr(side * 48, -808))
        e = s.ellipse(ex_, ey_, 7, 14, side * 10 + tilt)
        base(s, e, SKIN, 0.65, wob=0.8, edge=0.4)
        s.wash(e, ROSE, 0.3, wob=0.8, soft=2)
        pen(s, e, 1.3, SEPIA, 1.0, again=0)
    if scarf:
        sc_ = s.poly(F.P([(-66, -736), (-54, -770), (-30, -784), (0, -788), (30, -784), (54, -770), (66, -736), (40, -722), (0, -726), (-40, -722)]))
        tail = s.ribbon(F.P([(-40, -744), (-62, -700), (-72, -640), (-70, -588)]), 48, profile=lambda t: 1 - 0.1 * t)
        tail *= halfplane(s, F.p(-70, -588), unit(F.p(-72, -640), F.p(-70, -588)), 0)
        scm = np.maximum(sc_, tail)
        sx0, sx1 = F.p(-80, -700)[0], F.p(70, -700)[0]
        cloth(s, scm, MINT, LEAF, sx0, sx1, k=0.8, deep=0.40, light=0.15, fs=40, gran=0.5)
        s.wash(tail, LEAF, 0.20, wob=1.5, bleed=6, grad=s.vgrad(gy - 740, gy - 590, 0, 1), clip=tail)
        s.wash(sc_ * s.vgrad(gy - 760, gy - 724, 0, 1), LEAF, 0.25, wob=1.5, bleed=4, clip=sc_)
        for q in np.linspace(0, 1, 6):     # the knit runs round the neck on the band, and down the hanging end
            s.ink(F.P([(-62, -742 - 34 * q), (0, -752 - 34 * q), (62, -742 - 34 * q)]), 1.0, LEAF, 0.5, skip=0.4, clip=cv2.GaussianBlur(sc_, (0, 0), 1))
        for q in np.linspace(-0.36, 0.36, 6):
            s.ink(F.P([(-40 + q * 44, -738), (-62 + q * 46, -700), (-72 + q * 46, -640), (-70 + q * 46, -594)]), 1.0, LEAF, 0.5, skip=0.4, clip=cv2.GaussianBlur(tail, (0, 0), 1))
        pen(s, sc_, 2.0, INKBLUE, 1.05, seed=seed + 9)
        pen(s, tail, 2.0, INKBLUE, 1.05, seed=seed + 10)
        for i in range(10):      # fringe
            fx, fy = F.p(-92 + i * 4.6, -590 + abs(i - 4.5) * 0.6)
            s.ink([(fx, fy - 2), (fx + (r.random() - 0.5) * 3, fy + 8), (fx + (r.random() - 0.5) * 5, fy + 15 + r.random() * 6)], 1.6, LEAF, 0.9, taper=(0.1, 0.5))
    else:
        cl = s.poly(F.P([(-40, -754), (-24, -764), (0, -767), (24, -764), (40, -754), (28, -744), (0, -742), (-28, -744)]))
        base(s, cl, CREAM, 0.8, wob=1, edge=0.5)
        s.wash(cl, OCHRE, 0.3, wob=1, soft=2)
        for q in np.linspace(-34, 34, 13):
            s.ink(F.P([(q, -765 + abs(q) * 0.25), (q * 1.05, -744)]), 0.9, SIENNA, 0.6, clip=cl)
        pen(s, cl, 1.6, SEPIA, 1.1, again=0)
    # the hair, pulled back loosely to a low tie at the nape, then a short tail onto the back
    HP = [(0, -881), (-30, -875), (-49, -853), (-56, -822), (-54, -796), (-46, -778), (-31, -766), (-14, -760), (0, -758),
          (14, -760), (31, -766), (46, -778), (54, -796), (56, -822), (49, -853), (30, -875)]
    hm = s.poly(HQ(HP))
    tx, ty = F.p(*hr(0, -764))
    sw = -f * 5
    TP = [(-11, -0), (-19, 22), (-23, 54), (-21, 84), (-14 + sw * 0.6, 106), (-6 + sw, 122), (2 + sw, 126),
          (9 + sw, 116), (17 + sw * 0.6, 94), (22, 62), (19, 26), (11, 0)]
    tm = s.poly([(tx + a, ty + b) for (a, b) in TP])
    hair_paint(s, np.maximum(hm, tm), hx, hy, 56, HAIR_ME, seed + 11)
    mh = cv2.GaussianBlur(hm, (0, 0), 1)
    for i in range(34):          # strands from the hairline round the head into the tie
        a = -2.3 + 4.6 * (i + r.random() * 0.7) / 34
        sx_, sy_ = hx + math.sin(a) * 54, hy - math.cos(a) * 62
        if sy_ > ty - 10:
            continue
        mx, my = hx + math.sin(a) * 42, (sy_ + ty) / 2 + 14
        s.ink([(sx_, sy_), (mx, my), (tx + math.sin(a) * 6, ty)], 1.05, INK, 0.5, skip=0.35, clip=mh, tremor=0.8)
    for i in range(13):
        q = (i + r.random() * 0.5) / 13 - 0.5
        s.ink([(tx + q * 16, ty), (tx + q * 38 + sw * 0.5, ty + 60), (tx + q * 14 + sw, ty + 118 - abs(q) * 18)], 1.0, INK, 0.5, skip=0.3, clip=cv2.GaussianBlur(tm, (0, 0), 1), tremor=0.9)
    pen(s, hm, 2.0, INK, 1.3, seed=seed + 12, again=1)
    pen(s, tm, 1.9, INK, 1.25, seed=seed + 13, again=1)
    tie = s.ellipse(tx, ty + 1, 13, 7, tilt)
    base(s, tie, UMBER, 1.0, wob=0.6, edge=0.4)
    pen(s, tie, 1.5, INK, 1.2, again=0)
    # loose strands: behind the ears, at the nape, off the crown, at the tip of the tail
    for pts, w in (([(-47, -800), (-53, -780), (-49, -758), (-55, -742)], 1.1), ([(45, -797), (51, -777), (47, -760)], 1.0),
                   ([(-15, -763), (-23, -749), (-19, -737)], 0.9), ([(8, -763), (14, -752), (10, -742), (16, -734)], 0.9),
                   ([(-20, -877), (-30, -891), (-44, -887)], 0.9), ([(14, -879), (24, -894)], 0.8)):
        s.ink(HQ(pts), w, INK, 0.9, taper=(0.15, 0.6), tremor=0.9)
    for pts in (([(-12 + sw, 110), (-18 + sw, 126), (-14 + sw, 134)]), ([(10 + sw, 106), (16 + sw, 124)])):
        s.ink([(tx + a, ty + b) for a, b in pts], 0.9, INK, 0.9, taper=(0.15, 0.6))
    return (hx, hy), feet


def hair_paint(s, m, hx, hy, rad, col, seed):
    """Dark hair: one rich wash, cooler and darker on the shadow side and at the bottom, a curved sheen lifted where
    the light from the upper left rolls over the crown."""
    clip = cv2.GaussianBlur(m, (0, 0), 2)
    base(s, m, col, 1.2, wob=1.6, edge=0.5, flow=0.5, fs=rad * 0.8, gran=0.45)
    s.wash(m, INKBLUE, 0.35, wob=1.5, bleed=rad * 0.12, grad=s.hgrad(hx - rad * 0.2, hx + rad * 1.2, 0, 1), edge=0.2, clip=clip)
    s.wash(m, SEPIA, 0.35, wob=1.5, bleed=rad * 0.1, grad=s.vgrad(hy, hy + rad * 1.4, 0, 1), clip=clip)
    sheen = s.poly([(hx - rad * 0.75, hy - rad * 0.55), (hx - rad * 0.35, hy - rad * 0.95), (hx + rad * 0.15, hy - rad * 1.04), (hx + rad * 0.05, hy - rad * 0.88),
                    (hx - rad * 0.30, hy - rad * 0.76), (hx - rad * 0.62, hy - rad * 0.36)])
    s.lift(sheen * m, 0.42, soft=rad * 0.06, wob=2.5)
    s.wash(sheen * m, SIENNA, 0.25, wob=2, bleed=rad * 0.05, clip=clip)


# ------------------------------------------------------------------------------------------------ you
def navy(s, m, x0, x1, gy=None, H=None, k=1.0, deep=0.5):
    clip = cv2.GaussianBlur(m, (0, 0), 2.5)
    base(s, m, NAVY, k, wob=2, edge=0.45, flow=0.5, fs=90, gran=0.5)
    s.wash(m, INDIGO, deep, wob=2, bleed=10, grad=s.hgrad(x0, x1, 0.06, 1), edge=0.2, gran=0.4, clip=clip)
    if gy is not None:
        s.wash(m, PAYNE, deep * 0.5, wob=2, bleed=12, grad=s.vgrad(gy - H * 0.62, gy - H * 0.46, 0, 1), edge=0.15, clip=clip)
    s.wash(m, PAYNE, 0.25, wob=2, bleed=10, grad=s.hgrad(x0 + (x1 - x0) * 0.55, x1, 0, 1), edge=0.2, clip=clip)
    s.lift(m * s.hgrad(x0 - (x1 - x0) * 0.1, x0 + (x1 - x0) * 0.42, 1, 0), 0.22, soft=8, wob=2)


def navy_cuff(s):
    def f(cf, p, d):
        s.wash(cf, PAYNE, 0.35, wob=1, soft=2)
    return f


def you_walk(s, cx, gy, f, mode="free", lean=0.0, seed=30, wet=False, tilt=-2.0):
    """你, walking away. mode: "free" (arms swinging), "umb" (the left hand up in front holding the umbrella's shaft),
    "furl" (the right hand carrying the furled umbrella). Returns the head centre, the feet, the umbrella hand."""
    F = Fig(s, cx, gy, H_YOU, lean)
    r = rs(seed)
    arms = {}
    for side in (-1, 1):
        back = (side == -1) == (f == 1)
        amp = 0.5 if side == -1 else 1.0
        inner = -7 if side == -1 else 0
        if mode == "furl" and side == 1:
            amp = 0.3
        sh = (side * 104, -738)
        if back:
            el = (side * (114 + 5 * amp), -606 + 4 * amp)
            wr = (side * (118 + 14 * amp) + inner, -470 + 8 * amp)
        else:
            el = (side * (110 - 4 * amp), -608 + 6 * amp)
            wr = (side * (108 - 10 * amp) + inner, -464 + 14 * amp)
        arms[side] = (back, sh, el, wr)
    cuff = (22, NAVY, INKBLUE, navy_cuff(s))
    hand_pt = None
    wrists = {}

    def do_arm(side):
        back, sh, el, wr = arms[side]
        sl, P = arm(F, sh, el, wr, 64, lambda m, a, b: navy(s, m, a, b), INK, cuff, (35, 98), side, seed + side, under=not back)
        wrists[side] = P[4]
        return sl

    def raised_arm():
        """The left arm up: the upper arm down and out to the bent elbow; the forearm goes forward to the shaft,
        hidden by the body; the hand just shows past the shoulder."""
        sh, el = (-104, -742), (-140, -650)
        up = s.ribbon(F.P([sh, (-128, -700), el]), 68, profile=lambda t: np.interp(t, [0, 0.6, 1], [1.0, 0.98, 0.92]))
        fo = s.ribbon(F.P([el, (-128, -686), (-112, -712)]), 58)
        m = np.maximum(up, fo)
        navy(s, m, F.p(-176, 0)[0], F.p(-90, 0)[0])
        ex_, ey_ = F.p(*el)
        for i in range(4):          # the sleeve bunches at the bent elbow
            a0 = -0.5 + i * 0.32
            crease(s, ell(ex_ + 10, ey_ - 16, 26 - i * 3, 22 - i * 2, 1.3 + a0, 2.7 + a0, 10), 8, INKBLUE, 0.3, clip=m)
            s.ink(ell(ex_ + 10, ey_ - 16, 26 - i * 3, 22 - i * 2, 1.3 + a0, 2.7 + a0, 14), 1.2, INK, 0.55, skip=0.3, clip=m)
        pen(s, m, 2.2, INK, 1.3, seed=seed + 70)
        return F.p(-112, -748)

    for side in (-1, 1):
        if mode == "umb" and side == -1:
            continue
        if not arms[side][0]:
            do_arm(side)
    feet = legs(F, f, False, seed + 100)
    tee = s.poly(F.P([(-106, -462), (106, -462), (104, -434), (0, -430), (-104, -434)]))   # the tee under the jacket's hem
    base(s, tee, TEE, 0.7, wob=1, edge=0.4)
    s.wash(tee, SLATE, 0.3, wob=1, soft=2, grad=s.hgrad(F.p(-106, 0)[0], F.p(106, 0)[0], 0.2, 1))
    pen(s, tee, 1.6, SEPIA, 0.9, again=0)
    T = [(-36, -810), (-68, -802), (-98, -789), (-116, -770), (-124, -744), (-122, -708), (-114, -660), (-108, -600), (-105, -540),
         (-107, -490), (-110, -452), (-58, -442), (0, -444), (58, -442), (110, -452), (107, -490), (105, -540), (108, -600),
         (114, -660), (122, -708), (124, -744), (116, -770), (98, -789), (68, -802), (36, -810), (0, -812)]
    tor = s.poly(F.P(T))
    x0, x1 = F.p(-124, -600)[0], F.p(124, -600)[0]
    navy(s, tor, x0, x1, gy, H_YOU)
    yk = F.P([(-122, -722), (-60, -712), (0, -710), (60, -712), (122, -722)])     # yoke seam, back seams, gathered hem band
    s.ink(yk, 1.5, INK, 0.75, skip=0.15, clip=tor)
    crease(s, yk, 10, INDIGO, 0.3, clip=tor)
    for side in (-1, 1):
        s.ink(F.P([(side * 58, -712), (side * 52, -600), (side * 56, -480)]), 1.3, INK, 0.5, skip=0.35, clip=tor)
        crease(s, F.P([(side * 66, -700), (side * 60, -600), (side * 64, -480)]), 14, INKBLUE, 0.16, clip=tor)
    s.lift(tor * s.vgrad(gy - 812, gy - 722, 1, 0) * s.hgrad(x0, x1, 1, 0.2), 0.20, soft=6, wob=2)
    hb = tor * s.vgrad(gy - 476, gy - 470)
    s.wash(hb, INDIGO, 0.35, wob=1.5, soft=2, edge=0.4)
    s.ink(F.P([(-108, -474), (-50, -470), (0, -472), (50, -470), (108, -474)]), 1.5, INK, 0.8, skip=0.15)
    for i in range(9):      # puckers where the band gathers the cloth
        u = -96 + i * 24 + r.random() * 6
        crease(s, F.P([(u, -474), (u + 4, -498), (u + 10, -528 - r.random() * 20)]), 8, INKBLUE, 0.3, clip=tor)
        s.ink(F.P([(u, -475), (u + 3, -492)]), 1.1, INK, 0.6, clip=tor)
    for side in (-1, 1):
        back = arms[side][0]
        crease(s, [F.p(side * 82, -680), F.p(side * 46, -600 + (16 if back else 0)), F.p(side * 16, -540)], 22, INDIGO, 0.32 if back else 0.2, clip=tor)
    if wet:     # the right shoulder is out of the umbrella: the cloth darkens in patches
        wm = tor * s.hgrad(F.p(20, 0)[0], F.p(124, 0)[0], 0, 1) * s.vgrad(gy - 830, gy - 560, 1, 0)
        s.wash(wm, INDIGO, 0.5, wob=3, bleed=14, flow=0.9, fs=26, edge=0.5, clip=tor)
    pen(s, tor, 2.4, INK, 1.4, seed=seed + 8)
    cl = s.poly(F.P([(-46, -814), (-30, -826), (0, -830), (30, -826), (46, -814), (60, -794), (40, -786), (0, -790), (-40, -786), (-60, -794)]))
    navy(s, cl, F.p(-60, 0)[0], F.p(60, 0)[0], k=1.1, deep=0.4)
    s.wash(cl * s.vgrad(gy - 832, gy - 800, 0, 1), INKBLUE, 0.3, wob=1, soft=2)
    pen(s, cl, 1.8, INK, 1.3, again=1)
    s.ink(F.P([(-44, -812), (0, -816), (44, -812)]), 1.2, INK, 0.7, skip=0.2)
    for side in (-1, 1):
        if mode == "umb" and side == -1:
            hand_pt = raised_arm()
            continue
        if arms[side][0]:
            do_arm(side)
    hr = rot((0, -812), math.radians(tilt))
    HQ = lambda pts: F.P([hr(u, v) for (u, v) in pts])
    nk = s.poly(HQ([(-30, -854), (30, -854), (34, -810), (-34, -810)]))
    base(s, nk, SKIN, 0.6, wob=1, edge=0.3)
    s.wash(nk, BURNT, 0.3, wob=1, bleed=5, grad=s.hgrad(F.p(-30, 0)[0], F.p(34, 0)[0], 0, 1))
    s.wash(nk, UMBER, 0.4, wob=1, bleed=5, grad=s.vgrad(gy - 852, gy - 822, 1, 0.1))
    s.ink(HQ([(-5, -840), (-2, -824), (-4, -816)]), 1.2, BURNT, 0.5)       # the hollow of the nape
    hx, hy = F.p(*hr(0, -880))
    for side in (-1, 1):
        e = s.poly(HQ([(side * 46, -896), (side * 58, -902), (side * 62, -884), (side * 58, -862), (side * 48, -858)]))
        base(s, e, SKIN, 0.66, wob=0.8, edge=0.4)
        s.wash(e, ROSE, 0.32, wob=0.8, soft=2)
        pen(s, e, 1.5, SEPIA, 1.1, again=0)
        s.ink(HQ([(side * 52, -892), (side * 57, -880), (side * 52, -868)]), 1.1, BURNT, 0.6)
    HP = [(0, -941), (-32, -935), (-50, -914), (-55, -884), (-52, -856), (-44, -842), (-30, -836), (-14, -840), (0, -836),
          (14, -840), (30, -835), (44, -842), (52, -856), (55, -884), (50, -914), (32, -935)]
    hm = s.poly(HQ(HP))
    hair_paint(s, hm, hx, hy, 55, HAIR_YOU, seed + 11)
    crx, cry = F.p(*hr(6, -918))
    mm = cv2.GaussianBlur(hm, (0, 0), 1)
    for i in range(50):         # short hair: tufts that lie down and round from the crown; the nape cut close
        a = r.random() * 6.28
        d0 = 8 + r.random() * 32
        x_, y_ = crx + math.cos(a) * d0, cry + math.sin(a) * d0 * 0.9
        L = 12 + r.random() * 12
        dd = np.array([math.cos(a) * 0.5, abs(math.sin(a)) * 0.4 + 0.6])
        s.ink([(x_, y_), (x_ + dd[0] * L * 0.6, y_ + dd[1] * L * 0.5), (x_ + dd[0] * L, y_ + dd[1] * L)], 1.1, INK, 0.55, taper=(0.2, 0.6), clip=mm)
    for i in range(16):
        u = -40 + i * 5.3 + r.random() * 2
        a = F.p(*hr(u, -850 + abs(u) * 0.1))
        s.ink([a, (a[0] + (r.random() - 0.5) * 2, a[1] + 8 + r.random() * 4)], 1.0, INK, 0.55, taper=(0.1, 0.7))
    pen(s, hm, 2.0, INK, 1.35, seed=seed + 12, again=1)
    return (hx, hy), feet, hand_pt, wrists


# ------------------------------------------------------------------------------------------------ the umbrella
class Canopy:
    """A navy umbrella seen from behind and a little below its rim. Built in 3-D round the shaft (x right, d toward
    the camera, z up), projected (y = -e*d - z) and rolled by `tilt`. Panels whose outside faces us are drawn after
    the figures; the far ones, which show their underside below the near rim, before."""

    def __init__(self, cx, cy, R, tilt, bent=0, seed=5, n=8, th0=0.24):
        self.cx, self.cy, self.R, self.tilt, self.bent, self.n, self.th0 = cx, cy, R, tilt, bent, n, th0
        self.e, self.Hd = 0.14, R * 0.44
        self.r = rs(seed)
        self.th = [th0 + i * 2 * math.pi / n for i in range(n)]

    def z(self, r):
        return self.Hd * (1 - min(1.0, r / self.R) ** 2) ** 0.8

    def rib3(self, i, r):
        th = self.th[i % self.n]
        z = self.z(r)
        rr = r
        if i % self.n == self.bent and r > 0.5 * self.R:          # the bent rib: it kinks down past its middle
            q = (r - 0.5 * self.R) / (0.5 * self.R)
            z -= q * self.R * 0.13 + q * q * self.R * 0.05
            rr = r * (1 + 0.05 * q)
            th = th + 0.10 * q
        return np.array([rr * math.cos(th), rr * math.sin(th), z])

    def proj(self, X):
        x, y = X[0], -self.e * X[1] - X[2]
        c, s_ = math.cos(self.tilt), math.sin(self.tilt)
        return (self.cx + x * c - y * s_, self.cy + x * s_ + y * c)

    def edge(self, i, n=16):
        """The hem between tip i and tip i+1: it curves in and up between the tips; sags on the bent panel."""
        A, B = self.rib3(i, self.R), self.rib3(i + 1, self.R)
        out = []
        for q in np.linspace(0, 1, n):
            th = self.th[i % self.n] + q * 2 * math.pi / self.n
            ra = (np.hypot(A[0], A[1]) * (1 - q) + np.hypot(B[0], B[1]) * q) * (1 - 0.07 * math.sin(math.pi * q))
            z = A[2] * (1 - q) + B[2] * q + self.R * 0.035 * math.sin(math.pi * q)
            if i % self.n == self.bent or (i + 1) % self.n == self.bent:
                z -= self.R * 0.07 * math.sin(math.pi * q)
            out.append(np.array([ra * math.cos(th), ra * math.sin(th), z]))
        return out

    def panel(self, i):
        rib_a = [self.rib3(i, r) for r in np.linspace(0, self.R, 12)]
        rib_b = [self.rib3(i + 1, r) for r in np.linspace(self.R, 0, 12)]
        P = rib_a + self.edge(i)[1:-1] + rib_b
        return [self.proj(X) for X in P]

    def near(self, i):
        return math.sin(self.th[i] + math.pi / self.n) > -0.05

    def tip(self, i):
        return self.proj(self.rib3(i, self.R))

    def apex(self):
        return self.proj(np.array([0, 0, self.Hd]))

    def under(self, s):
        """The far panels, seen from inside: dark, with the ribs pale against them and the shaft rising to the top."""
        ins = s.zeros()
        for i in range(self.n):
            if not self.near(i):
                ins = np.maximum(ins, s.poly(self.panel(i), smooth=False))
        if ins.max() == 0:
            return ins
        base(s, ins, NAVY, 1.15, wob=1.5, edge=0.4, flow=0.4, fs=60)
        s.wash(ins, PAYNE, 0.55, wob=1.5, bleed=8)
        top = self.proj(np.array([0, 0, self.Hd - 14]))
        for i in range(self.n):
            if not self.near(i) or not self.near(i - 1):
                t = self.tip(i)
                s.ink([t, ((t[0] + top[0]) / 2, (t[1] + top[1]) / 2 + 4), top], 1.6, SLATE, 0.6, clip=cv2.GaussianBlur(ins, (0, 0), 1))
        pen(s, ins, 2.0, INK, 1.3, again=1)
        return ins

    def outer(self, s, wet=True):
        r = self.r
        L = np.array([-0.75, 0.55])         # light from the upper left, a little toward us
        ms = []
        for i in range(self.n):
            if self.near(i):
                ms.append((i, s.poly(self.panel(i), smooth=False)))
        ms.sort(key=lambda o: math.sin(self.th[o[0]] + math.pi / self.n))      # farther panels first
        allm = s.zeros()
        for i, m in ms:
            allm = np.maximum(allm, m)
            mid = self.th[i] + math.pi / self.n
            lit = 0.5 + 0.5 * (math.cos(mid) * L[0] + math.sin(mid) * L[1])
            base(s, m, NAVY, 0.95 + 0.25 * (1 - lit), wob=1.2, edge=0.55, flow=0.5, fs=70, gran=0.5)
            s.wash(m, INDIGO, 0.25 + 0.4 * (1 - lit), wob=1.5, bleed=10, edge=0.2)
            # the cloth stretched between two ribs: lighter down the middle of the panel, darker along the ribs
            a, b = self.tip(i), self.tip(i + 1)
            ap = self.apex()
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            sheen = s.ribbon([(ap[0] * 0.8 + mx * 0.2, ap[1] * 0.8 + my * 0.2), (ap[0] * 0.4 + mx * 0.6, ap[1] * 0.4 + my * 0.6), (mx, my)], 30 + 40 * lit, profile=lambda t: 0.3 + 0.7 * t)
            s.lift(sheen * m, 0.12 + 0.35 * lit, soft=6, wob=2)
            if i == self.bent or (i + 1) % self.n == self.bent:
                s.wash(m * s.rgrad(*self.tip(self.bent), 10, self.R * 0.55, 1, 0), PAYNE, 0.55, wob=2, bleed=10, clip=m)      # the sag holds a shadow
                # the cloth puckers from the kink out to the hem
                kk = self.proj(self.rib3(self.bent, self.R * 0.5))
                for e in self.edge(i, 7)[2:-2:2]:
                    pe = self.proj(e)
                    crease(s, [kk, ((kk[0] + pe[0]) / 2 + 3, (kk[1] + pe[1]) / 2 + 4), pe], 7, INK, 0.35, clip=m)
                    s.ink([kk, ((kk[0] + pe[0]) / 2 + 3, (kk[1] + pe[1]) / 2 + 4), pe], 1.1, INK, 0.55, skip=0.4, clip=cv2.GaussianBlur(m, (0, 0), 1))
        # ribs, hem, tips
        mm = cv2.GaussianBlur(allm, (0, 0), 1.0)
        for i in range(self.n):
            if not (self.near(i) or self.near(i - 1)):
                continue
            P = [self.proj(self.rib3(i, rr)) for rr in np.linspace(0, self.R, 10)]
            s.ink(P, 2.0, INK, 1.2, clip=mm)
            s.stroke(P, 9, PAYNE, 0.35, wob=1, bleed=3, clip=allm, taper=(0.3, 0.1), edge=0.3)
        pen(s, allm, 2.3, INK, 1.4, again=2)
        for i in range(self.n):
            if not (self.near(i) or self.near(i - 1)):
                continue
            t = self.tip(i)
            q = self.proj(self.rib3(i, self.R * 0.97))
            d = np.array(t) - np.array(q)
            d = d / (np.hypot(*d) + 1e-6)
            s.ink([t, (t[0] + d[0] * 9, t[1] + d[1] * 9)], 3.2, INK, 1.6, taper=(0, 0.3))
        # ferrule and top cap
        ap = self.apex()
        up = np.array([math.sin(self.tilt), -math.cos(self.tilt)])
        cap = s.ellipse(ap[0], ap[1] + 2, 16, 6, math.degrees(self.tilt))
        base(s, cap, NAVY, 1.2, wob=0.5, edge=0.5)
        pen(s, cap, 1.6, INK, 1.3, again=0)
        s.ink([ap, (ap[0] + up[0] * 34, ap[1] + up[1] * 34)], 4.0, INK, 1.6, taper=(0, 0.5))
        if wet:
            # rain on the cloth: beads (a dark rim, a lifted spark) and a few runs down the panels
            ys, xs = np.where(allm > 0.9)
            for _ in range(90):
                j = int(r.random() * len(xs))
                p = (xs[j] + r.random(), ys[j] + r.random())
                rr = 1.2 + 2.8 * r.random() ** 2
                s.wash(s.ellipse(p[0], p[1], rr, rr * 1.15), PAYNE, 0.6, wob=0.3, soft=0.6, edge=0.9, ew=1.2, gran=0, flow=0)
                s.lift(s.ellipse(p[0] - rr * 0.35, p[1] - rr * 0.4, rr * 0.45, rr * 0.45), 0.8, soft=0.5, wob=0)
            for _ in range(10):
                i = ms[int(r.random() * len(ms))][0]
                a, b, ap = np.array(self.tip(i)), np.array(self.tip(i + 1)), np.array(self.apex())
                u = 0.3 + 0.4 * r.random()
                p0 = ap + (a * (1 - u) + b * u - ap) * (0.3 + 0.3 * r.random())
                p1 = ap + (a * (1 - u) + b * u - ap) * 0.97
                _lift_line(s, [p0, (p0 + p1) / 2 + (r.random() - 0.5) * 6, p1], 3.2, 0.45, allm)
        return allm

    def drips(self, s, n_fall=3):
        """A drop hanging under every visible tip, and a few falling."""
        r = self.r
        out = []
        for i in range(self.n):
            if not (self.near(i) or self.near(i - 1)):
                continue
            t = self.tip(i)
            out.append(t)
            dx, dy = t[0] + 2, t[1] + 12
            drop = s.poly([(dx, dy - 8), (dx + 4.5, dy + 3), (dx + 3, dy + 8), (dx, dy + 9), (dx - 3, dy + 8), (dx - 4.5, dy + 3)])
            s.wash(drop, RAIN, 0.8, wob=0.3, soft=0.6, edge=0.9, ew=1.5, gran=0)
            s.lift(s.ellipse(dx - 1.2, dy + 4, 1.5, 2), 0.9, soft=0.4, wob=0)
            s.ink([(dx - 4.5, dy + 3), (dx - 2, dy + 8.5), (dx + 3, dy + 8)], 1.0, PAYNE, 0.9)
            for k in range(int(r.integers(0, n_fall + 1))):
                fy = dy + 30 + r.random() * 140
                if s.body[int(min(s.h - 1, fy + 20)), int(dx)] > 0.3 or s.body[int(fy), int(dx)] > 0.3:
                    continue
                L = 10 + r.random() * 10
                fd = s.ribbon([(dx + r.random() * 3, fy), (dx + r.random() * 3, fy + L)], 4.0, profile=lambda q: 0.35 + 0.65 * q)
                s.wash(fd, RAIN, 0.75, wob=0.2, soft=0.5, edge=0.8, ew=1.0, gran=0)
                s.lift(s.ellipse(dx + 1, fy + L * 0.75, 1.0, 1.6), 0.8, soft=0.3, wob=0)
        return out


def _lift_line(s, pts, w, amt, clip):
    m = s.ribbon(pts, w, taper=(0.3, 0.3)) * clip
    s.lift(m, amt, soft=0.8, wob=0.6)


def shaft(s, a, b, w=7):
    m = s.ribbon([a, b], w)
    base(s, m, PAYNE, 1.1, wob=0.4, edge=0.4)
    s.lift(s.ribbon([(a[0] - 1.5, a[1]), (b[0] - 1.5, b[1])], 1.8) * m, 0.5, soft=0.5, wob=0)
    pen(s, m, 1.3, INK, 1.2, again=0)
    return m


# ------------------------------------------------------------------------------------------------ the pair
# ------------------------------------------------------------------------------------------------ on the step
SW, SH, SX, SY = 760, 760, 380, 690      # the sitting sheets and their anchor (the seat point)
K_ME, K_YOU = 1.28, 1.36                  # drawn in the walking figures' units, scaled up: me ~540 px tall, you ~580
WF_YOU = 1.08                             # and you are broader


class SFig(Fig):
    """A sitting figure: (u, v) from the seat point in walking-figure units, scaled by k (and wf across)."""

    def __init__(self, s, k, wf=1.0):
        Fig.__init__(self, s, SX, SY, 880, 0.0)
        self.k, self.wf = k, wf

    def p(self, u, v):
        return (self.cx + u * self.k * self.wf, self.gy + v * self.k)


# the shapes, shared by the figure and by the jacket that is laid over it (me)
BACK = [(-24, -308), (-54, -303), (-80, -291), (-98, -271), (-107, -244), (-109, -210), (-106, -170), (-100, -128), (-95, -88),
        (-94, -50), (-96, -18), (-48, -12), (0, -14), (48, -12), (96, -18), (94, -50), (95, -88), (100, -128), (106, -170),
        (109, -210), (107, -244), (98, -271), (80, -291), (54, -303), (24, -308), (0, -310)]
SCARF_ME = [(-52, -292), (-44, -318), (-24, -330), (0, -332), (24, -330), (44, -318), (52, -292), (32, -282), (0, -285), (-32, -282)]
HAIR_SIT = [(0, -402), (-28, -397), (-45, -380), (-51, -354), (-48, -330), (-40, -316), (-27, -308), (-12, -304), (0, -303),
            (12, -304), (27, -308), (40, -316), (48, -330), (51, -354), (45, -380), (28, -397)]
TIE_SIT = (0, -307)
TAIL_SIT = [(-9, -308), (-16, -290), (-19, -264), (-17, -242), (-11, -226), (-4, -216), (3, -214), (9, -221), (15, -238),
            (19, -262), (16, -290), (9, -308)]
TILT_SIT = 6.0


def sit_head_rot():
    return rot((0, -312), math.radians(TILT_SIT))


def seat(F, me, seed):
    """The hips on the step, and the outsides of the thighs going forward and up to the knees."""
    s = F.s
    hips = s.poly(F.P([(-92, -54), (-100, -26), (-94, -3), (-48, 4), (0, 2), (48, 4), (94, -3), (100, -26), (92, -54)]))
    th = [s.ribbon(F.P([(side * 92, -18), (side * 104, -66), (side * 100, -112)]), 40 * F.k, profile=lambda t: 1 - 0.3 * t) for side in (-1, 1)]
    m = np.maximum(hips, np.maximum(*th))
    x0, x1 = F.p(-104, 0)[0], F.p(104, 0)[0]
    s.wash(s.ellipse(*F.p(6, 5), 118 * F.k * F.wf, 12 * F.k), PAYNE, 0.5, wob=2, bleed=8, edge=0.1)      # the shadow on the step
    if me:
        cloth(s, m, DENIM, SLATE, x0, x1, k=0.9, deep=0.4, light=0.2, fs=60)
        pc = INKBLUE
    else:
        cloth(s, m, TROU, INK, x0, x1, k=1.0, deep=0.35, light=0.14, fs=60)
        pc = INK
    s.wash(m, INDIGO, 0.25, wob=2, bleed=8, grad=s.vgrad(F.p(0, -110)[1], F.p(0, -30)[1], 1, 0), clip=m)     # the thighs turn away into shade
    for side in (-1, 1):
        a, b, c = F.p(side * 3, -24), F.p(side * 50, -6), F.p(side * 94, -14)
        s.ink([a, b, c], 1.5, pc, 0.7, skip=0.2, clip=m)
        crease(s, [a, b, c], 9, SLATE if me else INK, 0.3, clip=m)
        for i in range(2):
            y = -40 - i * 22
            s.ink([F.p(side * 88, y), F.p(side * 98, y - 6), F.p(side * 108, y - 2)], 1.2, pc, 0.55, skip=0.3, clip=m)
    pen(s, m, 2.1, pc, 1.25, seed=seed)
    return m


ARM_SH, ARM_EL, ARM_IN = (82, -262), (128, -150), [(128, -150), (116, -122), (88, -110)]


def sit_arm_masks(F, side, w):
    s = F.s
    sh, el = (side * ARM_SH[0], ARM_SH[1]), (side * ARM_EL[0], ARM_EL[1])
    up = s.ribbon(F.P([sh, ((sh[0] + el[0]) / 2 + side * 6, (sh[1] + el[1]) / 2 - 4), el]), w * F.k, profile=lambda t: 1 - 0.1 * t)
    fo = s.ribbon(F.P([(side * u, v) for u, v in ARM_IN]), w * 0.86 * F.k, profile=lambda t: 1 - 0.1 * t)
    return up, fo, sh, el


def sit_arms(F, me, seed, sleeve_paint, w, part, tee=False):
    """The arms round the knees: the upper arm down and out to the elbow (part "up", over the back); the forearm
    forward and in toward the knees (part "fo", under the back)."""
    s = F.s
    for side in (-1, 1):
        up, fo, sh, el = sit_arm_masks(F, side, w)
        pc = SEPIA if (me or tee) else INK
        if part == "fo":
            if tee:
                base(s, fo, SKIN, 0.62, wob=1, edge=0.35)
                s.wash(fo, BURNT, 0.4, wob=1, bleed=5, clip=fo)
            else:
                sleeve_paint(fo, *sorted((F.p(side * 80, 0)[0], F.p(side * 150, 0)[0])))
                s.wash(fo, PAYNE, 0.15, wob=1, bleed=4, clip=fo)
            pen(s, fo, 2.0, pc, 1.2, seed=seed + side, again=0)
            continue
        xa, xb = sorted((F.p(side * 60, 0)[0], F.p(side * 150, 0)[0]))
        d = unit(F.p(*sh), F.p(*el))
        if tee:
            cut = F.p(side * 100, -214)
            sk = up * (1 - halfplane(s, cut, d, 0))
            base(s, sk, SKIN, 0.62, wob=1, edge=0.35)
            s.wash(sk, BURNT, 0.32, wob=1, bleed=5, grad=s.hgrad(xa, xb, 0.2, 1) if side > 0 else s.hgrad(xa, xb, 1, 0.2), clip=sk)
            s.wash(sk, ROSE, 0.12, wob=1, bleed=5, clip=sk)
            pen(s, sk, 1.8, SEPIA, 1.2, seed=seed + 5 + side, again=1)
            ex_, ey_ = F.p(*el)
            s.ink([(ex_ - side * 10, ey_ - 22), (ex_ + side * 4, ey_ - 6), (ex_ - side * 4, ey_ + 8)], 1.2, BURNT, 0.6)
            sl = up * halfplane(s, cut, d, 0)
            base(s, sl, TEE, 0.75, wob=1, edge=0.45, flow=0.4)
            s.wash(sl, SLATE, 0.35, wob=1, bleed=5, clip=sl)
            pen(s, sl, 1.9, SEPIA, 1.15, seed=seed + 7 + side, again=0)
            for i in range(2):
                s.ink([F.p(side * (78 + i * 8), -250 + i * 12), F.p(side * (94 + i * 4), -236 + i * 8), F.p(side * (102 + i * 3), -226 + i * 6)], 1.1, SEPIA, 0.55, clip=sl)
            continue
        sleeve_paint(up, xa, xb)
        ex_, ey_ = F.p(*el)
        for i in range(4):     # the sleeve folds into the crook of the arm
            q = i / 3
            a = F.p(side * (104 + 8 * q), -200 + 34 * q)
            b = F.p(side * (116 + 6 * q), -194 + 34 * q)
            c = F.p(side * (130 + 2 * q), -200 + 36 * q)
            crease(s, [a, b, c], 8, SIENNA if me else INKBLUE, 0.3, clip=up)
            s.ink([a, b, c], 1.1, pc, 0.5, skip=0.35, clip=up)
        if me:
            P0 = np.array(F.P([sh, el]))
            n = np.array([-(P0[1][1] - P0[0][1]), P0[1][0] - P0[0][0]])
            n = n / np.hypot(*n)
            for q in np.linspace(-0.32, 0.32, 5):
                s.ink(P0 + n * q * w * F.k, 1.0, SIENNA, 0.35, skip=0.5, clip=cv2.GaussianBlur(up, (0, 0), 1.2))
        pen(s, up, 2.1, pc, 1.3, seed=seed + 5 + side)


def sit_me(s):
    """我 on the step, hugging my knees, head a little down and toward you."""
    F = SFig(s, K_ME)
    seed = 400
    r = rs(seed)
    hr = sit_head_rot()
    HQ = lambda pts: F.P([hr(u, v) for (u, v) in pts])
    paint = lambda m, a, b: me_knit(s, m, a, b)
    sit_arms(F, True, seed, paint, 42, "fo")
    seat(F, True, seed + 10)
    tor = s.poly(F.P(BACK))
    x0, x1 = F.p(-106, 0)[0], F.p(106, 0)[0]
    me_knit(s, tor, x0, x1)
    s.wash(tor, UMBER, 0.10, wob=2, bleed=14, grad=s.vgrad(F.p(0, -140)[1], F.p(0, -20)[1], 0, 1), clip=tor)
    # the back rounds over the knees: the light catches the top of it, the spine is a soft groove, the shoulder
    # blades stand out as the arms pull forward
    s.lift(tor * s.vgrad(F.p(0, -300)[1], F.p(0, -200)[1], 1, 0), 0.20, soft=8, wob=2)
    crease(s, F.P([(1, -280), (-1, -200), (1, -120), (0, -40)]), 24, OCHRE, 0.18, clip=tor)
    for side in (-1, 1):
        crease(s, F.P([(side * 20, -262), (side * 48, -226), (side * 56, -180)]), 26, DUSK, 0.16, clip=tor)
        crease(s, F.P([(side * 96, -110), (side * 64, -84), (side * 30, -76)]), 14, OCHRE, 0.2, clip=tor)
    cx = F.p(0, 0)[0]
    ribs(s, tor, np.arange(x0 + 8, x1 - 4, 9.0), lambda x: F.p(0, -302)[1] + 0.0028 * (x - cx) ** 2 + 4, F.p(0, -40)[1], seed=seed + 7, cx=cx, spread=0.08, bow=10)
    hb = tor * s.vgrad(F.p(0, -42)[1], F.p(0, -37)[1])
    s.wash(hb, OCHRE, 0.22, wob=1.5, soft=2, edge=0.4)
    for x in np.arange(x0 - 2, x1 + 2, 5.0):
        s.ink([(x, F.p(0, -40)[1]), (x + (x - cx) * 0.02, F.p(0, -14)[1] - 0.0003 * (x - cx) ** 2)], 0.95, SIENNA, 0.5, skip=0.25, clip=hb)
    s.ink(F.P([(-98, -42), (-50, -39), (0, -40), (50, -39), (98, -42)]), 1.3, SEPIA, 0.7, skip=0.2)
    pen(s, tor, 2.3, SEPIA, 1.35, seed=seed + 8)
    sit_arms(F, True, seed, paint, 42, "up")
    nk = s.poly(HQ([(-18, -334), (18, -334), (22, -292), (-22, -292)]))
    base(s, nk, SKIN, 0.6, wob=1, edge=0.3)
    s.wash(nk, UMBER, 0.35, wob=1, bleed=5, clip=nk)
    sc_ = s.poly(F.P(SCARF_ME))
    cloth(s, sc_, MINT, LEAF, F.p(-52, 0)[0], F.p(52, 0)[0], k=0.8, deep=0.40, light=0.15, fs=40, gran=0.5)
    s.wash(sc_ * s.vgrad(F.p(0, -300)[1], F.p(0, -282)[1], 0, 1), LEAF, 0.3, wob=1.5, bleed=4, clip=sc_)
    for q in np.linspace(0, 1, 6):
        s.ink(F.P([(-50, -288 - 36 * q), (0, -298 - 34 * q), (50, -288 - 36 * q)]), 1.0, LEAF, 0.5, skip=0.4, clip=cv2.GaussianBlur(sc_, (0, 0), 1))
    crease(s, F.P([(-40, -300), (0, -294), (40, -300)]), 10, LEAF, 0.35, clip=sc_)
    pen(s, sc_, 2.0, INKBLUE, 1.05, seed=seed + 9)
    hx, hy = F.p(*hr(0, -352))
    for side in (-1, 1):
        ex_, ey_ = F.p(*hr(side * 46, -344))
        e = s.ellipse(ex_, ey_, 8, 16, side * 10 + TILT_SIT)
        base(s, e, SKIN, 0.65, wob=0.8, edge=0.4)
        s.wash(e, ROSE, 0.3, wob=0.8, soft=2)
        pen(s, e, 1.3, SEPIA, 1.0, again=0)
    hm = s.poly(HQ(HAIR_SIT))
    tm = s.poly(F.P(TAIL_SIT))
    hair_paint(s, np.maximum(hm, tm), hx, hy - 4, 64, HAIR_ME, seed + 11)
    # a little from above: more of the crown, the strands run from the whorl back and down into the tie
    cwx, cwy = F.p(*hr(8, -390))
    tx, ty = F.p(*TIE_SIT)
    mh = cv2.GaussianBlur(hm, (0, 0), 1)
    for i in range(36):
        a = -2.4 + 4.8 * (i + r.random() * 0.7) / 36
        sx_, sy_ = cwx + math.sin(a) * 18, cwy - math.cos(a) * 7
        ex2, ey2 = hx + math.sin(a) * 60, hy - math.cos(a) * 26 + 10
        s.ink([(sx_, sy_), (ex2, ey2), (tx + math.sin(a) * 8, ty)], 1.0, INK, 0.5, skip=0.35, clip=mh, tremor=0.8)
    for i in range(12):
        q = (i + r.random() * 0.5) / 12 - 0.5
        s.ink([(tx + q * 18, ty), (tx + q * 44, ty + 50 * F.k), (tx + q * 16, ty + 92 * F.k - abs(q) * 18)], 1.0, INK, 0.5, skip=0.3, clip=cv2.GaussianBlur(tm, (0, 0), 1), tremor=0.9)
    pen(s, hm, 2.0, INK, 1.3, seed=seed + 12, again=1)
    pen(s, tm, 1.9, INK, 1.25, seed=seed + 13, again=1)
    tie = s.ellipse(tx, ty + 1, 15, 8, TILT_SIT)
    base(s, tie, UMBER, 1.0, wob=0.6, edge=0.4)
    pen(s, tie, 1.5, INK, 1.2, again=0)
    for pts, w in (([(-44, -338), (-49, -322), (-45, -304), (-50, -292)], 1.1), ([(42, -338), (47, -322), (43, -308)], 1.0),
                   ([(-12, -306), (-18, -295), (-15, -286)], 0.9), ([(-20, -398), (-30, -410), (-42, -406)], 0.9), ([(14, -400), (24, -412)], 0.8)):
        s.ink(HQ(pts), w, INK, 0.9, taper=(0.15, 0.6), tremor=0.9)
    for pts in (([(-10, -220), (-15, -207), (-12, -200)]), ([(8, -222), (13, -208)])):
        s.ink(F.P(pts), 0.9, INK, 0.9, taper=(0.15, 0.6))
    sh_ = F.p(0, -290)
    return {"shoulders": [round(sh_[0] - SX, 1), round(sh_[1] - SY, 1)], "head": [round(hx - SX, 1), round(hy - SY, 1)]}


def sit_you(s, tee=False):
    """你 on the step, the navy jacket on or (tee) off."""
    F = SFig(s, K_YOU, WF_YOU)
    seed = 450 + (5 if tee else 0)
    r = rs(seed)
    hr = rot((0, -312), math.radians(-3))
    HQ = lambda pts: F.P([hr(u, v) for (u, v) in pts])
    paint = lambda m, a, b: navy(s, m, a, b)
    sit_arms(F, False, seed, paint, 46, "fo", tee=tee)
    seat(F, False, seed + 10)
    tor = s.poly(F.P(BACK))
    x0, x1 = F.p(-106, 0)[0], F.p(106, 0)[0]
    if tee:
        clip = cv2.GaussianBlur(tor, (0, 0), 2)
        base(s, tor, TEE, 0.72, wob=2, edge=0.45, flow=0.15, fs=200, gran=0.4)
        s.wash(tor, SLATE, 0.36, wob=2, bleed=10, grad=s.hgrad(x0, x1, 0.1, 1), clip=clip)
        s.wash(tor, DUSK, 0.22, wob=2, bleed=12, grad=s.vgrad(F.p(0, -200)[1], F.p(0, -20)[1], 0, 1), clip=clip)
        s.lift(tor * s.vgrad(F.p(0, -300)[1], F.p(0, -210)[1], 1, 0), 0.25, soft=8, wob=2)
        crease(s, F.P([(0, -280), (-2, -200), (1, -120), (0, -50)]), 14, SLATE, 0.09, clip=tor)      # the spine under thin cotton
        for side in (-1, 1):
            crease(s, F.P([(side * 18, -266), (side * 50, -226), (side * 58, -180)]), 26, SLATE, 0.18, clip=tor)
            for i in range(3):        # the cotton pulled across the back by the arms, slack at the waist
                y = -70 + i * 18
                crease(s, F.P([(side * 98, y), (side * 64, y + 9), (side * 26, y + 4)]), 8, SLATE, 0.18, clip=tor)
                s.ink(F.P([(side * 96, y), (side * 68, y + 8)]), 1.1, SEPIA, 0.5, skip=0.3, clip=tor)
            crease(s, F.P([(side * 30, -250), (side * 70, -200), (side * 98, -170)]), 9, SLATE, 0.25, clip=tor)
        s.ink(F.P([(-94, -20), (-46, -15), (0, -17), (46, -15), (94, -20)]), 1.3, SEPIA, 0.6, skip=0.2)
        pen(s, tor, 2.2, SEPIA, 1.3, seed=seed + 8)
        nb = s.poly(F.P([(-32, -308), (-16, -315), (0, -317), (16, -315), (32, -308), (26, -298), (0, -302), (-26, -298)]))
        base(s, nb, TEE, 0.8, wob=1, edge=0.5)
        s.wash(nb, SLATE, 0.35, wob=1, soft=2)
        pen(s, nb, 1.5, SEPIA, 1.0, again=0)
    else:
        navy(s, tor, x0, x1)
        s.wash(tor, PAYNE, 0.25, wob=2, bleed=14, grad=s.vgrad(F.p(0, -170)[1], F.p(0, -20)[1], 0, 1), clip=tor)
        s.lift(tor * s.vgrad(F.p(0, -300)[1], F.p(0, -220)[1], 1, 0), 0.22, soft=8, wob=2)
        yk = F.P([(-100, -252), (-50, -244), (0, -242), (50, -244), (100, -252)])
        s.ink(yk, 1.5, INK, 0.75, skip=0.15, clip=tor)
        crease(s, yk, 10, INDIGO, 0.3, clip=tor)
        for side in (-1, 1):
            s.ink(F.P([(side * 46, -244), (side * 48, -150), (side * 54, -44)]), 1.3, INK, 0.5, skip=0.35, clip=tor)
            crease(s, F.P([(side * 20, -262), (side * 50, -220), (side * 58, -170)]), 26, INDIGO, 0.2, clip=tor)
        hb = tor * s.vgrad(F.p(0, -42)[1], F.p(0, -37)[1])
        s.wash(hb, INDIGO, 0.35, wob=1.5, soft=2, edge=0.4)
        s.ink(F.P([(-98, -42), (-50, -39), (0, -40), (50, -39), (98, -42)]), 1.5, INK, 0.8, skip=0.15)
        for i in range(9):
            u = -84 + i * 21 + r.random() * 5
            crease(s, F.P([(u, -40), (u + 3, -58), (u + 8, -78 - r.random() * 14)]), 8, INKBLUE, 0.3, clip=tor)
            s.ink(F.P([(u, -41), (u + 2, -54)]), 1.1, INK, 0.55, clip=tor)
        pen(s, tor, 2.4, INK, 1.4, seed=seed + 8)
        cl = s.poly(F.P([(-36, -310), (-22, -320), (0, -323), (22, -320), (36, -310), (48, -292), (30, -286), (0, -289), (-30, -286), (-48, -292)]))
        navy(s, cl, F.p(-48, 0)[0], F.p(48, 0)[0], k=1.1, deep=0.4)
        pen(s, cl, 1.8, INK, 1.3, again=1)
        s.ink(F.P([(-34, -308), (0, -311), (34, -308)]), 1.2, INK, 0.7, skip=0.2)
    sit_arms(F, False, seed, paint, 46, "up", tee=tee)
    nk = s.poly(HQ([(-24, -344), (24, -344), (27, -300), (-27, -300)]))
    base(s, nk, SKIN, 0.6, wob=1, edge=0.3)
    s.wash(nk, BURNT, 0.3, wob=1, bleed=5, grad=s.hgrad(F.p(-24, 0)[0], F.p(27, 0)[0], 0, 1), clip=nk)
    s.wash(nk, UMBER, 0.35, wob=1, bleed=5, grad=s.vgrad(F.p(0, -344)[1], F.p(0, -320)[1], 1, 0.1), clip=nk)
    s.ink(HQ([(-3, -334), (-1, -318), (-3, -308)]), 1.2, BURNT, 0.5)
    hx, hy = F.p(*hr(0, -368))
    for side in (-1, 1):
        e = s.poly(HQ([(side * 40, -380), (side * 51, -385), (side * 54, -368), (side * 51, -350), (side * 42, -346)]))
        base(s, e, SKIN, 0.66, wob=0.8, edge=0.4)
        s.wash(e, ROSE, 0.32, wob=0.8, soft=2)
        pen(s, e, 1.5, SEPIA, 1.1, again=0)
    HP = [(0, -422), (-30, -417), (-46, -398), (-50, -370), (-47, -346), (-40, -334), (-27, -328), (-12, -332), (0, -328),
          (12, -332), (27, -327), (40, -334), (47, -346), (50, -370), (46, -398), (30, -417)]
    hm = s.poly(HQ(HP))
    hair_paint(s, hm, hx, hy, 62, HAIR_YOU, seed + 11)
    crx, cry = F.p(*hr(6, -404))
    mm = cv2.GaussianBlur(hm, (0, 0), 1)
    for i in range(56):
        a = r.random() * 6.28
        d0 = 6 + r.random() * 38
        x_, y_ = crx + math.cos(a) * d0, cry + math.sin(a) * d0 * 0.8
        L = 12 + r.random() * 12
        dd = np.array([math.cos(a) * 0.6, math.sin(a) * 0.5 + 0.5])
        s.ink([(x_, y_), (x_ + dd[0] * L * 0.6, y_ + dd[1] * L * 0.5), (x_ + dd[0] * L, y_ + dd[1] * L)], 1.1, INK, 0.55, taper=(0.2, 0.6), clip=mm)
    for i in range(16):
        u = -36 + i * 4.8 + r.random() * 2
        a = F.p(*hr(u, -340 + abs(u) * 0.1))
        s.ink([a, (a[0] + (r.random() - 0.5) * 2, a[1] + 8 + r.random() * 4)], 1.0, INK, 0.55, taper=(0.1, 0.7))
    pen(s, hm, 2.0, INK, 1.35, seed=seed + 12, again=1)
    sh_ = F.p(0, -290)
    return {"shoulders": [round(sh_[0] - SX, 1), round(sh_[1] - SY, 1)], "head": [round(hx - SX, 1), round(hy - SY, 1)]}


def sp_me():
    s = Sheet(SW, SH, 401)
    s.save("sp_me", crop=True, anchor=(SX, SY), meta=sit_me(s))


def sp_you():
    s = Sheet(SW, SH, 451)
    s.save("sp_you", crop=True, anchor=(SX, SY), meta=sit_you(s))


def sp_you_tee():
    s = Sheet(SW, SH, 456)
    s.save("sp_you_tee", crop=True, anchor=(SX, SY), meta=sit_you(s, tee=True))


COAT = [(-30, -312), (-62, -308), (-96, -294), (-120, -268), (-136, -230), (-148, -186), (-154, -140), (-152, -92), (-146, -46),
        (-140, -10), (-70, -4), (0, -8), (70, -4), (140, -10), (146, -46), (152, -92), (154, -140), (148, -186), (136, -230),
        (120, -268), (96, -294), (62, -308), (30, -312), (0, -314)]


def sp_coat_on():
    """Your jacket over my shoulders, drawn to lie exactly on sp_me (same sheet, same anchor). It is bigger than me:
    it tents over the elbows; the empty sleeves hang at the sides. My hair, its tail and the scarf stay on top."""
    s = Sheet(SW, SH, 470)
    F = SFig(s, K_ME)
    hr = sit_head_rot()
    keep = np.maximum(np.maximum(s.poly(F.P([hr(u, v) for u, v in HAIR_SIT])), s.poly(F.P(TAIL_SIT))), s.poly(F.P(SCARF_ME)))
    keep = np.maximum(keep, s.ellipse(*F.p(*TIE_SIT), 16, 9))
    keep = cv2.GaussianBlur(cv2.dilate(keep, np.ones((3, 3), np.uint8)), (0, 0), 0.8)
    free = 1 - keep
    for side in (-1, 1):        # the empty sleeves first, half under the body of the jacket
        P = F.P([(side * 112, -262), (side * 140, -200), (side * 152, -120), (side * 150, -36)])
        sl = s.ribbon(P, 46 * F.k, profile=lambda t: 1 - 0.1 * t)
        d = unit(P[2], P[3])
        sl *= halfplane(s, P[3], d, 0)
        navy(s, sl, *sorted((F.p(side * 120, 0)[0], F.p(side * 176, 0)[0])))
        s.wash(sl, PAYNE, 0.25, wob=1, bleed=5, clip=sl)
        cf = sl * (1 - halfplane(s, P[3], d, -22))
        s.wash(cf, PAYNE, 0.35, wob=1, soft=2)
        for i in range(3):
            y = -150 + i * 30
            crease(s, F.P([(side * 134, y), (side * 152, y + 8), (side * 170, y + 2)]), 8, INKBLUE, 0.35, clip=sl)
        pen(s, sl, 2.1, INK, 1.3, seed=470 + side)
    body = s.poly(F.P(COAT)) * free
    x0, x1 = F.p(-154, 0)[0], F.p(154, 0)[0]
    navy(s, body, x0, x1)
    s.wash(body, PAYNE, 0.22, wob=2, bleed=14, grad=s.vgrad(F.p(0, -180)[1], F.p(0, -10)[1], 0, 1), clip=body)
    s.lift(body * s.vgrad(F.p(0, -300)[1], F.p(0, -230)[1], 1, 0), 0.22, soft=8, wob=2)
    mb = cv2.GaussianBlur(body, (0, 0), 1)
    for side in (-1, 1):        # it hangs from the shoulders and tents over the elbows: the folds fall from those points
        for (a, b, c, w, k) in (((side * 86, -270), (side * 112, -210), (side * 120, -130), 22, 0.35), ((side * 128, -150), (side * 134, -90), (side * 136, -16), 20, 0.35),
                                ((side * 50, -262), (side * 58, -170), (side * 66, -20), 26, 0.2), ((side * 104, -110), (side * 98, -60), (side * 96, -12), 14, 0.3)):
            crease(s, F.P([a, b, c]), w, INDIGO, k, clip=body)
            s.ink(F.P([a, b, c]), 1.2, INK, 0.45, skip=0.4, clip=mb)
    yk = F.P([(-122, -262), (-60, -252), (0, -250), (60, -252), (122, -262)])
    s.ink(yk, 1.5, INK, 0.75, skip=0.15, clip=body)
    crease(s, yk, 10, INDIGO, 0.3, clip=body)
    hb = body * s.vgrad(F.p(0, -36)[1], F.p(0, -31)[1])
    s.wash(hb, INDIGO, 0.35, wob=1.5, soft=2, edge=0.4)
    s.ink(F.P([(-144, -34), (-70, -28), (0, -32), (70, -28), (144, -34)]), 1.5, INK, 0.8, skip=0.15)
    pen(s, body, 2.4, INK, 1.4, seed=478)
    cl = s.poly(F.P([(-46, -306), (-30, -318), (0, -322), (30, -318), (46, -306), (60, -286), (34, -278), (0, -282), (-34, -278), (-60, -286)])) * free
    navy(s, cl, F.p(-60, 0)[0], F.p(60, 0)[0], k=1.1, deep=0.4)
    pen(s, cl, 1.8, INK, 1.3, again=1)
    sh_ = F.p(0, -290)
    s.save("sp_coat_on", crop=True, anchor=(SX, SY), meta={"shoulders": [round(sh_[0] - SX, 1), round(sh_[1] - SY, 1)]})


def sp_coat_fold():
    """The jacket in mid-air, folded over itself the way it hangs over a forearm: the fold on top, the two halves
    hanging unevenly, a sleeve swinging, the collar flopped at one end. About as wide as sp_coat_on; anchor its centre."""
    s = Sheet(600, 520, 480)
    cx, cy = 300, 250
    r = rs(480)
    # the far half (shows lower on the right, in shadow)
    far = s.poly([(cx - 130, cy - 70), (cx + 40, cy - 92), (cx + 150, cy - 70), (cx + 170, cy + 10), (cx + 164, cy + 120), (cx + 120, cy + 150),
                  (cx + 40, cy + 140), (cx - 40, cy + 120)])
    navy(s, far, cx - 130, cx + 170, k=1.15, deep=0.6)
    s.wash(far, PAYNE, 0.4, wob=2, bleed=10, clip=far)
    pen(s, far, 2.2, INK, 1.35, seed=481)
    sl = s.ribbon([(cx - 120, cy - 60), (cx - 166, cy + 10), (cx - 178, cy + 100), (cx - 160, cy + 190)], 58, profile=lambda t: 1 - 0.15 * t)
    sl *= halfplane(s, (cx - 160, cy + 190), unit((cx - 178, cy + 100), (cx - 160, cy + 190)), 0)
    navy(s, sl, cx - 200, cx - 120)
    for i in range(3):
        y = cy + 20 + i * 44
        crease(s, [(cx - 192, y), (cx - 170, y + 10), (cx - 148, y + 2)], 8, INKBLUE, 0.35, clip=sl)
    cf = sl * (1 - halfplane(s, (cx - 160, cy + 190), unit((cx - 178, cy + 100), (cx - 160, cy + 190)), -20))
    s.wash(cf, PAYNE, 0.35, wob=1, soft=2)
    pen(s, sl, 2.2, INK, 1.35, seed=483)
    # the near half, over the fold: its top edge bows where it lies over the arm, its hem swings unevenly
    near = s.poly([(cx - 150, cy - 64), (cx - 60, cy - 100), (cx + 40, cy - 108), (cx + 130, cy - 86), (cx + 150, cy - 50), (cx + 140, cy + 30),
                   (cx + 128, cy + 96), (cx + 60, cy + 112), (cx - 20, cy + 104), (cx - 100, cy + 92), (cx - 128, cy + 40), (cx - 140, cy - 10)])
    navy(s, near, cx - 150, cx + 150)
    s.lift(near * s.vgrad(cy - 108, cy - 50, 1, 0), 0.22, soft=6, wob=2)
    mb = cv2.GaussianBlur(near, (0, 0), 1)
    for (a, b, c, w) in (((cx - 110, cy - 60), (cx - 100, cy + 10), (cx - 110, cy + 86), 22), ((cx - 20, cy - 96), (cx - 10, cy), (cx - 22, cy + 100), 26),
                         ((cx + 70, cy - 90), (cx + 84, cy), (cx + 76, cy + 104), 20), ((cx + 40, cy - 60), (cx + 30, cy + 20), (cx + 40, cy + 100), 12)):
        crease(s, [a, b, c], w, INDIGO, 0.35, clip=near)
        s.ink([a, b, c], 1.2, INK, 0.5, skip=0.4, clip=mb)
    hb = near * s.vgrad(cy + 82, cy + 88)
    s.wash(hb, INDIGO, 0.35, wob=1.5, soft=2, edge=0.4)
    s.ink([(cx - 104, cy + 84), (cx - 20, cy + 96), (cx + 60, cy + 102), (cx + 126, cy + 88)], 1.5, INK, 0.8, skip=0.15)
    s.ink([(cx - 30, cy - 98), (cx - 26, cy - 10), (cx - 34, cy + 96)], 1.3, INK, 0.5, skip=0.35, clip=near)      # back seam
    pen(s, near, 2.4, INK, 1.4, seed=482)
    # the collar, flopped over the right end
    cl = s.poly([(cx + 112, cy - 92), (cx + 156, cy - 84), (cx + 178, cy - 52), (cx + 176, cy - 14), (cx + 154, cy - 20), (cx + 140, cy - 50), (cx + 116, cy - 70)])
    navy(s, cl, cx + 110, cx + 180, k=1.15, deep=0.4)
    pen(s, cl, 1.9, INK, 1.3, again=1)
    s.save("sp_coat_fold", crop=True, anchor=(cx, cy), meta={"fold": [[-150, -64], [150, -50]]})


def pair(frame, mode):
    """mode: "pair" (dusk, scarf), "umb" (rain, scarf, one umbrella), "furl" (daylight, no scarf, furled umbrella)."""
    s = Sheet(1000, 1300, 200 + frame + {"pair": 0, "umb": 10, "furl": 20}[mode])
    f = 1 if frame == 0 else -1
    mx, yx = AX + ME_DX, AX + YOU_DX
    meta = {"h_me": H_ME, "h_you": H_YOU}
    can = None
    if mode == "umb":
        can = Canopy(AX - 14, AY - 975, 335, math.radians(-8), bent=0, seed=5 + frame)
        can.under(s)
    head_y, feet_y, hand, wrists = you_walk(s, yx, AY, f, mode="umb" if mode == "umb" else ("furl" if mode == "furl" else "free"),
                                            lean=-3 * f, seed=30 + frame * 3, wet=(mode == "umb"))
    if mode == "furl":
        furled(s, wrists[1], AY)
    head_m, feet_m = me_walk(s, mx, AY, f, scarf=(mode != "furl"), lean=-3 * f + (4 if mode == "umb" else 0), seed=10 + frame * 3)
    if mode == "umb":
        top = can.proj(np.array([0, 0, can.Hd - 10]))
        shaft(s, hand, top)
        # the hand round the shaft, just showing past the shoulder
        hm = s.ellipse(hand[0] + 2, hand[1] - 2, 15, 20, -20)
        base(s, hm, SKIN, 0.62, wob=1, edge=0.4)
        s.wash(hm, BURNT, 0.3, wob=1, soft=3, grad=s.hgrad(hand[0] - 15, hand[0] + 15, 0, 1))
        pen(s, hm, 1.6, SEPIA, 1.2, again=0)
        for k in range(3):
            s.ink([(hand[0] - 12, hand[1] - 10 + k * 8), (hand[0] + 2, hand[1] - 7 + k * 8), (hand[0] + 12, hand[1] - 10 + k * 8)], 1.1, SEPIA, 0.7)
        can.outer(s)
        tips = can.drips(s)
        meta["umb"] = {"apex": [round(can.apex()[0] - AX, 1), round(can.apex()[1] - AY, 1)], "rim": [can.cx - AX, can.cy - AY], "r": can.R,
                       "tips": [[round(t[0] - AX, 1), round(t[1] - AY, 1)] for t in tips]}
    meta["head_me"] = [round(head_m[0] - AX, 1), round(head_m[1] - AY, 1)]
    meta["head_you"] = [round(head_y[0] - AX, 1), round(head_y[1] - AY, 1)]
    meta["feet"] = feet_m + feet_y
    name = {"pair": "fig_pair", "umb": "fig_umb", "furl": "fig_furl"}[mode]
    s.save(f"{name}_w{frame}", crop=True, anchor=(AX, AY), meta=meta)
    return s


def furled(s, wr, gy):
    """The furled umbrella hanging by its crook from your right hand, down beside the leg, the tip swung a little out."""
    x, y = wr
    top = (x + 3, y + 84)
    tip = (x + 30, gy - 96)
    body = s.ribbon([top, (x + 14, (top[1] + tip[1]) / 2), tip], 40, profile=lambda t: np.interp(t, [0, 0.1, 0.5, 1], [0.3, 1.0, 0.82, 0.16]))
    navy(s, body, x - 20, x + 40, k=1.1, deep=0.4)
    mb = cv2.GaussianBlur(body, (0, 0), 1)
    for k in range(6):          # the wrapped cloth, folds spiralling down
        q = 0.1 + k * 0.14
        y0 = top[1] + (tip[1] - top[1]) * q
        x0 = top[0] + (tip[0] - top[0]) * q
        wq = 20 * (1 - q * 0.7)
        s.ink([(x0 - wq, y0 - 8), (x0 - wq * 0.2, y0 + 6), (x0 + wq, y0 + 26)], 1.3, INK, 0.65, clip=mb)
        crease(s, [(x0 - wq, y0 - 6), (x0, y0 + 8), (x0 + wq, y0 + 28)], 8, PAYNE, 0.35, clip=body)
    sy = top[1] + (tip[1] - top[1]) * 0.42           # the strap, snapped
    sx = top[0] + (tip[0] - top[0]) * 0.42
    strap = s.ribbon([(sx - 18, sy - 4), (sx + 20, sy + 6)], 13)
    base(s, strap, NAVY, 1.25, wob=0.5, edge=0.5)
    pen(s, strap, 1.3, INK, 1.2, again=0)
    s.wash(s.ellipse(sx + 12, sy + 4, 4, 4), SLATE, 0.8, wob=0.3, soft=0.5)
    pen(s, body, 2.0, INK, 1.35)
    s.ink([(tip[0] - 1, tip[1] - 6), (tip[0] + 2, tip[1] + 20)], 4.0, PAYNE, 1.5, taper=(0, 0.35))
    # the crook: a wooden J over the curled fingers
    crook = s.ribbon([(top[0], top[1] + 6), (top[0], top[1] - 22), (top[0] - 8, top[1] - 38), (top[0] - 22, top[1] - 36), (top[0] - 26, top[1] - 22)], 13)
    base(s, crook, SIENNA, 1.0, wob=0.5, edge=0.5)
    s.wash(crook, BURNT, 0.5, wob=0.5, soft=1.5, grad=s.hgrad(top[0] - 28, top[0] + 6, 0, 1))
    s.lift(crook * s.hgrad(top[0] - 28, top[0] - 10, 1, 0), 0.3, soft=1, wob=0.5)
    pen(s, crook, 1.4, SEPIA, 1.3, again=0)
    # the fingers over the crook
    for k in range(3):
        fx = x - 12 + k * 9
        fm = s.ellipse(fx, y + 62 + k * 2, 6, 9)
        base(s, fm, SKIN, 0.62, wob=0.5, edge=0.4)
        s.wash(fm, BURNT, 0.25, wob=0.5, soft=2)
        pen(s, fm, 1.2, SEPIA, 1.0, again=0)


ALL = {
    "fig_pair": lambda: (pair(0, "pair"), pair(1, "pair")),
    "fig_umb": lambda: (pair(0, "umb"), pair(1, "umb")),
    "fig_furl": lambda: (pair(0, "furl"), pair(1, "furl")),
    "sp_me": sp_me, "sp_you": sp_you, "sp_you_tee": sp_you_tee, "sp_coat_on": sp_coat_on, "sp_coat_fold": sp_coat_fold,
}
