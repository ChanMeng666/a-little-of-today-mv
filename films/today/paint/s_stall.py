"""The breakfast shop downstairs (早餐店) and the soy milk cup close.  Also the hand habits shared by s_shop and s_step.

STALL  sheet 3072x1728, a spring morning, seen from across the lane at eye level.
  The lane is in cool shadow, the sun only reaches the top of the facade; inside the open front a warm light.
  Eye level y=1000, vanishing point x=1350; the shop front is a plane at 300 px/m, its floor line y=1480.
  st_bg     opaque. meta: pot [x,y] the soy milk pot's mouth (steam origin); steamers [[x,y],...] steam origins (the
            two steamer stacks' lids first, then leaks between tiers); griddle [x,y] (the griddle top, smoke origin);
            counter_y (the stainless counter's top edge); rice [x,y] the rice-milk pot's mouth (a smaller steam);
            opening [x0,y0,x1,y1] the open shop front (warm light); lane_y (the near kerb of the lane: a passing
            scooter's wheels run at about this y + 150); boss [x,y] where st_hands' anchor goes.
  st_fg     sprite, a concrete utility pole with stickers and its cables, close on the left edge (parallax: move it
            about 1.6x the background). Anchor = the pole's foot on the sheet (x, H): place it at its own anchor for
            the painted composition. meta: none beyond the anchor.
  st_hands  sprite, the boss behind the soy milk pot: white T-shirt sleeves, a dark apron's straps, the left hand on
            the ladle, the right resting on the rim; the head is lost in the steam. Anchor = the ladle's bowl, in the
            pot (place it at meta boss of st_bg). meta: cuff [dx,dy] the ladle arm's sleeve hem, grip [dx,dy] the hand
            on the ladle, handle [dx,dy] the ladle handle's top.  Stir: rotate a few degrees about the anchor.

SOY  sheet 2560x1440: the cup close on the counter, the warm kitchen behind out of focus.
  sy_bg     opaque. meta: counter_y (the back edge of the counter top), tray [x0,y0,x1,y1], cup [x,y] where the cup's
            anchor sits, straw [x,y] where the straw is laid down (its centre), steam [[x,y],...] kitchen steam far back.
  sy_cup    sprite, anchor = the bottom centre of the cup on the counter. meta: lid [cx,cy,rx,ry] the sealed lid ellipse
            relative to the anchor (the page writes "少糖" in it); steam [dx,dy] where steam leaves the lid edge.
  sy_straw  sprite, a straw in its paper wrapper, horizontal. Anchor = its centre. meta: len (px).
  sy_hand   sprite, the boss's left hand from the upper right, the straw pinched between thumb and index (the straw
            itself is not drawn: the page draws sy_straw at the anchor, under the hand). Anchor = the pinch point.
            meta: cuff [dx,dy] the white sleeve's hem; thumb [dx,dy] the thumb tip (draw sy_straw before this sprite).
"""
import math

import cv2
import numpy as np

from kit import *

FONTS = "D:/github_repository/a-little-of-today-mv/assets/fonts/"
KAI = FONTS + "LXGWWenKai-Regular.ttf"
SERIF = FONTS + "NotoSerifSC-VF.ttf"


# ================================================================================================ shared hand habits
class Sub(Sheet):
    """A rectangular part of a sheet that paints in the parent's coordinates (same paper, same pigment arrays), so a
    wash costs what its neighbourhood costs. (From 独身主义的咖啡's street.)"""

    def __init__(self, s, x0, y0, x1, y1):
        px, py = getattr(s, "ox", 0), getattr(s, "oy", 0)
        x0, y0 = max(px, int(x0)), max(py, int(y0))
        x1, y1 = min(px + s.w, int(x1)), min(py + s.h, int(y1))
        self.ox, self.oy = x0, y0
        self.w, self.h = x1 - x0, y1 - y0
        sl = (slice(y0 - py, y1 - py), slice(x0 - px, x1 - px))
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
    """A sheet of its own painted in world coordinates: its top-left corner sits at (ox, oy) on the world."""

    def __init__(self, w, h, ox, oy, seed=0):
        Sheet.__init__(self, w, h, seed)
        self.ox, self.oy, self._raw = int(ox), int(oy), False
        self._g = rs(seed + 977)


def contour(s, m, w=2.2, color=INK, k=1.5, level=0.5, step=5.0, **kw):
    """Pen the outline of a mask (kit's, made to work on a Sub: the mask is local, the pen takes world coordinates)."""
    cs, _ = cv2.findContours((m > level).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    o = np.array([getattr(s, "ox", 0), getattr(s, "oy", 0)], np.float64) if isinstance(s, Sub) else np.zeros(2)
    for c in cs:
        P = c[:, 0, :].astype(np.float64)
        if len(P) < 12:
            continue
        P = P[::max(1, int(step))]
        P = np.vstack([P, P[:2]])
        Pm = (np.roll(P, 1, 0) + P + np.roll(P, -1, 0)) / 3
        Pm[0], Pm[-1] = P[0], P[-1]
        s.ink(Pm + o, w, color, k, smooth=False, taper=(0.02, 0.04), **kw)


def J(r, v=1.0):
    return (r.random() - 0.5) * 2 * v


def U(r, a, b):
    return a + (b - a) * r.random()


def box(s, r, x0, y0, x1, y1, w=2.0, c=SEPIA, k=1.5, over=5.0, dbl=0.3, j=1.2, **kw):
    """A pen rectangle the way a hand draws it: four separate strokes, corners crossing, now and then gone over twice."""
    P = [(x0 + J(r, j), y0 + J(r, j)), (x1 + J(r, j), y0 + J(r, j)), (x1 + J(r, j), y1 + J(r, j)), (x0 + J(r, j), y1 + J(r, j))]
    for a, b in zip(P, P[1:] + P[:1]):
        seg(s, a, b, w, c, k, over=over * (0.2 + r.random()), r=r, **kw)
        if r.random() < dbl:
            d = (J(r, 2.2), J(r, 2.2))
            seg(s, (a[0] + d[0], a[1] + d[1]), (b[0] + d[0], b[1] + d[1]), w * 0.7, c, k * 0.6, over=over * r.random(), r=r, skip=0.25)


def line(s, r, a, b, w=2.0, c=SEPIA, k=1.5, over=4.0, dbl=0.0, **kw):
    seg(s, a, b, w, c, k, over=over * (0.3 + r.random()), r=r, **kw)
    if dbl and r.random() < dbl:
        d = (J(r, 2), J(r, 2))
        seg(s, (a[0] + d[0], a[1] + d[1]), (b[0] + d[0], b[1] + d[1]), w * 0.7, c, k * 0.6, r=r, skip=0.3)


def pl(s, r, pts, w=2.0, c=SEPIA, k=1.5, over=3.0, closed=False, **kw):
    P = list(pts) + ([pts[0]] if closed else [])
    for a, b in zip(P, P[1:]):
        line(s, r, a, b, w, c, k, over=over, **kw)


def put(s, m, color, k=0.8, lift=0.9, wob=1.2, **kw):
    """An object over what is already painted: lift most of what is under it, lay its wash, keep it opaque."""
    if lift > 0:
        s.lift(m, lift, soft=0.7, wob=0)
    return s.wash(m, color, k, wob=wob, body=1, **kw)


def hl(s, pts, w=2.0, amt=0.3, dry=0.0, clip=None):
    """A highlight: pigment lifted along a line."""
    s.wash(s.ribbon(pts, w, smooth=len(pts) > 2), None, amt, wob=0, soft=0.7, edge=0, gran=0.2, flow=0, lift=True, dry=dry, clip=clip)


def streaks(s, r, x0, x1, y0, y1, n, color=PAYNE, k=0.2, w=(5, 16), clip=None, dry=0.45):
    """Weather stains: thin dry-ish vertical drags of darker water."""
    for _ in range(n):
        x = U(r, x0, x1)
        ya = U(r, y0, y0 + (y1 - y0) * 0.5)
        yb = min(y1, ya + U(r, 0.25, 0.9) * (y1 - y0))
        s.stroke([(x, ya), (x + J(r, 3), (ya + yb) / 2), (x + J(r, 4), yb)], U(r, *w), color, k * U(r, 0.5, 1.3), dry=dry * U(r, 0.6, 1.2),
                 edge=0.3, wob=1.5, taper=(0.05, 0.6), clip=clip)


def mottle(s, r, x0, y0, x1, y1, colors, n=5, k=0.2, size=(60, 160), clip=None, bleed=18):
    """Wet-in-wet touches of neighbouring colour dropped into a wall while it is damp."""
    for i in range(n):
        c = colors[int(r.integers(0, len(colors)))]
        rx, ry = U(r, *size), U(r, *size)
        s.wash(s.blob(U(r, x0, x1), U(r, y0, y1), rx, ry * 0.8, irr=0.35), c, k * U(r, 0.5, 1.4), wob=4, bleed=bleed * U(r, 0.7, 1.4), edge=0.25, clip=clip)


def drags(s, r, x0, y0, x1, y1, c, n=6, k=(0.07, 0.15), w=(26, 70), clip=None):
    """A few drags of a nearly dry flat brush, so the hand shows on a flat wall."""
    for _ in range(n):
        if r.random() < 0.6:
            y = U(r, y0 + 10, y1 - 10)
            a = U(r, x0, x0 + (x1 - x0) * 0.5)
            b = min(x1, a + U(r, 0.3, 0.8) * (x1 - x0))
            pts = [(a, y), ((a + b) / 2, y + J(r, 8)), (b, y + J(r, 10))]
        else:
            x = U(r, x0 + 10, x1 - 10)
            a = U(r, y0, y0 + (y1 - y0) * 0.5)
            b = min(y1, a + U(r, 0.3, 0.7) * (y1 - y0))
            pts = [(x, a), (x + J(r, 8), (a + b) / 2), (x + J(r, 10), b)]
        s.stroke(pts, U(r, *w), c, U(r, *k), dry=U(r, 0.42, 0.6), edge=0.25, wob=2, taper=(0.1, 0.3), gran=0.5, clip=clip)


def cable(s, r, a, b, sag, w=1.4, c=INKBLUE, k=1.2):
    mid = ((a[0] + b[0]) / 2 + J(r, 6), (a[1] + b[1]) / 2 + sag)
    q1 = ((a[0] * 3 + b[0]) / 4, (a[1] * 3 + b[1]) / 4 + sag * 0.75)
    q3 = ((a[0] + b[0] * 3) / 4, (a[1] + b[1] * 3) / 4 + sag * 0.75)
    s.ink([a, q1, mid, q3, b], w, c, k, tremor=0.5, press=0.2, taper=(0.02, 0.02), dry=0.15)


def letters(s, txt, x, y, size, font=SERIF, weight=700, vertical=False, gap=0.06, rot=0.0, sx=1.0, sy=1.0):
    """Text as a mask, in world coordinates on a Sheet/Sub/Off (vertical: one character under another).
    sx, sy squash it about (x, y) (for lettering lying on the ground)."""
    from PIL import Image, ImageDraw, ImageFont
    ox, oy = getattr(s, "ox", 0), getattr(s, "oy", 0)
    im = Image.new("L", (s.w, s.h), 0)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(font, int(size))
    try:
        f.set_variation_by_axes([weight])
    except Exception:
        pass
    if vertical:
        yy = y - (len(txt) - 1) * size * (1 + gap) / 2
        for ch in txt:
            d.text((x - ox, yy - oy), ch, 255, f, anchor="mm")
            yy += size * (1 + gap)
    else:
        d.text((x - ox, y - oy), txt, 255, f, anchor="mm")
    m = np.asarray(im, np.float32) / 255
    if rot or sx != 1.0 or sy != 1.0:
        M = cv2.getRotationMatrix2D((x - ox, y - oy), rot, 1.0)
        M[0, :2] *= sx
        M[1, :2] *= sy
        M[0, 2] = (x - ox) - M[0, 0] * (x - ox) - M[0, 1] * (y - oy)
        M[1, 2] = (y - oy) - M[1, 0] * (x - ox) - M[1, 1] * (y - oy)
        m = cv2.warpAffine(m, M, (s.w, s.h))
    return m


def paint_letters(s, m, color, k=1.2, wob=0.5):
    """Sign-painter's lettering: a little wobble, pigment pooling at the edges of each stroke."""
    return s.wash(m, color, k, wob=wob, soft=0.5, edge=0.6, ew=1.6, gran=0.3, flow=0.4, fs=30)


def steam(s, r, x, y, h, w=40, amt=0.35, n=4, drift=0.0):
    """Painted steam: a few lifted, wet-edged wisps going up and opening out (the page adds moving steam over it)."""
    for i in range(n):
        ph = U(r, 0, 6.3)
        pts = []
        for q in np.linspace(0, 1, 7):
            pts.append((x + J(r, w * 0.2) + math.sin(ph + q * U(r, 3, 5)) * w * (0.3 + q) + drift * q * h, y - q * h))
        s.wash(s.ribbon(pts, w * U(r, 0.6, 1.2), taper=(0.3, 0.5), profile=lambda t: 0.6 + 0.9 * t), None, amt * U(r, 0.6, 1.0),
               wob=4, bleed=U(r, 8, 14), edge=0, gran=0.1, flow=0.3, fs=40, lift=True)


def tiles(s, r, x0, y0, x1, y1, t=20.0, c=SEPIA, k=0.55, patches=6, size=((60, 180), (40, 120)), singles=30, tints=(SLATE, CELADON), clip=None):
    """Small square tiles said in patches, the way a sketcher stops bothering, plus single tiles touched in."""
    for _ in range(patches):
        cx, cy = U(r, x0, x1), U(r, y0, y1)
        hw, hh = U(r, *size[0]) / 2, U(r, *size[1]) / 2
        j0, j1 = int(math.ceil((cy - hh - y0) / t)), int((cy + hh - y0) / t)
        i0, i1 = int(math.ceil((cx - hw - x0) / t)), int((cx + hw - x0) / t)
        for j in range(j0, j1 + 1):
            y = y0 + j * t
            if y0 < y < y1:
                a, b = max(x0, cx - hw * U(r, 0.6, 1.0)), min(x1, cx + hw * U(r, 0.6, 1.0))
                seg(s, (a, y + J(r, 0.5)), (b, y + J(r, 0.5)), 0.9, c, k * U(r, 0.6, 1.0), r=r, skip=0.25, bow=0.002, clip=clip)
        for i in range(i0, i1 + 1):
            x = x0 + i * t
            if x0 < x < x1:
                a, b = max(y0, cy - hh * U(r, 0.6, 1.0)), min(y1, cy + hh * U(r, 0.6, 1.0))
                seg(s, (x + J(r, 0.5), a), (x + J(r, 0.5), b), 0.9, c, k * U(r, 0.6, 1.0), r=r, skip=0.25, bow=0.002, clip=clip)
    nx, ny = max(1, int((x1 - x0) / t)), max(1, int((y1 - y0) / t))
    for _ in range(singles):
        i, j = int(r.integers(0, nx)), int(r.integers(0, ny))
        m = s.rect(x0 + i * t + 1.5, y0 + j * t + 1.5, x0 + (i + 1) * t - 1.5, y0 + (j + 1) * t - 1.5)
        if r.random() < 0.35:
            s.lift(m, U(r, 0.15, 0.35), soft=0.7, wob=0)
        else:
            s.wash(m, tints[int(r.integers(0, len(tints)))], U(r, 0.12, 0.35), wob=0, soft=0.7, edge=0.5, ew=2, flow=0, clip=clip)


def grille(s, r, x0, y0, x1, y1, under=34, c=MIST, pen=INKBLUE, gap=22.0, seed=0, plants=True):
    """An old iron window cage (鐵窗) seen from below: a frame, plain bars, a band of curls, its floor's underside,
    the window behind it, and whatever lives in it."""
    g = rs(seed)
    # the window behind: aluminium frames, glass with the sky in it, a curtain
    wx0, wy0, wx1, wy1 = x0 + 18, y0 + 18, x1 - 18, y1 - 8
    wm = s.rect(wx0, wy0, wx1, wy1)
    put(s, wm, SLATE, 0.55, lift=0.8, wob=0.8, edge=0.45, ew=4, flow=0.5, fs=60)
    s.wash(wm, PAYNE, 0.35, wob=1, soft=2, grad=s.vgrad(wy0, wy1, 1.0, 0.3))
    nl = 3 if (x1 - x0) > 500 else 2
    for i in range(nl):
        a = wx0 + (wx1 - wx0) * i / nl
        b = wx0 + (wx1 - wx0) * (i + 1) / nl
        s.lift(Q(s, [(a + 8, wy0 + 8), (a + (b - a) * U(g, 0.4, 0.8), wy0 + 8), (a + 8, wy0 + (wy1 - wy0) * U(g, 0.4, 0.7))]), U(g, 0.25, 0.45), soft=3, wob=1.5)
        if g.random() < 0.6:   # a curtain inside
            cc = (ROSE, CELADON, CREAM, OCHRE)[int(g.integers(0, 4))]
            s.wash(s.rect(a + 6, wy0 + 6, a + (b - a) * U(g, 0.3, 0.6), wy1 - 6), cc, 0.32, wob=1.5, soft=1, edge=0.5, ew=3)
        line(s, r, (a, wy0), (a + J(r, 1), wy1), 2.0, pen, 1.2)
    box(s, r, wx0, wy0, wx1, wy1, 1.6, pen, 1.2, over=3, dbl=0.3)
    line(s, r, (wx0 - 4, wy1 + 3), (wx1 + 4, wy1 + 3), 2.2, pen, 1.3, dbl=0.5)
    # the cage floor seen from underneath, and its shadow on the wall
    um = Q(s, [(x0, y1), (x1, y1), (x1 - 6, y1 + under), (x0 + 6, y1 + under)])
    put(s, um, SLATE, 0.6, lift=0.7, wob=0.8, edge=0.4, ew=3)
    s.wash(um, BURNT, 0.25, wob=2, bleed=6, edge=0.2, flow=0.7, fs=30)                   # rust
    s.wash(s.rect(x0 + 10, y1 + under, x1 - 10, y1 + under + 26), SLATE, 0.35, wob=1.5, bleed=6, edge=0.1)
    # plants that live in it: pots behind the lower band, leaves over it, a pothos falling through
    if plants:
        for _ in range(int(g.integers(2, 5))):
            px = U(g, x0 + 40, x1 - 40)
            for _ in range(int(g.integers(5, 12))):
                a = U(g, -2.6, -0.5)
                L = U(g, 20, 52)
                tip = (px + math.cos(a) * L, y1 - 34 + math.sin(a) * L)
                s.stroke([(px + J(g, 4), y1 - 30), ((px + tip[0]) / 2 + J(g, 5), (y1 - 30 + tip[1]) / 2), tip], U(g, 6, 12),
                         (MOSS, LEAF, TEAL)[int(g.integers(0, 3))], U(g, 0.5, 0.85), taper=(0.1, 0.7), wob=1, edge=0.5, body=0.9)
        for _ in range(int(g.integers(1, 3))):     # a trailing vine
            vx = U(g, x0 + 30, x1 - 30)
            L = U(g, 60, 160)
            pts = [(vx + math.sin(q * 3 + vx) * 10, y1 - 6 + q * L) for q in np.linspace(0, 1, 6)]
            s.ink(pts, 1.2, LEAF, 1.0, taper=(0.05, 0.2))
            for q in np.linspace(0.05, 1, 7):
                lx, ly = pts[0][0] + math.sin(q * 3 + vx) * 10, y1 - 6 + q * L
                sd = 1 if g.random() < 0.5 else -1
                s.wash(s.blob(lx + sd * 7, ly, 8, 6, irr=0.2), (MOSS, LEAF)[int(g.integers(0, 2))], U(g, 0.6, 0.9), wob=0.8, soft=0.8, edge=0.5, ew=2, body=0.9)
    # the bars
    x = x0 + gap * 0.5
    while x < x1 - 4:
        seg(s, (x + J(r, 0.8), y0 + 2), (x + J(r, 1.0), y1 - 2), 1.6, pen, U(r, 1.0, 1.4), r=r, skip=0.03, bow=0.002)
        hl(s, [(x - 2.2, y0 + 6), (x - 2.2, y1 - 6)], 1.4, 0.18)
        x += gap * U(r, 0.92, 1.08)
    # the band of curls, low down, and a top band of half-circles
    by0, by1 = y1 - 74, y1 - 34
    line(s, r, (x0, by0), (x1, by0 + J(r, 1)), 2.0, pen, 1.4)
    line(s, r, (x0, by1), (x1, by1 + J(r, 1)), 2.0, pen, 1.4)
    cx = x0 + gap
    while cx < x1 - gap * 0.5:
        rr = (by1 - by0) * 0.32
        s.ink(ell(cx - rr * 0.5, (by0 + by1) / 2 - rr * 0.35, rr * 0.7, rr * 0.7, 0.3, 4.9, 16), 1.2, pen, 1.1, tremor=0.4)
        s.ink(ell(cx + rr * 0.5, (by0 + by1) / 2 + rr * 0.35, rr * 0.7, rr * 0.7, 3.4, 8.0, 16), 1.2, pen, 1.1, tremor=0.4)
        cx += gap * 2
    ty = y0 + 40
    line(s, r, (x0, ty), (x1, ty + J(r, 1)), 1.6, pen, 1.2)
    cx = x0 + gap
    while cx < x1 - gap * 0.5:
        s.ink(ell(cx, ty, gap * 0.5, gap * 0.45, 0, math.pi, 12), 1.1, pen, 1.0, tremor=0.4)
        cx += gap
    box(s, r, x0, y0, x1, y1, 2.4, pen, 1.5, over=5, dbl=0.5)
    line(s, r, (x0 + 6, y1 + under), (x1 - 6, y1 + under), 1.8, pen, 1.3)
    line(s, r, (x0, y1), (x0 + 6, y1 + under), 1.6, pen, 1.3, over=1)
    line(s, r, (x1, y1), (x1 - 6, y1 + under), 1.6, pen, 1.3, over=1)
    # rust bleeding down the wall from its feet
    for xx in (x0 + 4, x1 - 4):
        s.stroke([(xx, y1 + under), (xx + J(r, 3), y1 + under + U(r, 60, 140))], U(r, 6, 12), BURNT, 0.22, dry=0.3, edge=0.3, wob=1, taper=(0, 0.7))


def acunit(s, r, x, y, w=150, h=100, pen=INKBLUE):
    """A window air-conditioner box hung on brackets, with its drip stain."""
    m = s.rect(x, y, x + w, y + h)
    put(s, m, CREAM, 0.35, lift=0.85, wob=0.7, edge=0.4, ew=3)
    s.wash(m, SLATE, 0.45, wob=1, soft=2, edge=0, grad=s.hgrad(x, x + w, 0.2, 1))
    s.wash(m, UMBER, 0.12, wob=2, bleed=5, edge=0.1, flow=0.8, fs=20)
    for q in np.linspace(0.18, 0.82, 8):
        line(s, r, (x + 10, y + h * q), (x + w * 0.55, y + h * q + J(r, 0.6)), 1.0, pen, 0.8, over=0, skip=0.15)
    s.wash(s.rect(x + w * 0.62, y + 12, x + w - 12, y + h - 12), SLATE, 0.3, wob=0.6, soft=1, edge=0.5, ew=2)
    box(s, r, x + w * 0.62, y + 12, x + w - 12, y + h - 12, 1.1, pen, 1.0, over=1)
    box(s, r, x, y, x + w, y + h, 2.0, pen, 1.4, over=3, dbl=0.4)
    for bx in (x + 12, x + w - 12):
        pl(s, r, [(bx, y + h), (bx, y + h + 22), (bx + (14 if bx < x + w / 2 else -14), y + h)], 1.6, pen, 1.3, over=1)
    s.stroke([(x + w * 0.8, y + h + 2), (x + w * 0.8 + J(r, 3), y + h + U(r, 120, 220))], 10, SLATE, 0.3, dry=0.25, edge=0.4, wob=1.2, taper=(0, 0.7))
    s.ink([(x + w * 0.8, y + h), (x + w * 0.8 + 6, y + h + 40), (x + w * 0.8 + 2, y + h + 90)], 1.2, pen, 0.9)


def wheel(s, r, cx, cy, R, pen=INK):
    """A scooter wheel side on: tyre, alloy rim, five spokes."""
    t = s.ellipse(cx, cy, R, R)
    put(s, t, INK, 0.75, lift=0.8, wob=0.6, edge=0.4, ew=4)
    s.wash(t, PAYNE, 0.6, wob=1, soft=2, edge=0.2, flow=0.5, fs=30)
    rim = s.ellipse(cx, cy, R * 0.62, R * 0.62)
    put(s, rim, MIST, 0.45, lift=1.0, wob=0.5, edge=0.4, ew=3)
    s.wash(rim, SLATE, 0.5, wob=1, soft=2, grad=s.vgrad(cy - R * 0.6, cy + R * 0.6, 0.2, 1))
    for i in range(5):
        a = i * 2 * math.pi / 5 + 0.3
        ho = s.ellipse(cx + math.cos(a + 0.63) * R * 0.38, cy + math.sin(a + 0.63) * R * 0.38, R * 0.15, R * 0.15)
        s.wash(ho, INK, 0.9, wob=0.5, soft=1, edge=0.4, ew=2)
        s.ink([(cx + math.cos(a) * R * 0.15, cy + math.sin(a) * R * 0.15), (cx + math.cos(a) * R * 0.58, cy + math.sin(a) * R * 0.58)], 1.4, pen, 1.0)
    s.wash(s.ellipse(cx, cy, R * 0.16, R * 0.16), SLATE, 0.8, wob=0.4, soft=0.8, edge=0.5)
    oval(s, cx, cy, R, R, 2.4, pen, 1.5)
    oval(s, cx, cy, R * 0.62, R * 0.62, 1.6, pen, 1.2)
    oval(s, cx, cy, R * 0.16, R * 0.16, 1.2, pen, 1.2)
    hl(s, ell(cx, cy, R * 0.85, R * 0.85, 3.6, 4.6, 10), 3.0, 0.25)


# ================================================================================================ the stall
W, H = 3072, 1728
HZ, VPX = 1000.0, 1350.0
G = 1480                              # the shop front's floor line
OPEN = (490, 650, 2510, G)            # the open front
CTOP = 1210                           # the stainless counter's top edge (front)
POT = (1070, 1048)                    # the soy milk pot's mouth
POTRX = 104
RICE = (1252, 1092)
STEAM1 = (1462, 866)                  # steamer stack lids
STEAM2 = (1690, 980)
GRID = (735, 1176)
BOWL = (1078, 1076)                   # the ladle's bowl, in the pot: st_hands' anchor

# the soy milk close (sheet 2560x1440)
SY_BACK = 720                         # the counter top's back edge
SY_TRAY = (600, 1010, 1900, 1390)     # the tray's back-left x, back y, front-right x, front y
SY_CUP = (1110, 1220)                 # the cup's foot
SY_STRAW = (1640, 1300)               # where the straw is laid down (its middle)


def P3(x, y, z):
    """A point on the front plane (x, y), pushed z metres back (negative: toward us)."""
    k = 8.0 / (8.0 + z)
    return (VPX + (x - VPX) * k, HZ + (y - HZ) * k)


def stool(s, r, cx, gy, sc=1.0, c=PERSIMMON, k=0.8, pen=SEPIA, stack=1):
    """A round plastic stool (the kind every stall has), a little from above: a seat with a hand hole, a skirt, splayed
    legs with arched openings between them (the back legs show through); `stack` of them one on another."""
    H_, R_ = 150 * sc, 50 * sc
    s.wash(s.ellipse(cx + 10 * sc, gy + 2, R_ * 1.4, 9 * sc), PAYNE, 0.4, wob=1, soft=4 * sc, edge=0)
    for i in range(stack):
        yb = gy - i * 30 * sc
        top = yb - H_
        sk = top + 26 * sc
        out = Q(s, [(cx - R_ * 0.98, top + 4 * sc), (cx + R_ * 0.98, top + 4 * sc), (cx + R_ * 1.16, yb), (cx - R_ * 1.16, yb)])
        holes = s.zeros()
        for xa, xb in ((-0.86, -0.16), (0.16, 0.86)):
            a, b = cx + xa * R_ * 1.06, cx + xb * R_ * 1.06
            mid = (a + b) / 2
            arch = [(a, yb + 2), (a + (b - a) * 0.02, sk + 26 * sc), (mid - (b - a) * 0.3, sk + 10 * sc), (mid, sk + 7 * sc), (mid + (b - a) * 0.3, sk + 10 * sc), (b - (b - a) * 0.02, sk + 26 * sc), (b, yb + 2)]
            holes = np.maximum(holes, s.poly(arch))
        if i == stack - 1 or True:
            # the back legs, seen through the openings
            for q in (-0.5, 0.5):
                bl = Q(s, [(cx + q * R_ - 5 * sc, sk + 12 * sc), (cx + q * R_ + 5 * sc, sk + 12 * sc), (cx + q * R_ * 1.25 + 6 * sc, yb - 8 * sc), (cx + q * R_ * 1.25 - 6 * sc, yb - 8 * sc)])
                s.wash(bl * holes, c, k * 0.9, wob=0.4, soft=0.7, edge=0.4, ew=2, body=1)
                s.wash(bl * holes, UMBER, 0.45, wob=0.4, soft=0.7)
        body = np.clip(out - holes, 0, 1)
        put(s, body, c, k, lift=0.95, wob=0.6, edge=0.5, ew=3, flow=0.5, fs=40)
        s.wash(body, UMBER, 0.32, wob=1, soft=2, grad=s.hgrad(cx - R_ * 0.2, cx + R_ * 1.1, 0.0, 1))
        s.wash(s.rect(cx - R_, top + 4 * sc, cx + R_, sk), UMBER, 0.18, wob=0.5, soft=1, clip=body)
        hl(s, [(cx - R_ * 0.7, sk + 6 * sc), (cx - R_ * 0.82, yb - 8 * sc)], 3 * sc, 0.3, clip=body)
        seat = s.ellipse(cx, top + 4 * sc, R_, 14 * sc)
        put(s, seat, c, k * 0.8, lift=0.95, wob=0.5, edge=0.5, ew=3)
        s.lift(s.ellipse(cx - R_ * 0.3, top + 1 * sc, R_ * 0.45, 5 * sc), 0.35, soft=2, wob=0.5)
        s.wash(s.ellipse(cx, top + 4 * sc, 14 * sc, 4 * sc), UMBER, 0.85, wob=0.4, soft=0.8, edge=0.5)            # the hand hole
        oval(s, cx, top + 4 * sc, R_, 14 * sc, 1.5, pen, 1.3)
        contour(s, body, 1.4, pen, 1.2)
        line(s, r, (cx - R_ * 0.98, sk), (cx + R_ * 0.98, sk + 1), 1.0, pen, 0.8, over=0, skip=0.2)


def stall_bg():
    S = Sheet(W, H, 301)
    r = rs(301)
    left_house(S, r)
    upper_facade(S, r)
    right_house(S, r)
    interior(S, r)
    front(S, r)
    counter(S, r)
    lane(S, r)
    outside(S, r)
    scooter(S, r)
    light(S, r)
    meta = {"pot": list(POT), "steamers": [list(STEAM1), list(STEAM2), [1570, 1036], [1398, 998], [1774, 1062]], "griddle": list(GRID),
            "counter_y": CTOP, "rice": list(RICE), "opening": list(OPEN), "lane_y": 1530, "boss": list(BOWL)}
    S.save("st_bg", opaque=True, q=90, meta=meta)


def left_house(S, r):
    """The neighbour on the left: two storeys of brown 二丁掛 tile, a water tank on the roof against the morning sky,
    a closed shop under it (its shutter still down)."""
    s = Sub(S, 0, 0, 470, G + 30)
    # sky
    sk = s.rect(-10, -10, 480, 190)
    s.wash(sk, RAIN, 0.42, wob=0, soft=0, edge=0, grad=s.vgrad(-40, 200, 1, 0.35), flow=0.4, fs=200, gran=0.15)
    s.wash(s.blob(160, 70, 240, 40, irr=0.4), None, 0.3, wob=6, bleed=20, edge=0, lift=True)
    s.wash(s.blob(330, 30, 180, 30, irr=0.4), MIST, 0.2, wob=6, bleed=16, edge=0.2)
    # the roof: parapet, a stainless tank on a stand, a tin shed
    tm = s.rect(150, 46, 300, 142)
    put(s, tm, MIST, 0.4, lift=0.9, wob=0.8, edge=0.4, ew=3)
    s.wash(tm, SLATE, 0.5, wob=1, soft=2, grad=s.hgrad(170, 300, 0, 1))
    s.lift(s.rect(176, 52, 196, 136), 0.5, soft=2)
    for yy in (70, 100, 128):
        line(s, r, (150, yy), (300, yy + J(r, 1)), 1.1, INKBLUE, 0.9, over=1, skip=0.15)
    s.wash(s.ellipse(225, 46, 75, 9), SLATE, 0.5, wob=0.5, soft=1)
    oval(s, 225, 46, 75, 9, 1.4, INKBLUE, 1.2)
    box(s, r, 150, 46, 300, 142, 1.8, INKBLUE, 1.3, over=3)
    for xx in (160, 290):
        line(s, r, (xx, 142), (xx + J(r, 2), 178), 1.6, INKBLUE, 1.3, over=1)
    line(s, r, (160, 160), (290, 165), 1.1, INKBLUE, 1.0)
    shed = Q(s, [(310, 130), (470, 118), (470, 180), (310, 180)])
    put(s, shed, SLATE, 0.45, lift=0.6, wob=0.8, edge=0.4, ew=3)
    s.wash(shed, BURNT, 0.2, wob=2, bleed=6, flow=0.8, fs=20)
    for xx in np.arange(318, 470, 13):
        seg(s, (xx, 129 - (xx - 310) * 0.075), (xx, 180), 0.9, INKBLUE, 0.7, r=r, skip=0.2)
    line(s, r, (306, 131), (474, 117), 2.0, INKBLUE, 1.3)
    # the facade
    fm = s.rect(-10, 172, 404, G + 10)
    s.wash(fm, SIENNA, 0.42, wob=1.5, soft=0.9, edge=0.4, ew=8, flow=0.5, fs=200, gran=0.45, body=1)
    s.wash(fm, BURNT, 0.25, wob=3, bleed=12, edge=0.15, flow=0.6, fs=150, grad=s.vgrad(172, 700, 1, 0.3), clip=fm)
    mottle(s, r, 0, 200, 400, 1300, [UMBER, VIOLET, OCHRE], 5, 0.12, clip=fm)
    s.wash(s.rect(-10, 172, 410, 190), SLATE, 0.6, wob=0.8, soft=0.8, edge=0.4, ew=3, body=1)    # parapet coping
    line(s, r, (-5, 172), (412, 171), 2.2, SEPIA, 1.5, dbl=0.5)
    line(s, r, (-5, 190), (410, 190), 1.4, SEPIA, 1.2, skip=0.1)
    # 二丁掛: long brick-shaped tiles in courses
    for _ in range(9):
        cx, cy = U(r, 20, 380), U(r, 210, 1180)
        for j in range(int(r.integers(2, 5))):
            y = 200 + (round((cy - 200) / 16) + j) * 16
            seg(s, (cx - U(r, 40, 90), y), (cx + U(r, 40, 90), y + J(r, 0.6)), 0.9, SEPIA, 0.5, r=r, skip=0.25)
            off = 25 if (round((cy - 200) / 16) + j) % 2 else 0
            for i in range(-2, 3):
                xx = cx + i * 50 + off
                if r.random() < 0.7:
                    seg(s, (xx, y + 1), (xx, y + 15), 0.8, SEPIA, 0.5, r=r)
    grille(s, r, 60, 236, 360, 420, under=30, c=MIST, pen=INKBLUE, gap=20, seed=11)
    streaks(s, r, 20, 390, 450, 560, 5, UMBER, 0.2, (6, 14), clip=fm)
    # canopy slab over the shop
    cs = s.rect(-10, 540, 410, 566)
    put(s, cs, MIST, 0.45, lift=0.85, wob=0.8, edge=0.4, ew=3)
    s.wash(s.rect(-10, 566, 404, 600), SLATE, 0.45, wob=0, soft=0, edge=0, grad=s.vgrad(566, 600, 1, 0))
    line(s, r, (-5, 540), (412, 540), 1.8, SEPIA, 1.4, dbl=0.4)
    line(s, r, (-5, 566), (412, 567), 2.2, SEPIA, 1.5, dbl=0.4)
    # a faded sign board and a rolled-down shutter
    sb = s.rect(20, 604, 380, 680)
    put(s, sb, CREAM, 0.3, lift=0.85, wob=0.8, edge=0.4, ew=4)
    s.wash(sb, TEAL, 0.35, wob=1.2, soft=1, edge=0.5, ew=3, grad=s.hgrad(20, 380, 1, 0.6))
    s.lift(letters(s, "文具 · 影印", 200, 642, 44, SERIF, 800), 0.75, soft=0.6, wob=0.4)
    box(s, r, 20, 604, 380, 680, 1.8, SEPIA, 1.4, over=3, dbl=0.4)
    sh = s.rect(30, 720, 384, G - 4)
    put(s, sh, SLATE, 0.55, lift=0.85, wob=1, edge=0.4, ew=5, flow=0.5, fs=90, gran=0.45)
    s.wash(sh, CELADON, 0.3, wob=2, bleed=10, edge=0.1)
    s.wash(sh, PAYNE, 0.3, wob=1, soft=2, edge=0, grad=s.vgrad(720, G, 0.2, 1))
    y = 732
    while y < G - 10:
        seg(s, (32 + J(r, 1), y + J(r, 0.6)), (382 + J(r, 1), y + J(r, 0.6)), U(r, 0.9, 1.4), SEPIA, U(r, 0.5, 0.9), r=r, skip=0.12, bow=0.002)
        if r.random() < 0.25:
            a = U(r, 40, 200)
            hl(s, [(a, y - 6), (a + U(r, 60, 160), y - 6)], 2.2, U(r, 0.12, 0.22))
        y += 12 * U(r, 0.92, 1.08)
    s.wash(s.rect(26, 690, 388, 722), SLATE, 0.65, wob=0.8, soft=0.8, edge=0.4, ew=3, body=1)
    box(s, r, 26, 690, 388, 722, 1.8, SEPIA, 1.4, over=3)
    box(s, r, 30, 722, 384, G - 4, 2.0, SEPIA, 1.4, over=4, dbl=0.4)
    s.wash(s.rect(180, G - 30, 230, G - 8), SEPIA, 0.6, wob=0.5, soft=0.8)                       # the lock
    # a tag of a sticker or two and a rental notice taped on the shutter
    nm = Q(s, [(250, 960), (330, 956), (334, 1060), (254, 1064)])
    put(s, nm, CREAM, 0.15, lift=0.9, wob=0.6, edge=0.3)
    paint_letters(s, letters(s, "出租", 292, 990, 30, SERIF, 800), PERSIMMON, 1.3, wob=0.3)
    for q in range(3):
        line(s, r, (262, 1022 + q * 12), (322, 1021 + q * 12), 1.0, SEPIA, 0.7, over=0, skip=0.3)
    path(s, [(250, 960), (330, 956), (334, 1060), (254, 1064)], 1.2, SEPIA, 1.0, closed=True, over=1, r=r)
    line(s, r, (404, 172), (404, G), 2.2, SEPIA, 1.5, dbl=0.4)


def upper_facade(S, r):
    """Our building above the shop: pale green-grey square tiles, two iron window cages, an air-conditioner."""
    s = Sub(S, 380, 0, 2640, 500)
    fm = s.rect(396, -10, 2604, 478)
    s.wash(fm, TILE, 0.5, wob=1.2, soft=0.9, edge=0.3, ew=8, flow=0.5, fs=260, gran=0.42, body=1)
    s.wash(fm, CELADON, 0.25, wob=3, bleed=12, edge=0.1, flow=0.6, fs=200, clip=fm)
    mottle(s, r, 420, 0, 2580, 470, [SLATE, CELADON, OCHRE, MIST], 8, 0.14, size=(80, 200), clip=fm)
    drags(s, r, 420, 10, 2580, 460, SLATE, 8, clip=fm)
    tiles(s, r, 400, -6, 2600, 470, 21.0, SEPIA, 0.5, patches=14, size=((80, 260), (50, 160)), singles=120, tints=(SLATE, CELADON, TILE), clip=fm)
    grille(s, r, 560, 20, 1240, 360, under=40, pen=INKBLUE, gap=23, seed=21)
    grille(s, r, 1470, 20, 2150, 360, under=36, pen=INKBLUE, gap=23, seed=22)
    # a towel and a T-shirt drying in the right cage
    for lx, lw, lh, c in ((1600, 70, 90, ROSE), (1700, 90, 110, MIST), (1820, 60, 70, OCHRE)):
        cm = Q(s, [(lx, 92), (lx + lw, 94), (lx + lw - 4 + J(r, 3), 92 + lh), (lx + 3 + J(r, 3), 92 + lh + J(r, 4))])
        s.wash(cm, c, 0.4, wob=1, soft=0.8, edge=0.5, ew=3, body=0.8)
        s.wash(cm, SLATE, 0.35, wob=1, soft=2, edge=0, grad=s.hgrad(lx, lx + lw, 0.2, 1))
        s.ink(spline([(lx, 92), (lx + 2, 92 + lh), (lx + lw - 3, 92 + lh), (lx + lw, 94)]), 1.1, SEPIA, 1.0, smooth=False, skip=0.1)
    s.ink([(1560, 92), (1760, 96), (1960, 91)], 1.2, INKBLUE, 1.1)
    acunit(s, r, 2290, 200, 170, 110)
    acunit(s, r, 1280, 250, 140, 92)
    # a downpipe and a bundle of conduit
    for x in (2560, 2574):
        s.ink([(x, -10), (x + J(r, 1), 240), (x, 478)], 2.0, INKBLUE, 1.2)
    hl(s, [(2566, 0), (2566, 470)], 4, 0.3)
    for yy in (90, 230, 380):
        line(s, r, (2552, yy), (2584, yy + 1), 2.0, INKBLUE, 1.3, over=1)
    streaks(s, r, 420, 2580, 360, 470, 14, SLATE, 0.18, (6, 14), clip=fm)
    line(s, r, (2604, -10), (2604, 478), 2.2, SEPIA, 1.5, dbl=0.4)


def right_house(S, r):
    """The neighbour on the right: pale pink tile, a window with a cage, and at street level the stair door of the
    flats upstairs: a red iron door, a row of letterboxes, a plant."""
    s = Sub(S, 2590, 0, W, G + 30)
    fm = s.rect(2604, -10, W + 10, G + 10)
    s.wash(fm, CREAM, 0.5, wob=1.2, soft=0.9, edge=0.3, ew=8, flow=0.5, fs=220, gran=0.4, body=1)
    s.wash(fm, ROSE, 0.22, wob=3, bleed=12, edge=0.1, flow=0.6, fs=180, clip=fm)
    mottle(s, r, 2620, 0, 3060, 1400, [OCHRE, SLATE, ROSE], 6, 0.14, clip=fm)
    tiles(s, r, 2604, -6, W, 1440, 30.0, SEPIA, 0.45, patches=8, size=((80, 200), (60, 160)), singles=40, tints=(ROSE, OCHRE, SLATE), clip=fm)
    grille(s, r, 2700, 60, 3040, 380, under=34, pen=INKBLUE, gap=22, seed=31)
    # a slab and a little tiled canopy over the door
    cs = s.rect(2604, 540, W + 10, 566)
    put(s, cs, MIST, 0.45, lift=0.85, wob=0.8, edge=0.4, ew=3)
    s.wash(s.rect(2604, 566, W + 10, 600), SLATE, 0.45, wob=0, soft=0, edge=0, grad=s.vgrad(566, 600, 1, 0))
    line(s, r, (2600, 540), (W, 540), 1.8, SEPIA, 1.4, dbl=0.4)
    line(s, r, (2600, 566), (W, 567), 2.2, SEPIA, 1.5, dbl=0.4)
    # the stair door: red iron, a grille in its top half, the stairs' light behind
    dx0, dy0, dx1, dy1 = 2690, 800, 2930, G - 6
    dm = s.rect(dx0, dy0, dx1, dy1)
    put(s, dm, PERSIMMON, 0.62, lift=0.9, wob=1, edge=0.5, ew=5, flow=0.5, fs=50)
    s.wash(dm, BURNT, 0.4, wob=1, soft=2, edge=0, grad=s.vgrad(dy0, dy1, 0.3, 1))
    gm = s.rect(dx0 + 22, dy0 + 30, dx1 - 22, dy0 + 300)
    s.wash(gm, SLATE, 0.7, wob=0.6, soft=1, edge=0.5, ew=3)
    s.wash(s.rect(dx0 + 22, dy0 + 30, dx1 - 22, dy0 + 140), OCHRE, 0.3, wob=1, soft=3)
    xx = dx0 + 30
    while xx < dx1 - 24:
        seg(s, (xx, dy0 + 30), (xx + J(r, 0.5), dy0 + 300), 1.4, SEPIA, 1.2, r=r)
        xx += 18
    for yy in (dy0 + 100, dy0 + 200):
        line(s, r, (dx0 + 22, yy), (dx1 - 22, yy), 1.4, SEPIA, 1.2, over=1)
    box(s, r, dx0 + 22, dy0 + 30, dx1 - 22, dy0 + 300, 1.6, SEPIA, 1.3, over=2)
    for yy in (dy0 + 340, dy0 + 500):
        box(s, r, dx0 + 24, yy, dx1 - 24, yy + 130, 1.4, SEPIA, 1.2, over=2, dbl=0.3)
        hl(s, [(dx0 + 28, yy + 4), (dx1 - 28, yy + 4)], 2, 0.3)
    s.wash(s.ellipse(dx1 - 30, dy0 + 420, 7, 7), OCHRE, 0.5, wob=0.3, soft=0.6)
    box(s, r, dx0, dy0, dx1, dy1, 2.2, SEPIA, 1.5, over=5, dbl=0.5)
    box(s, r, dx0 - 14, dy0 - 16, dx1 + 14, dy1 + 2, 1.6, SEPIA, 1.3, over=4, dbl=0.3)
    # the number plate, the letterboxes
    pm = s.rect(2760, 742, 2860, 782)
    put(s, pm, TEAL, 0.6, lift=0.9, wob=0.6, edge=0.5, ew=2)
    s.lift(letters(s, "12號", 2810, 762, 28, SERIF, 800), 0.9, soft=0.5, wob=0.3)
    box(s, r, 2760, 742, 2860, 782, 1.4, SEPIA, 1.3, over=2)
    for i in range(3):
        for j in range(2):
            bx, by = 2950 + i * 40, 900 + j * 52
            m = s.rect(bx, by, bx + 34, by + 44)
            put(s, m, (SLATE, CELADON, MIST)[(i + j) % 3], 0.5, lift=0.8, wob=0.5, edge=0.5, ew=2)
            line(s, r, (bx + 6, by + 12), (bx + 28, by + 12), 1.2, SEPIA, 1.1, over=0)
            box(s, r, bx, by, bx + 34, by + 44, 1.3, SEPIA, 1.2, over=1)
    # a potted plant beside the door
    pot = [(2948, 1380), (3020, 1380), (3010, G), (2958, G)]
    put(s, Q(s, pot), TERRA, 0.7, lift=0.85, wob=0.8, edge=0.5, ew=3)
    s.wash(Q(s, pot), UMBER, 0.35, wob=1, soft=2, grad=s.hgrad(2950, 3020, 0.1, 1))
    path(s, pot, 1.7, SEPIA, 1.4, closed=True, over=2, r=r)
    for _ in range(22):
        a = U(r, -2.9, -0.25)
        L = U(r, 60, 170)
        b0 = (2984 + J(r, 18), 1382)
        tip = (b0[0] + math.cos(a) * L, b0[1] + math.sin(a) * L)
        s.stroke([b0, ((b0[0] + tip[0]) / 2 + J(r, 8), (b0[1] + tip[1]) / 2), tip], U(r, 9, 18), (MOSS, LEAF, TEAL)[int(r.integers(0, 3))],
                 U(r, 0.5, 0.9), taper=(0.1, 0.8), wob=1, edge=0.5, body=0.9)
        if r.random() < 0.5:
            s.ink([b0, tip], 1.0, SEPIA, 0.8, skip=0.2)
    s.wash(s.rect(2604, G - 60, W, G + 10), SEPIA, 0.25, wob=3, bleed=10, edge=0.1, clip=fm)


def interior(S, r):
    """Inside the open front: back wall, ceiling and its tube light, the back counter, the seating corner."""
    x0, y0, x1, y1 = OPEN
    s = Sub(S, x0 - 4, y0 - 4, x1 + 4, y1 + 4)
    B = (P3(x0, y0, 3.5), P3(x1, y1, 3.5))
    bx0, by0 = B[0]
    bx1, by1 = B[1]
    om = s.rect(x0, y0, x1, y1)
    s.lift(om, 1.0, soft=0.5, wob=0)
    s.wash(om, CREAM, 0.35, wob=0, soft=0.5, edge=0, flow=0.5, fs=200, body=1)
    # ceiling, side walls, floor: the planes of a small room
    ceil = Q(s, [(x0, y0), (x1, y0), (bx1, by0), (bx0, by0)])
    s.wash(ceil, SEPIA, 0.35, wob=1, soft=1, edge=0.3, ew=6, flow=0.5, fs=90, grad=s.vgrad(y0, by0, 1, 0.4))
    lw = Q(s, [(x0, y0), (bx0, by0), (bx0, by1), (x0, y1)])
    s.wash(lw, TILE, 0.45, wob=1, soft=1, edge=0.3, ew=6, flow=0.5, fs=80)
    s.wash(lw, UMBER, 0.25, wob=2, bleed=8, edge=0.1, grad=s.hgrad(x0, bx0, 1, 0.2))
    rw = Q(s, [(x1, y0), (bx1, by0), (bx1, by1), (x1, y1)])
    s.wash(rw, CELADON, 0.42, wob=1, soft=1, edge=0.3, ew=6, flow=0.5, fs=80)
    s.wash(rw, UMBER, 0.2, wob=2, bleed=8, edge=0.1, grad=s.hgrad(bx1, x1, 0.2, 1))
    fl = Q(s, [(bx0, by1), (bx1, by1), (x1, y1), (x0, y1)])
    s.wash(fl, SIENNA, 0.3, wob=1, soft=1, edge=0.3, ew=6, flow=0.5, fs=60, gran=0.5)
    s.wash(fl, SLATE, 0.2, wob=2, bleed=8, grad=s.vgrad(by1, y1, 0.2, 1))
    for z in np.arange(0.3, 3.5, 0.32):                       # floor tile joints
        a, b = P3(x0, y1, z), P3(x1, y1, z)
        seg(s, a, b, 0.9, SEPIA, 0.45, r=r, skip=0.3, bow=0.001)
    for xx in np.arange(x0, x1 + 1, 92):
        seg(s, (xx, y1), P3(xx, y1, 3.5), 0.9, SEPIA, 0.45, r=r, skip=0.3, bow=0.001)
    # back wall: white tile to shoulder height, plaster above
    bw = s.rect(bx0, by0, bx1, by1)
    s.wash(bw, CREAM, 0.3, wob=1, soft=1, edge=0.3, ew=5, flow=0.5, fs=90)
    tl = 1000 + (1480 - 1.4 * 300 - 1000) * 8 / 11.5
    s.wash(s.rect(bx0, tl, bx1, by1), MIST, 0.32, wob=1, soft=1, edge=0.4, ew=3)
    tiles(s, r, bx0, tl, bx1, by1, 15.0, SEPIA, 0.35, patches=6, size=((60, 160), (30, 80)), singles=20, tints=(MIST, CELADON))
    line(s, r, (bx0, tl), (bx1, tl), 1.2, SEPIA, 0.9, skip=0.15)
    path(s, [(x0, y0), (bx0, by0), (bx1, by0), (x1, y0)], 1.3, SEPIA, 0.9, over=2, r=r)
    path(s, [(bx0, by0), (bx0, by1), (bx1, by1), (bx1, by0)], 1.3, SEPIA, 0.9, over=2, r=r)
    path(s, [(x0, y1), (bx0, by1)], 1.2, SEPIA, 0.8, over=2, r=r)
    path(s, [(x1, y1), (bx1, by1)], 1.2, SEPIA, 0.8, over=2, r=r)
    # the tube light and its glow; a ceiling fan
    ta, tb = P3(1000, 650, 1.6), P3(1600, 650, 1.6)
    ta, tb = (ta[0], ta[1] + 10), (tb[0], tb[1] + 10)
    gl = s.rgrad((ta[0] + tb[0]) / 2, ta[1] + 40, 60, 520, 1, 0, sy=0.5)
    s.wash(gl, None, 0.5, wob=0, soft=0, edge=0, gran=0.1, flow=0.3, fs=120, lift=True)
    s.wash(gl, OCHRE, 0.25, wob=0, soft=0, edge=0, flow=0.4, fs=100)
    tm = s.ribbon([ta, tb], 9, smooth=False)
    s.lift(tm, 1.0, soft=0.8, wob=0.4)
    s.wash(s.ribbon([(ta[0] - 6, ta[1] - 7), (tb[0] + 6, tb[1] - 7)], 7, smooth=False), MIST, 0.6, wob=0.5, soft=0.6, edge=0.4)
    line(s, r, (ta[0] - 6, ta[1] - 11), (tb[0] + 6, tb[1] - 11), 1.4, SEPIA, 1.1, over=2)
    line(s, r, (ta[0], ta[1] + 5), (tb[0], tb[1] + 5), 1.0, SEPIA, 0.7, over=1, skip=0.3)
    fx, fy = P3(2050, 650, 1.2)
    s.ink([(fx, fy - 4), (fx, fy + 16)], 2.0, SEPIA, 1.3)
    for a in (0.15, 2.25, 4.3):
        blade = s.ellipse(fx + math.cos(a) * 52, fy + 20 + math.sin(a) * 6, 52, 7, math.degrees(math.atan2(math.sin(a) * 6, math.cos(a) * 52)))
        s.wash(blade, CREAM, 0.5, wob=0.5, soft=0.7, edge=0.5, ew=2, body=1)
        contour(s, blade, 1.2, SEPIA, 1.0)
    s.wash(s.ellipse(fx, fy + 20, 14, 7), SLATE, 0.7, wob=0.4, soft=0.7)
    oval(s, fx, fy + 20, 14, 7, 1.2, SEPIA, 1.2)
    # back wall things: the god's shelf with its two red lamps, a clock, a tear-off calendar, a shelf of jars
    gx, gy = bx0 + 70, by0 + 40
    gs = s.rect(gx, gy, gx + 120, gy + 52)
    put(s, gs, PERSIMMON, 0.65, lift=0.8, wob=0.6, edge=0.5, ew=3)
    s.wash(gs, BURNT, 0.4, wob=1, soft=1.5, grad=s.vgrad(gy, gy + 52, 0.2, 1))
    for lx in (gx + 22, gx + 98):
        g2 = s.rgrad(lx, gy + 18, 3, 30, 1, 0)
        s.wash(g2, ROSE, 0.5, wob=0, soft=0, edge=0)
        s.lift(s.ellipse(lx, gy + 18, 5, 7), 0.8, soft=1)
        s.wash(s.ellipse(lx, gy + 18, 5, 7), PERSIMMON, 0.35, wob=0.3, soft=0.6)
    s.wash(s.rect(gx + 46, gy + 8, gx + 74, gy + 44), OCHRE, 0.5, wob=0.4, soft=0.6, edge=0.5)
    box(s, r, gx, gy, gx + 120, gy + 52, 1.4, SEPIA, 1.2, over=3)
    line(s, r, (gx - 8, gy + 52), (gx + 128, gy + 52), 2.0, SEPIA, 1.3)
    cx, cy = bx0 + 470, by0 + 62
    put(s, s.ellipse(cx, cy, 26, 26), CREAM, 0.15, lift=0.9, wob=0.4, edge=0.3)
    oval(s, cx, cy, 26, 26, 1.8, SEPIA, 1.4)
    s.ink([(cx, cy), (cx - 2, cy - 17)], 1.6, SEPIA, 1.4)
    s.ink([(cx, cy), (cx + 12, cy + 4)], 1.6, SEPIA, 1.4)
    kx, ky = bx0 + 610, by0 + 26
    cal = s.rect(kx, ky, kx + 66, ky + 92)
    put(s, cal, CREAM, 0.1, lift=0.95, wob=0.5, edge=0.3)
    s.wash(s.rect(kx, ky, kx + 66, ky + 18), PERSIMMON, 0.8, wob=0.4, soft=0.6, edge=0.5)
    paint_letters(s, letters(s, "16", kx + 33, ky + 54, 40, SERIF, 900), PERSIMMON, 1.4, wob=0.2)
    box(s, r, kx, ky, kx + 66, ky + 92, 1.2, SEPIA, 1.1, over=2)
    sy_ = by0 + 168
    put(s, s.rect(bx0 + 4, sy_, bx0 + 820, sy_ + 9), SIENNA, 0.6, lift=0.6, wob=0.6, edge=0.4)
    line(s, r, (bx0 + 2, sy_ + 9), (bx0 + 822, sy_ + 9), 1.6, SEPIA, 1.3)
    xx = bx0 + 14
    while xx < bx0 + 800:
        kind = r.random()
        if kind < 0.4:      # a jar
            w_, h_ = U(r, 22, 34), U(r, 30, 48)
            jm = s.rect(xx, sy_ - h_, xx + w_, sy_)
            put(s, jm, (OCHRE, BURNT, UMBER, CELADON)[int(r.integers(0, 4))], U(r, 0.35, 0.7), lift=0.7, wob=0.5, edge=0.5, ew=2)
            s.wash(s.rect(xx - 1, sy_ - h_ - 6, xx + w_ + 1, sy_ - h_), PERSIMMON, 0.6, wob=0.3, soft=0.6)
            s.lift(s.rect(xx + 4, sy_ - h_ + 4, xx + 8, sy_ - 4), 0.4, soft=1)
            box(s, r, xx, sy_ - h_ - 6, xx + w_, sy_, 1.1, SEPIA, 1.1, over=1)
        elif kind < 0.7:    # a stack of bowls or cups
            w_, h_ = U(r, 30, 46), U(r, 26, 60)
            jm = s.rect(xx, sy_ - h_, xx + w_, sy_)
            put(s, jm, MIST, 0.3, lift=0.8, wob=0.5, edge=0.4, ew=2)
            for q in np.arange(sy_ - h_ + 8, sy_, 8):
                line(s, r, (xx, q), (xx + w_, q), 0.8, SEPIA, 0.7, over=0, skip=0.2)
            box(s, r, xx, sy_ - h_, xx + w_, sy_, 1.1, SEPIA, 1.0, over=1)
        else:               # a bottle
            w_, h_ = U(r, 12, 18), U(r, 44, 64)
            bm = np.maximum(s.rect(xx, sy_ - h_ * 0.7, xx + w_, sy_), s.rect(xx + w_ * 0.3, sy_ - h_, xx + w_ * 0.7, sy_ - h_ * 0.7))
            put(s, bm, (UMBER, LEAF, BURNT)[int(r.integers(0, 3))], U(r, 0.5, 0.9), lift=0.6, wob=0.4, edge=0.5, ew=2)
            contour(s, bm, 1.0, SEPIA, 1.0)
        xx += w_ + U(r, 4, 18)
    # the drinks cooler against the back wall, right
    fx0, fy0, fx1, fy1 = bx1 - 300, by1 - 430, bx1 - 70, by1
    fm = s.rect(fx0, fy0, fx1, fy1)
    put(s, fm, MIST, 0.4, lift=0.9, wob=0.6, edge=0.4, ew=3)
    gm = s.rect(fx0 + 14, fy0 + 52, fx1 - 14, fy1 - 40)
    s.wash(gm, CELADON, 0.4, wob=0.6, soft=1, edge=0.5, ew=3)
    s.lift(s.rect(fx0 + 12, fy0 + 10, fx1 - 12, fy0 + 44), 0.9, soft=0.8)
    s.wash(s.rect(fx0 + 12, fy0 + 10, fx1 - 12, fy0 + 44), PERSIMMON, 0.35, wob=0.4, soft=0.6)
    for j in range(5):
        ly = fy0 + 52 + (fy1 - 40 - fy0 - 52) * (j + 1) / 5.4
        line(s, r, (fx0 + 14, ly), (fx1 - 14, ly), 1.1, SEPIA, 0.9, over=0)
        xx = fx0 + 20
        while xx < fx1 - 26:
            bh = U(r, 30, 52)
            bmm = s.rect(xx, ly - bh, xx + 13, ly - 2)
            s.wash(bmm, (OCHRE, PERSIMMON, CREAM, LEAF, ROSE, TEAL)[int(r.integers(0, 6))], U(r, 0.4, 0.8), wob=0.4, soft=0.6, edge=0.5, ew=2)
            s.wash(s.rect(xx + 2, ly - bh - 6, xx + 11, ly - bh), SLATE, 0.5, wob=0.3, soft=0.5)
            xx += U(r, 15, 21)
    s.lift(Q(s, [(fx0 + 20, fy0 + 60), (fx0 + 70, fy0 + 60), (fx0 + 20, fy0 + 220)]), 0.35, soft=4, wob=2)
    box(s, r, fx0, fy0, fx1, fy1, 2.0, SEPIA, 1.4, over=3, dbl=0.4)
    box(s, r, fx0 + 14, fy0 + 52, fx1 - 14, fy1 - 40, 1.3, SEPIA, 1.1, over=2)
    line(s, r, (fx1 - 30, fy0 + 160), (fx1 - 30, fy0 + 260), 2.6, SEPIA, 1.3, over=0)
    # the back counter (left): rice cooker, the cup sealer, egg trays; mostly hidden by the front counter
    bt = P3(x0, 1480 - 0.9 * 300, 2.6)[1]
    bc = s.rect(bx0 - 20, bt, bx0 + 860, by1 + 40)
    put(s, bc, MIST, 0.35, lift=0.7, wob=0.8, edge=0.4, ew=3)
    s.wash(bc, SLATE, 0.3, wob=1, soft=2, grad=s.vgrad(bt, by1, 0.1, 1))
    line(s, r, (bx0 - 20, bt), (bx0 + 860, bt), 1.8, SEPIA, 1.3, dbl=0.4)
    rc = (bx0 + 70, bt)       # 大同電鍋
    rcm = np.maximum(s.rect(rc[0] - 50, rc[1] - 80, rc[0] + 50, rc[1]), s.ellipse(rc[0], rc[1] - 80, 50, 14))
    put(s, rcm, CELADON, 0.85, lift=0.9, wob=0.6, edge=0.5, ew=3)
    s.wash(rcm, LEAF, 0.35, wob=1, soft=2, grad=s.hgrad(rc[0] - 50, rc[0] + 50, 0, 1))
    s.wash(s.ellipse(rc[0], rc[1] - 90, 36, 9), MIST, 0.4, wob=0.4, soft=0.8)
    s.wash(s.rect(rc[0] - 12, rc[1] - 108, rc[0] + 12, rc[1] - 94), SEPIA, 0.8, wob=0.3, soft=0.6)
    s.wash(s.rect(rc[0] - 10, rc[1] - 40, rc[0] + 10, rc[1] - 24), PERSIMMON, 0.8, wob=0.3, soft=0.6)
    contour(s, rcm, 1.6, SEPIA, 1.3)
    sm = (bx0 + 380, bt)      # the cup sealer
    smm = s.rect(sm[0] - 40, sm[1] - 110, sm[0] + 40, sm[1])
    put(s, smm, SLATE, 0.6, lift=0.85, wob=0.5, edge=0.5, ew=3)
    s.wash(s.rect(sm[0] - 30, sm[1] - 60, sm[0] + 30, sm[1] - 36), INK, 0.6, wob=0.3, soft=0.6)
    s.ink([(sm[0] + 30, sm[1] - 108), (sm[0] + 70, sm[1] - 140), (sm[0] + 76, sm[1] - 136)], 3.0, SEPIA, 1.4)
    box(s, r, sm[0] - 40, sm[1] - 110, sm[0] + 40, sm[1], 1.5, SEPIA, 1.3, over=2)
    for i in range(4):        # egg trays
        ey = bt - 8 - i * 9
        put(s, s.rect(bx0 + 520, ey - 8, bx0 + 640, ey), CREAM, 0.4, lift=0.6, wob=0.4, edge=0.4, ew=2)
        for q in range(6):
            s.wash(s.ellipse(bx0 + 530 + q * 20, ey - 8, 7, 4), OCHRE, 0.4, wob=0.3, soft=0.5)
        line(s, r, (bx0 + 520, ey), (bx0 + 640, ey), 0.9, SEPIA, 0.9, over=0)
    # the seating corner (right): two tables and stools
    for tz, txa, txb in ((2.4, 2120, 2420), (1.1, 2160, 2470)):
        k = 8.0 / (8 + tz)
        ty = HZ + (1480 - 225 - HZ) * k
        gy = HZ + (1480 - HZ) * k
        xa, xb = VPX + (txa - VPX) * k, VPX + (txb - VPX) * k
        for sx_ in (xa + (xb - xa) * 0.15, xb - (xb - xa) * 0.12):
            stool(s, r, sx_, gy + 8, 0.55 * k / 0.8, (PERSIMMON, TEAL)[int(r.integers(0, 2))], 0.7)
        tm = s.rect(xa, ty - 6, xb, ty + 6)
        put(s, tm, MIST, 0.4, lift=0.9, wob=0.5, edge=0.4, ew=2)
        s.wash(Q(s, [(xa, ty - 6), (xb, ty - 6), (xb - 12, ty - 18), (xa + 12, ty - 18)]), CREAM, 0.25, wob=0.4, soft=0.8, body=1)
        for lx in (xa + 12, xb - 12):
            s.ink([(lx, ty + 6), (lx + J(r, 2), gy)], 2.6 * k, SEPIA, 1.3)
        box(s, r, xa, ty - 6, xb, ty + 6, 1.4, SEPIA, 1.3, over=2)
        line(s, r, (xa + 12, ty - 18), (xb - 12, ty - 18), 1.1, SEPIA, 1.0, over=2, skip=0.15)
        s.wash(s.rect(xa + 8, ty + 6, xb - 8, ty + 30), UMBER, 0.3, wob=0, soft=0, edge=0, grad=s.vgrad(ty + 6, ty + 30, 1, 0))
        # a newspaper, a chilli jar, a chopstick tub
        np0 = xa + (xb - xa) * 0.25
        s.wash(Q(s, [(np0, ty - 9), (np0 + 70 * k, ty - 10), (np0 + 76 * k, ty - 17), (np0 + 4, ty - 16)]), MIST, 0.3, wob=0.4, soft=0.6, body=1)
        s.wash(s.rect(xb - 60 * k, ty - 34 * k, xb - 44 * k, ty - 8), PERSIMMON, 0.8, wob=0.3, soft=0.6, body=1)
        s.wash(s.rect(xb - 92 * k, ty - 40 * k, xb - 74 * k, ty - 8), SIENNA, 0.6, wob=0.3, soft=0.6, body=1)
        for q in range(4):
            s.ink([(xb - 88 * k + q * 4, ty - 40 * k), (xb - 86 * k + q * 4, ty - 58 * k)], 1.0, SEPIA, 1.0)
    # a wall fan on the right wall, and a mirror with red lettering (a shop-opening present)
    wf = P3(2510, 900, 1.6)
    fmm = s.ellipse(wf[0], wf[1], 44, 52)
    s.wash(fmm, MIST, 0.35, wob=0.5, soft=1, edge=0.5, ew=3)
    for q in np.linspace(-0.8, 0.8, 7):
        s.ink([(wf[0] + q * 44, wf[1] - 50 * math.sqrt(1 - q * q)), (wf[0] + q * 44, wf[1] + 50 * math.sqrt(1 - q * q))], 0.9, SEPIA, 0.7, skip=0.2)
    oval(s, wf[0], wf[1], 44, 52, 1.6, SEPIA, 1.3)
    s.wash(s.ellipse(wf[0], wf[1], 10, 12), CELADON, 0.8, wob=0.3, soft=0.6)
    s.ink([(wf[0] + 10, wf[1] + 50), (wf[0] + 20, wf[1] + 90)], 3.0, SEPIA, 1.3)
    mr = [P3(2510, 1010, 3.0), P3(2510, 1010, 2.2), P3(2510, 1250, 2.2), P3(2510, 1250, 3.0)]
    mm = Q(s, mr)
    put(s, mm, MIST, 0.35, lift=0.8, wob=0.5, edge=0.4, ew=3)
    s.lift(Q(s, [mr[0], mr[1], (mr[2][0] - 4, (mr[1][1] + mr[2][1]) / 2)]), 0.3, soft=3)
    mt = letters(s, "生意興隆", (mr[0][0] + mr[1][0]) / 2, (mr[0][1] + mr[3][1]) / 2, 30, KAI, 400, vertical=True, sx=0.75)
    paint_letters(s, mt, PERSIMMON, 1.2, wob=0.3)
    path(s, mr, 1.6, SEPIA, 1.3, closed=True, over=2, r=r)
    # the room's light: warm, falling off toward the corners
    s.wash(om, OCHRE, 0.22, wob=0, soft=0, edge=0, grad=s.rgrad(1350, 900, 200, 1200, 0.4, 1.0, sy=0.6), flow=0.5, fs=150)
    s.wash(om, UMBER, 0.28, wob=0, soft=0, edge=0, grad=s.rgrad(1500, 950, 500, 1300, 0, 1, sy=0.55), flow=0.5, fs=150)
    s.wash(s.rect(x0, y0, x1, y0 + 70), SEPIA, 0.45, wob=0, soft=0, edge=0, grad=s.vgrad(y0, y0 + 70, 1, 0), flow=0.5, fs=80)   # under the shutter box
    s.wash(s.rect(x0, by1 - 30, x1, y1), UMBER, 0.3, wob=2, bleed=10, edge=0, grad=s.vgrad(by1 - 30, y1, 0.2, 1))          # the floor by the counters
    mottle(s, r, x0 + 40, y0 + 60, x1 - 40, by1, [OCHRE, SIENNA, ROSE], 6, 0.12, size=(100, 220), clip=om)


def front(S, r):
    """The shop's front: tiled piers, the shutter box, the fascia sign, the hanging menu boards and bags."""
    x0, y0, x1, y1 = OPEN
    s = Sub(S, 380, 450, 2640, G + 20)
    for px0, px1 in ((400, 490), (2510, 2604)):
        pm = s.rect(px0, 470, px1, G + 2)
        s.wash(pm, TILE, 0.55, wob=1, soft=0.9, edge=0.4, ew=5, flow=0.5, fs=90, gran=0.45, body=1)
        s.wash(pm, SLATE, 0.3, wob=2, bleed=8, edge=0.1, grad=s.hgrad(px0, px1, 0.2, 1))
        tiles(s, r, px0, 600, px1, G, 21.0, SEPIA, 0.5, patches=3, size=((40, 90), (60, 200)), singles=14)
        s.wash(s.rect(px0, G - 120, px1, G + 2), SLATE, 0.45, wob=1, soft=1, edge=0.4, ew=3)     # a darker plinth
        line(s, r, (px0, G - 120), (px1, G - 121), 1.4, SEPIA, 1.2)
        line(s, r, (px0, 470), (px0 + J(r, 1.5), G), 2.2, SEPIA, 1.5, dbl=0.5)
        line(s, r, (px1, 470), (px1 + J(r, 1.5), G), 2.2, SEPIA, 1.5, dbl=0.5)
    # stickers on the left pier: a gas delivery number, a lost cat
    for (sx, sy, sw, sh, c, t1) in ((418, 860, 58, 80, NOTE, "瓦斯"), (420, 980, 56, 70, MIST, "尋貓")):
        m = Q(s, [(sx, sy), (sx + sw, sy + 2), (sx + sw - 1, sy + sh), (sx + 1, sy + sh - 2)])
        put(s, m, c, 0.45, lift=0.9, wob=0.4, edge=0.4, ew=2)
        paint_letters(s, letters(s, t1, sx + sw / 2, sy + 20, 20, SERIF, 900), PERSIMMON, 1.3, wob=0.2)
        for q in range(3):
            line(s, r, (sx + 8, sy + 38 + q * 10), (sx + sw - 8, sy + 38 + q * 10), 0.9, SEPIA, 0.7, over=0, skip=0.25)
        path(s, [(sx, sy), (sx + sw, sy + 2), (sx + sw - 1, sy + sh), (sx + 1, sy + sh - 2)], 1.0, SEPIA, 0.9, closed=True, over=1, r=r)
    # the shutter box under the sign, the shutter's bottom bar just showing
    sb = s.rect(x0, 604, x1, y0 + 2)
    put(s, sb, SLATE, 0.6, lift=0.9, wob=0.8, edge=0.4, ew=4, flow=0.5, fs=80)
    s.wash(sb, CELADON, 0.25, wob=2, bleed=6, edge=0.1)
    s.wash(sb, PAYNE, 0.3, wob=0, soft=0, edge=0, grad=s.vgrad(604, y0, 0, 1))
    for yy in (622, 638):
        line(s, r, (x0 + 4, yy), (x1 - 4, yy + J(r, 1)), 1.0, SEPIA, 0.8, over=0, skip=0.2)
    s.wash(s.rect(x0, y0 - 6, x1, y0 + 4), INK, 0.5, wob=0.4, soft=0.6, edge=0.4)
    box(s, r, x0, 604, x1, y0, 1.8, SEPIA, 1.4, over=4, dbl=0.4)
    # the fascia: a long sign, cream ground, red lettering, a strip of blue
    fx0, fy0, fx1, fy1 = 410, 476, 2594, 600
    fm = s.rect(fx0, fy0, fx1, fy1)
    put(s, fm, CREAM, 0.28, lift=0.95, wob=0.8, edge=0.35, ew=6, flow=0.5, fs=150, gran=0.35)
    s.wash(s.rect(fx0, fy1 - 22, fx1, fy1), TEAL, 0.65, wob=0.8, soft=0.8, edge=0.5, ew=3)
    s.wash(fm, UMBER, 0.12, wob=2, bleed=10, edge=0.1, flow=0.8, fs=60)
    streaks(s, r, fx0 + 20, fx1 - 20, fy0 + 4, fy1, 10, UMBER, 0.12, (6, 14), clip=fm)
    paint_letters(s, letters(s, "巷口早點", 1500, 530, 92, SERIF, 900, sx=1.25), PERSIMMON, 1.55, wob=0.5)
    paint_letters(s, letters(s, "巷口早點", 1500, 530, 92, SERIF, 900, sx=1.25), ROSE, 0.35, wob=0.5)
    paint_letters(s, letters(s, "豆漿 · 燒餅 · 油條", 760, 534, 40, SERIF, 800), PERSIMMON, 1.3, wob=0.4)
    paint_letters(s, letters(s, "早上 5:30 – 11:00", 2240, 534, 38, SERIF, 700), INKBLUE, 1.1, wob=0.4)
    s.lift(letters(s, "電話外送 2551-0816", 1500, 589, 18, SERIF, 700), 0.7, soft=0.5, wob=0.2)
    box(s, r, fx0, fy0, fx1, fy1, 2.4, SEPIA, 1.5, over=6, dbl=0.5)
    s.wash(s.rect(fx0 + 6, fy1, fx1 - 6, fy1 + 8), SEPIA, 0.6, wob=0.6, soft=0.6, edge=0.4)
    for xx in np.linspace(fx0 + 20, fx1 - 20, 9):
        s.wash(s.ellipse(xx, fy0 + 12, 3, 3), SEPIA, 0.6, wob=0.2, soft=0.5)
    # the hanging menu boards: red characters, prices in black
    bar = (560, 1330, 674)
    put(s, s.rect(bar[0] - 10, bar[2] - 5, bar[1] + 10, bar[2] + 5), SIENNA, 0.8, lift=0.8, wob=0.5, edge=0.4)
    line(s, r, (bar[0] - 12, bar[2] - 5), (bar[1] + 12, bar[2] - 6), 1.6, SEPIA, 1.3)
    line(s, r, (bar[0] - 12, bar[2] + 5), (bar[1] + 12, bar[2] + 5), 1.6, SEPIA, 1.3)
    menu = (("豆漿", "15"), ("米漿", "20"), ("蛋餅", "30"), ("飯糰", "35"), ("燒餅", "25"), ("油條", "20"))
    for i, (name, price) in enumerate(menu):
        cx = 612 + i * 132 + J(r, 3)
        a = J(r, 1.2)                              # each hangs a little crooked
        top, bot, hw = 700, 958, 47
        ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
        rot = lambda dx, dy: (cx + dx * ca - dy * sa, top + dx * sa + dy * ca)
        quad = [rot(-hw, 0), rot(hw, 0), rot(hw, bot - top), rot(-hw, bot - top)]
        pm = Q(s, quad)
        put(s, pm, CREAM, 0.18, lift=1.0, wob=0.6, edge=0.4, ew=4, flow=0.5, fs=40)
        s.wash(pm, OCHRE, 0.12, wob=1, soft=2, edge=0, grad=s.vgrad(top, bot, 0.2, 1))
        s.ink([(cx - 30, bar[2]), rot(-30, 0)], 1.0, SEPIA, 1.0, taper=(0, 0))
        s.ink([(cx + 30, bar[2]), rot(30, 0)], 1.0, SEPIA, 1.0, taper=(0, 0))
        nx, ny = rot(0, 92)
        paint_letters(s, letters(s, name, nx, ny, 62, SERIF, 900, vertical=True, gap=0.1, rot=-a), PERSIMMON, 1.6, wob=0.35)
        px_, py_ = rot(0, 214)
        paint_letters(s, letters(s, price, px_, py_, 38, SERIF, 800, rot=-a), INK, 1.2, wob=0.3)
        bl = [rot(-hw + 6, 6), rot(hw - 6, 6), rot(hw - 6, bot - top - 6), rot(-hw + 6, bot - top - 6)]
        pl(s, r, bl, 1.0, PERSIMMON, 0.9, over=1, closed=True, skip=0.2)
        pl(s, r, quad, 1.7, SEPIA, 1.4, over=2, closed=True)
        s.wash(Q(s, [rot(-hw + 4, bot - top), rot(hw + 4, bot - top), rot(hw + 10, bot - top + 14), rot(-hw + 10, bot - top + 14)]), SEPIA, 0.2, wob=1, bleed=4)
    # plastic bags hanging on a rod by the door: red-and-white stripes, pink, clear
    rod = (1872, 2034, 690)
    s.ink([(rod[0], rod[2]), (rod[1], rod[2] + 2)], 3.0, SEPIA, 1.4)
    for i, (bx, c, kind) in enumerate(((1898, PERSIMMON, "stripe"), (1950, ROSE, "plain"), (2004, MIST, "clear"))):
        L, hw = U(r, 96, 120), U(r, 24, 28)
        bt = rod[2] + 30
        s.ink([(bx, rod[2] - 3), (bx + 2, rod[2] + 8)], 1.6, SEPIA, 1.3)
        for sd in (-1, 1):            # the handles gathered on the hook
            s.ink([(bx + 1, rod[2] + 8), (bx + sd * hw * 0.6, bt - 14), (bx + sd * hw * 0.8, bt + 2)], 2.2, c if kind != "clear" else SLATE, 0.8)
        for j in range(3):            # a few bags one behind another, each hanging a little differently
            o = (2 - j) * 4
            pts = [(bx - hw + o, bt + j), (bx + hw + o, bt + j), (bx + hw * 1.12 + o, bt + L * 0.55), (bx + hw + o + J(r, 3), bt + L - j * 6),
                   (bx + o, bt + L + 4 - j * 6 + J(r, 4)), (bx - hw + o + J(r, 3), bt + L - j * 6), (bx - hw * 1.1 + o, bt + L * 0.5)]
            bag = s.poly(pts)
            if kind == "clear":
                s.lift(bag, 0.25, soft=1, wob=0.6)
                s.wash(bag, MIST, 0.18, wob=0.6, soft=0.8, edge=0.7, ew=2)
            else:
                put(s, bag, CREAM if kind == "stripe" else c, 0.2 if kind == "stripe" else 0.42, lift=0.85, wob=0.6, edge=0.4, ew=2)
                if kind == "stripe":
                    for q in np.arange(-hw + 3, hw, 8):
                        s.wash(s.ribbon([(bx + o + q, bt + 4), (bx + o + q * 1.08, bt + L)], 3.6, smooth=False), c, 0.85, wob=0.6, soft=0.5, clip=bag)
            s.wash(bag, SLATE, 0.18, wob=1, soft=2, grad=s.hgrad(bx - hw, bx + hw, 0, 1))
            contour(s, bag, 1.0, SEPIA, 0.8 if j < 2 else 1.1)
            s.ink([(bx - hw * 0.4 + o, bt + 8), (bx - hw * 0.3 + o, bt + L * 0.8)], 0.9, SEPIA, 0.5, skip=0.3)   # a crease


def counter(S, r):
    """The stainless counter across the front and what is on it: the griddle, the soy milk pot, the rice-milk pot,
    two stacks of bamboo steamers, a warming case of 燒餅 and 油條."""
    s = Sub(S, 490, 560, 2120, G + 10)
    CX0, CX1 = 520, 2060
    top_back = P3(0, CTOP, 0.7)[1]
    # the front: brushed stainless, cool from the lane, warm where it catches the room
    cf = s.rect(CX0, CTOP, CX1, G)
    put(s, cf, MIST, 0.42, lift=1.0, wob=0.8, edge=0.35, ew=5, flow=0.4, fs=200, gran=0.25)
    s.wash(cf, SLATE, 0.32, wob=1, soft=2, edge=0, grad=s.vgrad(CTOP + 20, G, 0.2, 1), flow=0.5, fs=150)
    s.wash(cf, RAIN, 0.2, wob=3, bleed=16, edge=0.1, flow=0.8, fs=120)
    for _ in range(60):          # the grain of the brushed steel, and the soft upright reflections
        x = U(r, CX0 + 4, CX1 - 4)
        if r.random() < 0.5:
            hl(s, [(x, CTOP + U(r, 14, 60)), (x + J(r, 1), G - U(r, 10, 80))], U(r, 1.2, 3.0), U(r, 0.1, 0.25), dry=0.3, clip=cf)
        else:
            s.stroke([(x, CTOP + U(r, 14, 60)), (x + J(r, 1), G - U(r, 10, 80))], U(r, 2, 8), SLATE, U(r, 0.08, 0.2), dry=0.45, edge=0.2, wob=0.6, taper=(0.1, 0.2), clip=cf)
    for xa, xb, c in ((640, 820, OCHRE), (1250, 1480, OCHRE), (1780, 1990, AMBER)):
        s.wash(s.rect(xa, CTOP + 20, xb, G - 40), c, 0.12, wob=3, bleed=24, edge=0, clip=cf)
    s.wash(s.rect(CX0, G - 40, CX1, G), PAYNE, 0.45, wob=0.8, soft=1, edge=0.4)                    # the kick space
    for xx in (900, 1300, 1700):                                                                  # panel seams
        line(s, r, (xx, CTOP + 16), (xx + J(r, 1), G - 40), 1.4, SEPIA, 1.1, over=1)
        hl(s, [(xx + 3, CTOP + 18), (xx + 3, G - 42)], 2, 0.3)
    line(s, r, (CX0 - 6, CTOP + 16), (CX1 + 6, CTOP + 16), 1.6, SEPIA, 1.3, dbl=0.5)
    line(s, r, (CX0 - 2, G - 40), (CX1 + 2, G - 40), 1.4, SEPIA, 1.2)
    # paper signs taped to the steel: 內用 外帶, and a red one for what is made to order
    for (x0, y0, x1, y1, txt, c, sz) in ((1040, 1262, 1180, 1340, "內用 · 外帶", INK, 24), (1520, 1258, 1610, 1372, "現做", PERSIMMON, 34)):
        q = [(x0, y0), (x1, y0 + 2), (x1 - 1, y1), (x0 + 2, y1 - 2)]
        m = Q(s, q)
        put(s, m, CREAM if c is INK else NOTE, 0.12 if c is INK else 0.45, lift=0.9, wob=0.5, edge=0.4, ew=2)
        s.wash(m, SLATE, 0.15, wob=1, soft=2)
        vt = txt == "現做"
        paint_letters(s, letters(s, txt, (x0 + x1) / 2, (y0 + y1) / 2, sz, KAI, 400, vertical=vt), c, 1.3, wob=0.3)
        for (tx, ty) in ((x0 + 8, y0 + 2), (x1 - 8, y0 + 3)):
            s.wash(s.rect(tx - 12, ty - 6, tx + 12, ty + 6), MIST, 0.35, wob=0.3, soft=0.6, body=1)
        path(s, q, 1.1, SEPIA, 1.0, closed=True, over=1, r=r)
    # the counter's top: a bright strip, and its lip
    tm = Q(s, [(CX0, CTOP), (CX1, CTOP), P3(CX1, CTOP, 0.7), P3(CX0, CTOP, 0.7)])
    put(s, tm, MIST, 0.25, lift=1.0, wob=0.5, edge=0.3, ew=2)
    hl(s, [(CX0, CTOP + 2), (CX1, CTOP + 2)], 3.5, 0.6)
    s.wash(s.rect(CX0, CTOP + 4, CX1, CTOP + 16), SLATE, 0.3, wob=0.5, soft=0.8)
    line(s, r, (CX0 - 8, CTOP), (CX1 + 8, CTOP), 2.2, SEPIA, 1.5, dbl=0.5)
    pl(s, r, [P3(CX0, CTOP, 0.7), (CX0, CTOP), (CX0, G)], 2.0, SEPIA, 1.5)
    pl(s, r, [P3(CX1, CTOP, 0.7), (CX1, CTOP), (CX1, G)], 2.0, SEPIA, 1.5)
    # ---- the griddle: a black plate in a steel frame, two egg pancakes, a spatula, a squeeze bottle, scallions
    gx0, gx1, gt = 548, 930, 1168
    gm = Q(s, [(gx0, CTOP), (gx1, CTOP), (gx1 - 8, gt), (gx0 + 8, gt)])
    put(s, gm, MIST, 0.5, lift=0.9, wob=0.6, edge=0.4, ew=3)
    s.wash(gm, SLATE, 0.4, wob=1, soft=1.5, grad=s.vgrad(gt, CTOP, 0.3, 1))
    plate = Q(s, [(gx0 + 10, gt), (gx1 - 10, gt), (gx1 - 30, gt - 12), (gx0 + 30, gt - 12)])
    put(s, plate, INK, 0.8, lift=0.6, wob=0.5, edge=0.4, ew=2)
    s.wash(plate, AMBER, 0.2, wob=1, soft=1.5, grad=s.hgrad(gx0, gx1, 0, 1))
    hl(s, [(gx0 + 34, gt - 11), (gx1 - 34, gt - 11)], 2, 0.4)
    for ex, ew_ in ((640, 70), (780, 64)):
        em = s.ellipse(ex, gt - 15, ew_, 7)
        put(s, em, NOTE, 0.5, lift=0.9, wob=0.6, edge=0.6, ew=2)
        s.wash(em, AMBER, 0.3, wob=1, soft=1.5, grad=s.hgrad(ex - ew_, ex + ew_, 0, 1))
        s.spatter(6, LEAF, 0.7, box=(ex - ew_ * 0.6, gt - 20, ex + ew_ * 0.6, gt - 12), r=(1, 2))
        oval(s, ex, gt - 15, ew_, 7, 1.2, SEPIA, 1.1)
    sp = [(870, gt - 14), (910, gt - 34), (940, gt - 72)]
    put(s, s.ribbon([(842, gt - 10), (878, gt - 16)], 10, smooth=False), MIST, 0.5, lift=0.8, wob=0.4)
    s.stroke(sp, 7, BURNT, 0.9, taper=(0.1, 0.2), wob=0.5, edge=0.4, body=1)
    s.ink(sp, 1.2, SEPIA, 1.1)
    ob = (575, gt - 12)
    obm = np.maximum(s.rect(ob[0] - 13, ob[1] - 62, ob[0] + 13, ob[1]), Q(s, [(ob[0] - 4, ob[1] - 62), (ob[0] + 4, ob[1] - 62), (ob[0], ob[1] - 84)]))
    put(s, obm, NOTE, 0.6, lift=0.8, wob=0.4, edge=0.5, ew=2)
    s.wash(s.rect(ob[0] - 13, ob[1] - 62, ob[0] + 13, ob[1] - 30), PERSIMMON, 0.5, wob=0.3, soft=0.6)
    contour(s, obm, 1.2, SEPIA, 1.2)
    bw = s.ellipse(890, gt - 30, 30, 18)       # a bowl of beaten egg
    put(s, bw, MIST, 0.3, lift=0.9, wob=0.4, edge=0.4)
    s.wash(s.ellipse(890, gt - 44, 28, 6), NOTE, 0.7, wob=0.3, soft=0.6)
    oval(s, 890, gt - 40, 30, 7, 1.2, SEPIA, 1.2)
    s.ink(ell(890, gt - 40, 30, 22, 0.1, math.pi - 0.1, 16), 1.2, SEPIA, 1.2)
    path(s, [(gx0, CTOP), (gx0 + 8, gt), (gx1 - 8, gt), (gx1, CTOP)], 1.6, SEPIA, 1.3, over=2, r=r)
    steam(s, r, 700, gt - 22, 110, 30, 0.22, 3)
    # ---- the soy milk pot, and the rice-milk pot beside it
    def pot(cx, mouth, rx, base, warm=0.2, lid=False, milk=SOY):
        body = np.maximum(s.rect(cx - rx, mouth, cx + rx, base - 6), s.ellipse(cx, base - 6, rx, 7))
        put(s, body, MIST, 0.4, lift=1.0, wob=0.6, edge=0.4, ew=4, flow=0.4, fs=60)
        s.wash(body, SLATE, 0.55, wob=0, soft=2, edge=0, grad=s.hgrad(cx - rx, cx - rx * 0.55, 1, 0.1))
        s.wash(body, SLATE, 0.45, wob=0, soft=2, edge=0, grad=s.hgrad(cx + rx * 0.2, cx + rx, 0, 1))
        s.wash(body, OCHRE, warm, wob=1, soft=3, edge=0, grad=s.hgrad(cx + rx * 0.45, cx + rx, 0, 1))
        hl(s, [(cx - rx * 0.42, mouth + 8), (cx - rx * 0.44, base - 10)], rx * 0.12, 0.7)
        hl(s, [(cx + rx * 0.1, mouth + 10), (cx + rx * 0.1, base - 14)], rx * 0.05, 0.3)
        for yy in (mouth + 20, base - 26):     # rolled bands
            s.ink(ell(cx, yy, rx, 6, 0.1, math.pi - 0.1, 24), 1.1, SEPIA, 0.9, skip=0.2)
        for sd in (-1, 1):                      # handles
            hx = cx + sd * rx
            s.ink([(hx, mouth + 30), (hx + sd * 16, mouth + 32), (hx + sd * 18, mouth + 46), (hx, mouth + 50)], 3.0, SEPIA, 1.4)
        rim = s.ellipse(cx, mouth, rx + 2, 9)
        put(s, rim, MIST, 0.3, lift=1.0, wob=0.4, edge=0.3)
        s.wash(s.ellipse(cx, mouth + 1, rx - 5, 6), milk, 0.9, wob=0.3, soft=0.6, edge=0.5, ew=2)
        s.wash(s.ellipse(cx, mouth + 1, rx - 5, 6), SIENNA, 0.15, wob=0.3, soft=1)
        oval(s, cx, mouth, rx + 2, 9, 1.8, SEPIA, 1.4)
        s.ink([(cx - rx, mouth), (cx - rx + J(r, 1), base - 6)], 2.0, SEPIA, 1.5)
        s.ink([(cx + rx, mouth), (cx + rx + J(r, 1), base - 6)], 2.2, SEPIA, 1.5)
        s.ink(ell(cx, base - 6, rx, 7, 0.05, math.pi - 0.05, 24), 1.8, SEPIA, 1.4)
        return body
    burner = s.rect(POT[0] - 118, CTOP - 22, POT[0] + 118, CTOP)
    put(s, burner, INK, 0.6, lift=0.6, wob=0.5, edge=0.4)
    s.wash(burner, AMBER, 0.2, wob=0.5, soft=2, grad=s.vgrad(CTOP - 22, CTOP, 0, 1))
    box(s, r, POT[0] - 118, CTOP - 22, POT[0] + 118, CTOP, 1.4, SEPIA, 1.3, over=2)
    for q in np.linspace(-0.8, 0.8, 9):
        s.wash(s.ellipse(POT[0] + q * 100, CTOP - 4, 4, 3), PRUSSIAN, 0.6, wob=0.2, soft=0.5)   # the blue of the gas
    pot(POT[0], POT[1], POTRX, CTOP - 20, 0.22)
    lid = s.ellipse(POT[0] - 128, POT[1] + 74, 26, 92, 12)                                            # its lid leant against it
    put(s, lid, MIST, 0.4, lift=0.9, wob=0.5, edge=0.4)
    s.wash(lid, SLATE, 0.45, wob=1, soft=2, grad=s.hgrad(POT[0] - 160, POT[0] - 100, 0, 1))
    hl(s, [(POT[0] - 140, POT[1] + 10), (POT[0] - 150, POT[1] + 130)], 4, 0.5)
    contour(s, lid, 1.6, SEPIA, 1.3)
    pot(RICE[0], RICE[1], 62, CTOP, 0.26, milk=SIENNA)
    s.ink([(RICE[0] + 20, RICE[1] + 2), (RICE[0] + 50, RICE[1] - 60), (RICE[0] + 56, RICE[1] - 92)], 3.2, SEPIA, 1.4)      # its ladle
    hl(s, [(RICE[0] + 52, RICE[1] - 62), (RICE[0] + 56, RICE[1] - 88)], 1.6, 0.4)
    # a short stack of plastic cups and their lids by the pots
    cs_ = (1340, CTOP)
    cm = Q(s, [(cs_[0] - 22, cs_[1]), (cs_[0] + 22, cs_[1]), (cs_[0] + 26, cs_[1] - 120), (cs_[0] - 26, cs_[1] - 120)])
    s.lift(cm, 0.6, soft=0.8, wob=0.4)
    s.wash(cm, MIST, 0.3, wob=0.4, soft=0.6, edge=0.6, ew=2, body=0.8)
    for q in np.arange(cs_[1] - 116, cs_[1], 9):
        s.ink(ell(cs_[0], q, 25, 3, 0.1, math.pi - 0.1, 10), 0.8, SEPIA, 0.7)
    path(s, [(cs_[0] - 22, cs_[1]), (cs_[0] - 26, cs_[1] - 120), (cs_[0] + 26, cs_[1] - 120), (cs_[0] + 22, cs_[1])], 1.2, SEPIA, 1.1, over=1, r=r)
    # ---- the steamers
    def stack(cx, rx, base_top, n, th=38, seed=0):
        g = rs(seed)
        bp = np.maximum(s.rect(cx - rx - 6, base_top, cx + rx + 6, CTOP - 4), s.ellipse(cx, CTOP - 4, rx + 6, 6))
        put(s, bp, MIST, 0.45, lift=1.0, wob=0.6, edge=0.4, ew=3)
        s.wash(bp, SLATE, 0.5, wob=0, soft=2, grad=s.hgrad(cx - rx, cx + rx, 0.2, 1))
        hl(s, [(cx - rx * 0.5, base_top + 6), (cx - rx * 0.5, CTOP - 10)], 8, 0.6)
        s.wash(s.rect(cx - rx - 6, base_top, cx + rx + 6, base_top + 14), SLATE, 0.45, wob=0.4, soft=0.8, edge=0.4)
        contour(s, bp, 1.8, SEPIA, 1.4)
        s.ink(ell(cx, base_top + 14, rx + 6, 5, 0.05, math.pi - 0.05, 24), 1.2, SEPIA, 1.0)
        y = base_top
        for i in range(n):
            t0, t1 = y - th, y
            tm = s.rect(cx - rx, t0, cx + rx, t1)
            put(s, tm, OCHRE, 0.45, lift=0.95, wob=0.6, edge=0.4, ew=3, flow=0.5, fs=30, gran=0.5)
            s.wash(tm, SIENNA, 0.4, wob=0, soft=2, edge=0, grad=s.hgrad(cx - rx * 0.2, cx + rx, 0, 1))
            s.wash(tm, UMBER, 0.25, wob=0, soft=2, edge=0, grad=s.hgrad(cx - rx, cx - rx * 0.6, 1, 0))
            s.wash(s.rect(cx - rx, t0, cx + rx, t0 + 7), BURNT, 0.45, wob=0.4, soft=0.6, edge=0.4)          # the bound rim
            s.wash(s.rect(cx - rx, t1 - 6, cx + rx, t1), BURNT, 0.35, wob=0.4, soft=0.6, edge=0.4)
            for q in np.linspace(-0.85, 0.85, 9):
                xx = cx + q * rx
                if g.random() < 0.7:
                    seg(s, (xx, t0 + 8), (xx + J(g, 0.5), t1 - 6), 0.9, SEPIA, 0.6, r=r, skip=0.3)
            s.ink(ell(cx, t1, rx, 5, 0.05, math.pi - 0.05, 24), 1.4, SEPIA, 1.2)
            s.ink([(cx - rx, t0), (cx - rx, t1)], 1.8, SEPIA, 1.4)
            s.ink([(cx + rx, t0), (cx + rx, t1)], 1.8, SEPIA, 1.4)
            hl(s, [(cx - rx * 0.62, t0 + 9), (cx - rx * 0.62, t1 - 8)], 5, 0.3)
            s.wash(s.ellipse(cx - rx * 0.2 + J(g, rx * 0.5), t1, U(g, 16, 30), 5), None, 0.35, wob=1, bleed=6, lift=True)   # steam leaking
            y = t0
        dome = s.ellipse(cx, y, rx, 32) * (1 - s.vgrad(y - 1, y + 1, 0, 1))
        put(s, dome, OCHRE, 0.45, lift=0.95, wob=0.6, edge=0.4, ew=3, gran=0.55)
        s.wash(dome, SIENNA, 0.4, wob=0, soft=2, edge=0, grad=s.hgrad(cx - rx * 0.1, cx + rx, 0, 1))
        s.hatch(dome, 0.5, 9, 0.9, SEPIA, 0.5, skip=0.3)
        s.hatch(dome, -0.5, 9, 0.9, SEPIA, 0.5, skip=0.3)
        s.ink(ell(cx, y, rx, 32, math.pi + 0.05, 2 * math.pi - 0.05, 30), 1.8, SEPIA, 1.4)
        s.ink([(cx - 10, y - 31), (cx - 4, y - 42), (cx + 6, y - 42), (cx + 10, y - 31)], 1.6, SEPIA, 1.3)
        return y - 32
    stack(1462, 104, 1112, 6, 38, 61)
    stack(1690, 84, 1132, 3, 36, 62)
    # ---- the warming case: 燒餅 and 油條 behind glass, a price card taped on
    wx0, wy0, wx1, wy1 = 1800, 1040, 2046, CTOP
    wm = s.rect(wx0, wy0, wx1, wy1)
    s.lift(wm, 0.8, soft=0.6, wob=0)
    s.wash(wm, OCHRE, 0.32, wob=0.6, soft=1, edge=0.4, ew=3)
    s.wash(s.rect(wx0, wy0, wx1, wy0 + 16), AMBER, 0.4, wob=0.4, soft=2)
    for i in range(4):                                   # 燒餅 stacked
        bm = s.ellipse(1858 + J(r, 3), wy1 - 66 - i * 11, 46, 9)
        put(s, bm, AMBER, 0.55, lift=0.5, wob=0.5, edge=0.5, ew=2, gran=0.5)
        s.wash(bm, BURNT, 0.3, wob=1, soft=1.5, grad=s.vgrad(wy1 - 76 - i * 11, wy1 - 58 - i * 11, 0, 1))
        s.spatter(5, None, 0.6, box=(1820, wy1 - 74 - i * 11, 1900, wy1 - 62 - i * 11), r=(1, 1.5), lift=True)
        oval(s, 1858, wy1 - 66 - i * 11, 46, 9, 1.1, SEPIA, 1.0, skip=0.1)
    for i in range(5):                                   # 油條 heaped
        y = wy1 - 54 - i * 10 + J(r, 3)
        pts = [(1920, y + 6), (1970, y - 2), (2030, y + 4)]
        rb = s.ribbon(pts, 14)
        put(s, rb, AMBER, 0.6, lift=0.5, wob=0.5, edge=0.5, ew=2, gran=0.5)
        s.wash(rb, BURNT, 0.35, wob=1, bleed=3)
        for q in np.linspace(0.1, 0.9, 6):
            xx = 1920 + 110 * q
            s.ink([(xx - 3, y - 4), (xx + 3, y + 6)], 0.9, SEPIA, 0.8)
        s.ink(pts, 1.0, SEPIA, 0.9, skip=0.2)
    s.wash(s.rect(wx0, wy1 - 50, wx1, wy1 - 44), MIST, 0.4, wob=0.3, soft=0.6)
    s.lift(Q(s, [(wx0 + 12, wy0 + 20), (wx0 + 60, wy0 + 20), (wx0 + 14, wy1 - 60)]), 0.4, soft=3, wob=1)
    card = Q(s, [(1960, wy0 + 24), (2030, wy0 + 22), (2032, wy0 + 66), (1962, wy0 + 68)])
    put(s, card, CREAM, 0.1, lift=0.9, wob=0.3)
    paint_letters(s, letters(s, "燒餅油條", 1996, wy0 + 38, 15, SERIF, 900), PERSIMMON, 1.3, wob=0.2)
    paint_letters(s, letters(s, "45", 1996, wy0 + 56, 17, SERIF, 900), INK, 1.2, wob=0.2)
    s.wash(s.rect(1975, wy0 + 18, 2016, wy0 + 26), CREAM, 0.3, wob=0.3, soft=0.5)
    box(s, r, wx0, wy0, wx1, wy1, 2.0, SEPIA, 1.5, over=3, dbl=0.4)
    line(s, r, (wx0, wy1 - 50), (wx1, wy1 - 50), 1.3, SEPIA, 1.1, over=1)
    pl(s, r, [(wx0, wy0), P3(wx0, wy0, 0.5), P3(wx1, wy0, 0.5), (wx1, wy0)], 1.4, SEPIA, 1.2, over=2)
    # painted steam: the page will add moving steam over these
    steam(s, r, POT[0], POT[1] - 6, 260, 54, 0.42, 5)
    steam(s, r, RICE[0], RICE[1] - 6, 150, 30, 0.3, 3)
    steam(s, r, STEAM1[0], STEAM1[1], 140, 60, 0.3, 4)
    steam(s, r, STEAM2[0], STEAM2[1], 120, 44, 0.3, 3)


def lane(S, r):
    """The lane: the gutter with its slotted covers along the building, a red kerb line, damp tarmac, a puddle with
    the shop's light in it, a road-painted 慢 lying in perspective."""
    s = Sub(S, 0, G - 20, W, H)
    gm = s.rect(-10, G, W + 10, H + 10)
    s.wash(gm, SLATE, 0.36, wob=1, soft=0.9, edge=0.2, ew=6, flow=0.6, fs=260, gran=0.55, body=1)
    s.wash(gm, PAYNE, 0.14, wob=3, bleed=14, edge=0.1, flow=0.8, fs=180, gran=0.5, grad=s.vgrad(G, H, 0.3, 1), clip=gm)
    s.wash(gm, VIOLET, 0.12, wob=3, bleed=14, edge=0.1, flow=0.9, fs=120, clip=gm)
    for _ in range(14):           # fallen leaves from somebody's plant, a bottle cap
        lx, ly = U(r, 100, W - 100), U(r, G + 60, H - 20)
        a = U(r, 0, 3.1)
        lm = s.ellipse(lx, ly, U(r, 8, 14), U(r, 3, 5), math.degrees(a) * 0.3)
        s.wash(lm, (OCHRE, SIENNA, MOSS)[int(r.integers(0, 3))], U(r, 0.5, 0.8), wob=0.6, soft=0.6, edge=0.5, ew=2)
    s.wash(s.ellipse(1180, 1700, 8, 3), PERSIMMON, 0.8, wob=0.3, soft=0.5)
    for y in (G + 80, G + 150):  # a long seam in the tarmac
        xx = -10.0
        while xx < W:
            ln = U(r, 300, 900)
            if r.random() < 0.6:
                seg(s, (xx, y + J(r, 3)), (xx + ln, y + J(r, 3)), 1.0, SEPIA, U(r, 0.4, 0.7), r=r, skip=0.35, bow=0.002)
            xx += ln + U(r, 50, 300)
    mottle(s, r, 0, G + 60, W, H, [UMBER, PAYNE, RAIN, OCHRE], 9, 0.14, size=(100, 260), clip=gm)
    # patches of newer tarmac, cracks
    for _ in range(6):
        x, y = U(r, 0, W), U(r, G + 80, H - 30)
        w_, h_ = U(r, 140, 420), U(r, 20, 60)
        pm = Q(s, [(x, y), (x + w_, y + J(r, 4)), (x + w_ + J(r, 10), y + h_), (x + J(r, 10), y + h_ + J(r, 4))])
        s.wash(pm, PAYNE, U(r, 0.08, 0.2), wob=3, soft=1, edge=0.5, ew=4, gran=0.6, bleed=3)
    for _ in range(10):
        x, y = U(r, 0, W), U(r, G + 60, H)
        pts = [(x, y)]
        for _ in range(4):
            x, y = x + U(r, 20, 70) * (1 if r.random() < 0.5 else -1), y + J(r, 8)
            pts.append((x, y))
        s.ink(pts, 1.0, SEPIA, 0.6, skip=0.3, smooth=False)
    # the gutter covers, slot by slot
    gt0, gt1 = G + 2, G + 34
    cm = s.rect(-10, gt0, W + 10, gt1)
    put(s, cm, MIST, 0.45, lift=0.85, wob=0.8, edge=0.4, ew=3, gran=0.5)
    s.wash(cm, SLATE, 0.3, wob=2, bleed=6, edge=0.1, flow=0.8, fs=60)
    x = -20.0
    while x < W:
        seg(s, (x, gt0), (x - 2, gt1), 1.4, SEPIA, 1.0, r=r)
        for q in np.arange(x + 14, x + 118, 13):
            s.wash(s.rect(q, gt0 + 9, q + 5, gt1 - 8), INK, 0.7, wob=0.3, soft=0.5, edge=0.4)
        x += 130 + J(r, 2)
    line(s, r, (-10, gt0), (W + 10, gt0), 1.8, SEPIA, 1.3, dbl=0.4)
    line(s, r, (-10, gt1), (W + 10, gt1 + 1), 1.8, SEPIA, 1.3, dbl=0.4)
    # the red line
    rl = s.ribbon([(-10, gt1 + 6), (W / 2, gt1 + 7), (W + 10, gt1 + 6)], 9)
    s.wash(rl, PERSIMMON, 0.65, wob=1, soft=0.8, edge=0.5, ew=2, dry=0.2, gran=0.6)
    s.wash(rl, ROSE, 0.2, wob=1, soft=0.8)
    # a manhole cover
    oval(s, 2050, 1690, 130, 22, 1.8, SEPIA, 1.3)
    s.wash(s.ellipse(2050, 1690, 128, 21), PAYNE, 0.35, wob=0.6, soft=1, edge=0.5, ew=3)
    s.hatch(s.ellipse(2050, 1690, 118, 18), 0.0, 6, 0.9, SEPIA, 0.6, skip=0.3)
    # the puddle: the shop's warm light and a bit of sky in it
    pd = s.blob(1420, 1612, 330, 26, irr=0.25, n=14)
    s.wash(pd, None, 0.6, wob=2, soft=1.5, edge=0, lift=True)
    s.wash(pd, OCHRE, 0.35, wob=2, soft=1, edge=0.5, ew=3, grad=s.hgrad(1100, 1750, 1, 0.3))
    s.wash(pd, RAIN, 0.25, wob=2, soft=1, grad=s.hgrad(1150, 1750, 0, 1))
    for xx in (1250, 1340, 1470, 1580):
        s.stroke([(xx, 1594), (xx + J(r, 2), 1630)], U(r, 6, 14), PERSIMMON if xx < 1400 else SIENNA, 0.25, wob=1, soft=2, edge=0, taper=(0.2, 0.4))
    contour(s, pd, 1.2, SEPIA, 0.8, skip=0.3)


def outside(S, r):
    """In front of the shop: a folding table and plastic stools set out for the morning, a stack of stools."""
    s = Sub(S, 480, 1150, 2350, H)
    gy = 1540
    sh = s.ellipse(2010, gy + 4, 210, 14)
    s.wash(sh, PAYNE, 0.35, wob=1, soft=6, edge=0)
    # the table
    tx0, tx1, tt = 1840, 2190, 1310
    for lx in (tx0 + 20, tx1 - 20, tx0 + 60, tx1 - 60):
        k = 1.0 if lx in (tx0 + 20, tx1 - 20) else 0.6
        s.ink([(lx, tt + 10), (lx + J(r, 2), gy - (0 if k == 1.0 else 18))], 4.0 * k, SEPIA, 1.4 * k)
        hl(s, [(lx - 2, tt + 14), (lx - 2, gy - 20)], 1.6, 0.3 * k)
    s.ink([(tx0 + 24, tt + 90), (tx1 - 24, tt + 92)], 2.0, SEPIA, 1.1)
    top = Q(s, [(tx0, tt), (tx1, tt), (tx1 - 26, tt - 18), (tx0 + 26, tt - 18)])
    put(s, top, CREAM, 0.25, lift=0.95, wob=0.6, edge=0.4, ew=3)
    s.wash(top, RAIN, 0.2, wob=1, soft=2, edge=0)
    ed = s.rect(tx0, tt, tx1, tt + 12)
    put(s, ed, MIST, 0.5, lift=0.8, wob=0.5, edge=0.4)
    path(s, [(tx0, tt), (tx0 + 26, tt - 18), (tx1 - 26, tt - 18), (tx1, tt)], 1.6, SEPIA, 1.3, over=2, r=r)
    box(s, r, tx0, tt, tx1, tt + 12, 1.6, SEPIA, 1.3, over=3)
    # a napkin box and a chilli jar on it
    nb = s.rect(1900, tt - 46, 1960, tt - 12)
    put(s, nb, MIST, 0.25, lift=0.9, wob=0.4)
    s.wash(s.rect(1904, tt - 40, 1956, tt - 30), PERSIMMON, 0.5, wob=0.3, soft=0.5)
    box(s, r, 1900, tt - 46, 1960, tt - 12, 1.2, SEPIA, 1.2, over=1)
    s.ink([(1930, tt - 46), (1924, tt - 60), (1936, tt - 66)], 1.0, SEPIA, 0.8)
    stool(s, r, 1790, gy + 10, 1.0, PERSIMMON, 0.8)
    stool(s, r, 2210, gy + 6, 0.95, TEAL, 0.75)
    stool(s, r, 2020, gy - 20, 0.85, PERSIMMON, 0.7)
    stool(s, r, 556, 1500, 0.95, TEAL, 0.7, stack=3)


def scooter(S, r):
    """A parked scooter on the red line, nose to the left: faded blue, a dark seat, a helmet on the mirror and a
    breakfast in a pink bag on the hook."""
    s = Sub(S, 2240, 1040, W, H)
    F, Rw, Rd = (2420, 1566), (2950, 1566), 92
    GY = Rw[1] + Rd
    s.wash(s.ellipse(2690, GY + 2, 380, 16), INK, 0.45, wob=2, soft=8, edge=0)
    s.wash(s.ellipse(2690, GY, 320, 7), INK, 0.6, wob=1, soft=3, edge=0)
    wheel(s, r, Rw[0], Rw[1], Rd)
    wheel(s, r, F[0], F[1], Rd)
    # the fork, behind the fender
    s.ink([(2470, 1300), (2440, 1470), (F[0] + 4, F[1])], 10, SLATE, 0.9)
    s.ink([(2470, 1300), (2440, 1470), (F[0] + 4, F[1])], 1.4, INK, 1.0)
    # engine and swingarm, dark, between the floor and the rear wheel
    eng = s.poly([(2760, 1560), (2800, 1520), (2900, 1500), (2990, 1520), (3000, 1586), (2930, 1610), (2790, 1606)])
    put(s, eng, PAYNE, 0.8, lift=0.85, wob=0.6, edge=0.4, ew=3)
    s.wash(eng, SEPIA, 0.45, wob=1, soft=2)
    for q in range(5):
        s.ink([(2810 + q * 18, 1530), (2806 + q * 18, 1596)], 1.0, INK, 0.8)
    contour(s, eng, 1.4, INK, 1.1)
    BODY = (0.40, 0.55, 0.68)
    front = s.poly([(2494, 1240), (2474, 1256), (2458, 1300), (2446, 1380), (2440, 1440), (2452, 1470), (2500, 1482), (2548, 1500), (2580, 1540), (2596, 1594),
                    (2632, 1596), (2630, 1556), (2606, 1500), (2588, 1440), (2578, 1360), (2570, 1290), (2556, 1248)])
    fender = s.poly([(2322, 1548), (2342, 1494), (2390, 1462), (2450, 1458), (2500, 1480), (2520, 1512), (2490, 1500), (2440, 1488), (2390, 1494), (2350, 1520)])
    floor = s.poly([(2610, 1508), (2790, 1508), (2800, 1586), (2620, 1596)], smooth=False)
    rear = s.poly([(2770, 1590), (2756, 1500), (2770, 1430), (2810, 1388), (2880, 1372), (3010, 1374), (3062, 1396), (3072, 1440), (3072, 1500),
                   (3050, 1500), (3030, 1468), (2990, 1446), (2950, 1442), (2900, 1452), (2862, 1482), (2846, 1530), (2830, 1592)])
    body = np.maximum(np.maximum(front, floor), np.maximum(rear, fender))
    put(s, body, BODY, 0.72, lift=1.0, wob=0.8, edge=0.45, ew=5, flow=0.5, fs=60, gran=0.35)
    s.wash(body, PAYNE, 0.42, wob=1, soft=3, edge=0, grad=s.vgrad(1360, 1600, 0.05, 1))
    s.wash(body, RAIN, 0.2, wob=2, bleed=8, edge=0.1, flow=0.8, fs=40)
    s.wash(s.rect(2610, 1570, 2800, 1600), PAYNE, 0.5, wob=0.6, soft=1.5, clip=floor)
    s.wash(s.rect(2610, 1500, 2790, 1512), MIST, 0.4, wob=0.5, soft=0.8, body=1)                  # the floor's rubber mat edge
    hl(s, [(2452, 1440), (2454, 1360), (2466, 1300), (2486, 1260)], 6, 0.5, dry=0.2)
    s.wash(s.ribbon([(2560, 1270), (2574, 1380), (2596, 1480)], 16), PAYNE, 0.3, wob=1, soft=4, clip=front)
    mottle(s, r, 2440, 1260, 3060, 1580, [PRUSSIAN, VIOLET, MIST], 5, 0.12, size=(40, 90), clip=body)
    pm = s.rect(3018, 1436, 3072, 1470)                                                           # the number plate
    put(s, pm, CREAM, 0.15, lift=0.95, wob=0.4, edge=0.4)
    s.wash(letters(s, "MKA-0816", 3060, 1453, 22, SERIF, 800), INK, 1.2, wob=0.3, soft=0.4)
    box(s, r, 3018, 1436, 3080, 1470, 1.3, INK, 1.2, over=1)
    hl(s, [(2786, 1420), (2830, 1388), (2940, 1380), (3040, 1390)], 5, 0.45, dry=0.2)
    hl(s, [(2350, 1500), (2400, 1470), (2460, 1466)], 3, 0.45)
    contour(s, front, 2.0, INK, 1.4)
    contour(s, rear, 2.0, INK, 1.4)
    contour(s, fender, 1.8, INK, 1.3)
    pl(s, r, [(2610, 1508), (2790, 1508), (2800, 1586), (2620, 1596)], 1.8, INK, 1.3, over=2)
    s.ink([(2800, 1440), (2900, 1420), (3010, 1422)], 1.1, INK, 0.9, skip=0.3)                     # a panel line
    s.ink([(2470, 1330), (2480, 1420), (2530, 1480)], 1.1, INK, 0.8, skip=0.3)
    tl = s.ellipse(3062, 1410, 9, 16)
    put(s, tl, PERSIMMON, 0.8, lift=0.9, wob=0.3, edge=0.5)
    contour(s, tl, 1.2, INK, 1.2)
    # the exhaust across the rear wheel
    ex = s.poly([(2900, 1576), (3060, 1548), (3072, 1572), (2920, 1606)])
    put(s, ex, SLATE, 0.6, lift=0.9, wob=0.5, edge=0.4, ew=2)
    s.wash(ex, INK, 0.45, wob=0.5, soft=2, grad=s.vgrad(1550, 1606, 0, 1))
    hl(s, [(2920, 1578), (3060, 1552)], 3, 0.55)
    contour(s, ex, 1.6, INK, 1.2)
    # the seat
    seat = s.poly([(2776, 1388), (2796, 1350), (2880, 1336), (3010, 1340), (3060, 1360), (3056, 1386), (2900, 1392)])
    put(s, seat, SEPIA, 0.9, lift=0.9, wob=0.5, edge=0.4, ew=3, flow=0.5, fs=30)
    s.wash(seat, INKBLUE, 0.4, wob=1, soft=2, grad=s.vgrad(1336, 1392, 0, 1))
    hl(s, [(2800, 1352), (2880, 1342), (3010, 1346)], 5, 0.35, dry=0.2)
    contour(s, seat, 1.8, INK, 1.4)
    # steering: the handlebar cover with its headlight, the grips, the mirrors
    hb = s.poly([(2456, 1216), (2500, 1188), (2590, 1186), (2612, 1214), (2560, 1240), (2470, 1240)])
    put(s, hb, BODY, 0.8, lift=1.0, wob=0.5, edge=0.45, ew=3)
    s.wash(hb, PAYNE, 0.35, wob=1, soft=2, grad=s.vgrad(1186, 1240, 0, 1))
    hlm = s.ellipse(2470, 1220, 18, 15)
    put(s, hlm, CREAM, 0.15, lift=1.0, wob=0.3, edge=0.4)
    s.lift(s.ellipse(2466, 1216, 8, 6), 0.8, soft=1)
    contour(s, hb, 1.8, INK, 1.4)
    contour(s, hlm, 1.4, INK, 1.2)
    s.stroke([(2500, 1204), (2452, 1196)], 12, INK, 0.9, taper=(0, 0), wob=0.4, edge=0.4, body=1)
    s.stroke([(2590, 1204), (2632, 1208)], 12, INK, 0.9, taper=(0, 0), wob=0.4, edge=0.4, body=1)
    for (bx, by), (mx, my) in (((2490, 1194), (2466, 1100)), ((2590, 1194), (2614, 1104))):
        s.ink([(bx, by), (bx + (mx - bx) * 0.5 + J(r, 2), by + (my - by) * 0.5), (mx, my + 12)], 2.2, INK, 1.4)
        mm = s.ellipse(mx, my, 24, 16, -10)
        put(s, mm, INK, 0.7, lift=0.9, wob=0.4, edge=0.4)
        s.lift(s.ellipse(mx - 4, my - 3, 14, 7, -10), 0.5, soft=2)
        contour(s, mm, 1.4, INK, 1.2)
    # a half-helmet hung on the left mirror's stem
    hm = np.maximum(s.ellipse(2471, 1166, 52, 46) * (1 - s.vgrad(1164, 1168, 0, 1)), s.rect(2419, 1160, 2523, 1172))
    put(s, hm, CREAM, 0.3, lift=1.0, wob=0.5, edge=0.45, ew=3)
    s.wash(hm, SLATE, 0.4, wob=1, soft=2, grad=s.hgrad(2424, 2522, 0, 1))
    s.wash(s.ribbon([(2432, 1154), (2474, 1130), (2516, 1156)], 7), PERSIMMON, 0.8, wob=0.5, soft=0.6)
    contour(s, hm, 1.8, INK, 1.3)
    s.ink([(2430, 1170), (2442, 1210), (2460, 1222)], 1.4, SEPIA, 1.2)
    # breakfast on the hook: a pink bag, a cup and a rice ball showing through
    bag = s.poly([(2596, 1300), (2608, 1298), (2636, 1326), (2644, 1398), (2626, 1420), (2580, 1420), (2564, 1396), (2572, 1328)])
    put(s, bag, ROSE, 0.35, lift=0.95, wob=0.6, edge=0.45, ew=3)
    s.wash(s.rect(2576, 1350, 2608, 1418), SOY, 0.6, wob=0.4, soft=1, clip=bag)
    s.wash(s.ellipse(2622, 1390, 16, 24), CREAM, 0.4, wob=0.4, soft=1, clip=bag)
    s.wash(bag, SLATE, 0.2, wob=1, soft=2, grad=s.hgrad(2566, 2644, 0, 1))
    contour(s, bag, 1.4, SEPIA, 1.2)
    s.ink([(2596, 1300), (2602, 1284), (2612, 1298)], 1.4, SEPIA, 1.2)


def light(S, r):
    """The morning: the sun has only reached the top of the facade; the street and the shop front are in a cool
    shadow; the room's light spills warm onto the counter top and the ground in front."""
    s = S
    x0, y0, x1, y1 = OPEN
    inside = cv2.GaussianBlur(s.rect(x0 + 8, y0 + 8, x1 - 8, y1 - 40), (0, 0), 6)
    sun = 1 - sstep(0, 1, (s.Y - (120 + s.X * 0.06)) / 50.0)
    sh = (1 - sun) * (1 - inside)
    s.wash(sh, RAIN, 0.3, wob=0, soft=0, edge=0, flow=0.5, fs=400, gran=0.25)
    s.wash(sh, VIOLET, 0.08, wob=0, soft=0, edge=0, flow=0.7, fs=300, gran=0.2)
    s.wash(sun, OCHRE, 0.2, wob=0, soft=0, edge=0, flow=0.5, fs=300)
    s.wash(sun, None, 0.12, wob=0, soft=0, edge=0, lift=True)
    # the shadow edge, wet and a little ragged
    edge = np.exp(-((s.Y - (120 + s.X * 0.06)) / 9.0) ** 2) * (s.Y < 700)
    s.wash(edge.astype(np.float32), SLATE, 0.16, wob=3, soft=1, edge=0, flow=0.8, fs=40)
    spill = s.poly([(560, G - 30), (2480, G - 30), (2700, H), (380, H)], smooth=False) * s.vgrad(G, G + 230, 1, 0)
    s.wash(cv2.GaussianBlur(spill, (0, 0), 30), None, 0.4, wob=0, soft=0, edge=0, lift=True, gran=0.3, flow=0.4, fs=80)
    s.wash(cv2.GaussianBlur(spill, (0, 0), 30), OCHRE, 0.2, wob=0, soft=0, edge=0, flow=0.5, fs=90)


def stall_fg():
    """A concrete utility pole close on the left edge: yellow-and-black stripes at its foot, a junction box, step bolts,
    a crust of small-ad stickers, its cables going up out of the picture."""
    s = Off(1100, H, 0, 0, 302)
    r = rs(302)
    xc, wb, wt = 128, 96, 76                       # centre, half-width at the foot and at the top
    hw = lambda y: wt + (wb - wt) * y / H
    pts = [(xc - wt, -10), (xc + wt, -10), (xc + wb, H + 10), (xc - wb, H + 10)]
    pm = Q(s, pts)
    put(s, pm, MIST, 0.55, lift=1.0, wob=1.0, edge=0.4, ew=6, flow=0.5, fs=120, gran=0.6)
    s.wash(pm, SLATE, 0.6, wob=0, soft=2, edge=0, grad=s.hgrad(xc - 20, xc + wb, 0, 1), flow=0.5, fs=90, gran=0.5)
    s.wash(pm, PAYNE, 0.4, wob=0, soft=2, edge=0, grad=s.hgrad(xc + 40, xc + wb, 0, 1), flow=0.5, fs=60)
    s.wash(pm, OCHRE, 0.1, wob=0, soft=2, edge=0, grad=s.hgrad(xc + 50, xc + wb, 0, 1))          # the shop's light, reflected
    s.wash(pm, SLATE, 0.3, wob=0, soft=2, edge=0, grad=s.hgrad(xc - wb, xc - 50, 1, 0))
    s.wash(pm, RAIN, 0.2, wob=3, bleed=12, edge=0.1, flow=0.8, fs=80)
    mottle(s, r, xc - 80, 0, xc + 80, H, [UMBER, SLATE, MOSS, OCHRE], 7, 0.14, size=(30, 90), clip=pm)
    streaks(s, r, xc - 70, xc + 70, 0, H, 14, UMBER, 0.2, (5, 12), clip=pm)
    hl(s, [(xc - 34, 0), (xc - 40, H)], 9, 0.28, dry=0.35, clip=pm)
    # the stripes at its foot
    sm = pm * s.vgrad(1380, 1382, 0, 1)
    s.lift(sm, 0.9, soft=0.8, wob=0)
    s.wash(sm, NOTE, 0.9, wob=0.6, soft=0.8, edge=0.45, ew=4, gran=0.45, flow=0.5, fs=40)
    for i in range(-4, 8):
        y0 = 1380 + i * 70
        band = s.poly([(xc - 120, y0 + 40), (xc - 40, y0 + 30), (xc + 40, y0 + 6), (xc + 120, y0 - 20), (xc + 120, y0 + 15), (xc + 40, y0 + 42), (xc - 40, y0 + 66), (xc - 120, y0 + 75)])
        s.wash(band, INK, 0.9, wob=0.8, soft=0.8, edge=0.45, ew=3, gran=0.5, dry=0.12, clip=sm)
    s.wash(sm, SLATE, 0.35, wob=0, soft=2, grad=s.hgrad(xc - 10, xc + wb, 0, 1))
    s.wash(sm, UMBER, 0.25, wob=2, bleed=8, flow=0.8, fs=30, grad=s.vgrad(1500, H, 0, 1))
    line(s, r, (xc - hw(1380), 1380), (xc + hw(1380), 1378), 1.6, SEPIA, 1.3)
    # step bolts and a junction box with its conduit
    for i, y in enumerate(range(80, 900, 120)):
        sd = -1 if i % 2 else 1
        x = xc + sd * hw(y)
        s.ink([(x - sd * 6, y), (x + sd * 30, y + 2)], 6, SLATE, 1.0)
        s.ink([(x - sd * 6, y + 3), (x + sd * 30, y + 5)], 1.4, INK, 1.2)
        s.ink([(x + sd * 30, y - 6), (x + sd * 30, y + 8)], 3, INK, 1.2)
    jb = s.rect(xc + 40, 330, xc + 170, 520)
    put(s, jb, SLATE, 0.6, lift=0.9, wob=0.8, edge=0.45, ew=4, flow=0.5, fs=40)
    s.wash(jb, PAYNE, 0.35, wob=1, soft=3, grad=s.hgrad(xc + 40, xc + 170, 0.2, 1))
    s.wash(jb, BURNT, 0.2, wob=2, bleed=6, flow=0.8, fs=20, grad=s.vgrad(400, 520, 0, 1))
    hl(s, [(xc + 46, 336), (xc + 164, 336)], 3, 0.4)
    box(s, r, xc + 40, 330, xc + 170, 520, 2.2, INK, 1.4, over=4, dbl=0.4)
    line(s, r, (xc + 52, 352), (xc + 52, 500), 1.1, INK, 0.9, over=1)
    s.wash(s.rect(xc + 140, 410, xc + 152, 440), INK, 0.7, wob=0.3, soft=0.5)
    for y in (360, 490):
        s.ink([(xc + 40, y), (xc + hw(y) - 30, y + 2)], 5, INK, 1.0)
    s.ink([(xc + 100, 520), (xc + 102, 700), (xc + 88, 900)], 9, SLATE, 0.8)
    s.ink([(xc + 96, 520), (xc + 98, 700), (xc + 84, 900)], 1.4, INK, 1.1)
    s.ink([(xc + 106, 520), (xc + 108, 700), (xc + 94, 900)], 1.6, INK, 1.2)
    for y in (300, 560, 760):                       # steel bands
        bm = s.ellipse(xc, y + 7, hw(y) + 4, 12) * 0 + np.clip(s.ellipse(xc, y + 14, hw(y) + 4, 7) + s.rect(xc - hw(y) - 4, y, xc + hw(y) + 4, y + 14) - s.ellipse(xc, y, hw(y) + 4, 7) * s.vgrad(y - 1, y + 1, 1, 0), 0, 1)
        put(s, bm, SLATE, 0.6, lift=0.6, wob=0.4, edge=0.4)
        s.wash(bm, BURNT, 0.25, wob=1, bleed=3, flow=0.8, fs=10)
        s.ink(ell(xc, y, hw(y) + 4, 7, 0.1, math.pi - 0.1, 24), 1.4, INK, 1.2)
        s.ink(ell(xc, y + 14, hw(y) + 4, 7, 0.1, math.pi - 0.1, 24), 1.6, INK, 1.3)
    # the number plate
    pp = s.rect(xc - 50, 800, xc + 30, 880)
    put(s, pp, CREAM, 0.15, lift=0.95, wob=0.4, edge=0.4)
    s.wash(letters(s, "巷口", xc - 10, 822, 22, SERIF, 900), INK, 1.3, wob=0.3, soft=0.4)
    s.wash(letters(s, "12-3", xc - 10, 852, 22, SERIF, 900), INK, 1.3, wob=0.3, soft=0.4)
    box(s, r, xc - 50, 800, xc + 30, 880, 1.6, INK, 1.3, over=2)
    # stickers, layered, torn, faded: drains unblocked, locks opened, septic tanks pumped, a flat to let
    ads = (("通水管", NOTE, PERSIMMON), ("專業開鎖", MIST, PERSIMMON), ("抽水肥", CREAM, INKBLUE), ("搬家", (0.62, 0.78, 0.86), INK),
           ("套房出租", CREAM, PERSIMMON), ("通水管", NOTE, INK), ("收購舊屋", MIST, PERSIMMON), ("開鎖", (0.98, 0.82, 0.80), INK))
    for i, (txt, bgc, fgc) in enumerate(ads):
        y = 940 + i * 54 + J(r, 14)
        x = xc + J(r, 30)
        w_, h_ = 14 + 24 * len(txt), 46
        a = J(r, 5)
        ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
        q = [(x + dx * ca - dy * sa, y + dx * sa + dy * ca) for dx, dy in ((-w_ / 2, -h_ / 2), (w_ / 2, -h_ / 2), (w_ / 2, h_ / 2), (-w_ / 2, h_ / 2))]
        if r.random() < 0.4:                        # torn off at a corner
            q[2] = ((q[2][0] + q[1][0]) / 2 + J(r, 6), (q[2][1] + q[1][1]) / 2)
        m = Q(s, q) * pm
        put(s, m, bgc, U(r, 0.35, 0.6), lift=0.9, wob=0.5, edge=0.4, ew=2)
        s.wash(letters(s, txt, x, y - 8, 22, SERIF, 900, rot=-a), fgc, 1.2, wob=0.3, soft=0.5, clip=m)
        s.wash(letters(s, "09" + str(10 + i * 7) + "-" + str(300 + i * 91) + "-" + str(100 + i * 53), x, y + 12, 13, SERIF, 700, rot=-a), fgc, 1.0, wob=0.2, soft=0.4, clip=m)
        s.wash(m, SLATE, 0.18, wob=1, soft=2, grad=s.hgrad(xc, xc + wb, 0, 1))
        s.lift(m * s.noise(18, 2), U(r, 0.1, 0.3), soft=1, wob=0)          # faded
        path(s, q, 1.1, SEPIA, 1.0, closed=True, over=1, r=r)
    for _ in range(10):                             # the ghosts of stickers scraped off
        x, y = xc + J(r, 60), U(r, 900, 1350)
        m = s.rect(x - U(r, 20, 50), y - 14, x + U(r, 20, 50), y + 14) * pm
        s.lift(m, 0.25, soft=1, wob=1)
        s.wash(m, CREAM, 0.15, wob=1, soft=1, edge=0.6, ew=2, dry=0.4)
    # the pole's edges
    line(s, r, (xc - wt, -10), (xc - wb, H + 10), 2.4, INK, 1.5, dbl=0.5)
    line(s, r, (xc + wt, -10), (xc + wb, H + 10), 2.8, INK, 1.6, dbl=0.5)
    # its cables: a bundle that leaves for the right, one for the left, a drop to the box
    for i in range(5):
        y0 = 150 + i * 12
        cable(s, r, (xc + wt - 4, y0), (1110, -60 + i * 26), 120 + i * 20, 2.6 - i * 0.3, INK, 1.3)
    for i in range(2):
        cable(s, r, (xc - wt + 4, 210 + i * 16), (-20, 120 + i * 30), 40, 2.2, INK, 1.3)
    for y in (150, 210):
        s.ink([(xc - wt - 8, y + 6), (xc + wt + 8, y + 6)], 5, INK, 1.0)
    s.ink([(xc + 120, 330), (xc + 170, 240), (xc + 260, 170), (xc + 380, 140)], 2.2, INK, 1.3)
    s.save("st_fg", crop=True, anchor=(xc, H), q=92)


def stall_hands():
    """The boss behind the soy milk pot, small at the distance of the lane: the white T-shirt's shoulders, the dark
    apron's straps, the left hand round the ladle, the right resting on the rim. Above the shoulders the figure
    goes into the steam (painted in st_bg); below the rim it is hidden by the pot (the sprite is clipped there)."""
    ox, oy = 930, 800
    s = Off(300, 340, ox, oy, 303)
    r = rs(303)
    cx, rim = POT[0], POT[1]
    bx = 1092                                       # the boss's middle
    shoulder = 984
    # what the pot hides: below its back rim, inside its outline (the front-rim hand and the ladle are drawn over)
    potm = np.maximum(s.rect(cx - POTRX - 3, rim - 8, cx + POTRX + 3, 1200), s.ellipse(cx, rim, POTRX + 3, 10))
    vis = 1 - cv2.GaussianBlur(potm, (0, 0), 0.8)
    fade = s.vgrad(952, 976, 0, 1)
    # the body: a T-shirt, the apron's bib and straps
    torso = s.poly([(bx - 58, shoulder + 8), (bx - 40, shoulder - 4), (bx - 12, shoulder - 8), (bx + 12, shoulder - 8), (bx + 40, shoulder - 4), (bx + 58, shoulder + 8),
                    (bx + 60, 1080), (bx - 60, 1080)])
    neck = s.ellipse(bx, shoulder - 14, 13, 16)
    s.wash(neck * fade * vis, SKIN, 0.5, wob=0.6, soft=1, edge=0.4, ew=2, body=0.9)
    s.wash(neck * fade * vis, BURNT, 0.25, wob=0.6, soft=2, grad=s.vgrad(shoulder - 24, shoulder - 6, 0, 1))
    tm = torso * fade * vis
    s.wash(tm, CREAM, 0.16, wob=0.6, soft=0.8, edge=0.4, ew=2, body=1)
    s.wash(tm, RAIN, 0.3, wob=1, soft=2, grad=s.hgrad(bx - 60, bx + 60, 0.2, 1))
    s.wash(tm, OCHRE, 0.2, wob=1, soft=3, grad=s.hgrad(bx - 60, bx, 1, 0))
    bib = s.poly([(bx - 26, shoulder + 14), (bx + 26, shoulder + 14), (bx + 30, 1080), (bx - 30, 1080)]) * fade * vis
    s.wash(bib, NAVY, 0.5, wob=0.6, soft=0.8, edge=0.45, ew=2, body=1)
    s.wash(bib, INDIGO, 0.25, wob=1, soft=2, grad=s.hgrad(bx - 30, bx + 30, 0, 1))
    for sd in (-1, 1):
        st = s.ribbon([(bx + sd * 22, shoulder + 16), (bx + sd * 20, shoulder), (bx + sd * 13, shoulder - 14), (bx + sd * 6, shoulder - 28)], 5) * s.vgrad(944, 962, 0, 1) * vis
        s.wash(st, NAVY, 0.8, wob=0.4, soft=0.6, edge=0.4, body=1)
    # sleeves on both shoulders
    for sd in (-1, 1):
        sl = s.poly([(bx + sd * 40, shoulder - 4), (bx + sd * 62, shoulder + 6), (bx + sd * 72, shoulder + 38), (bx + sd * 48, shoulder + 42), (bx + sd * 40, shoulder + 14)]) * vis
        s.wash(sl, CREAM, 0.2, wob=0.6, soft=0.8, edge=0.45, ew=2, body=1)
        s.wash(sl, RAIN, 0.35, wob=1, soft=2, grad=s.vgrad(shoulder, shoulder + 40, 0, 1))
        contour(s, sl * fade, 1.1, SEPIA, 1.0)
        arm = s.poly([(bx + sd * 50, shoulder + 40), (bx + sd * 70, shoulder + 38), (bx + sd * 74, 1080), (bx + sd * 52, 1080)]) * vis
        s.wash(arm, SKIN, 0.55, wob=0.6, soft=0.8, edge=0.4, ew=2, body=1)
        s.wash(arm, BURNT, 0.3, wob=1, soft=2, grad=s.hgrad(bx + sd * 50, bx + sd * 74, 0, 1) if sd > 0 else s.hgrad(bx + sd * 74, bx + sd * 50, 1, 0))
        contour(s, arm, 1.0, SEPIA, 0.9)
    contour(s, tm * s.vgrad(970, 978, 0, 1), 1.1, SEPIA, 0.9)
    # the ladle: a long steel handle out of the milk; the left hand round it
    hx0, hy0 = 1094, 1046                       # where the handle leaves the milk
    top = (1156, 968)
    s.ink([(hx0, hy0), ((hx0 + top[0]) / 2 + 2, (hy0 + top[1]) / 2), top], 3.2, SLATE, 1.0, taper=(0, 0))
    s.ink([(hx0 - 1, hy0), ((hx0 + top[0]) / 2 + 1, (hy0 + top[1]) / 2), (top[0] - 1, top[1])], 1.0, INK, 1.2, taper=(0, 0.1))
    hl(s, [(hx0 + 1, hy0 - 4), (top[0] + 1, top[1] + 2)], 1.0, 0.5)
    s.ink(ell(top[0] + 4, top[1] - 2, 5, 4, 0, 6.3, 12), 1.2, INK, 1.0)           # the hook at its end
    g = (1134, 1004)                            # the grip
    fore = s.ribbon([(1168, 1046), (1158, 1026), (g[0] + 8, g[1] + 6)], 15, profile=lambda t: 1 - 0.25 * t) * vis
    fist = s.blob(g[0], g[1], 11, 10, irr=0.12, n=9, seed=7)
    thumb = finger(s, [(g[0] - 4, g[1] - 6), (g[0] - 12, g[1] - 9), (g[0] - 16, g[1] - 5)], 6)
    hm = np.maximum(np.maximum(fore, fist), thumb)
    s.wash(hm, SKIN, 0.6, wob=0.5, soft=0.7, edge=0.4, ew=2, body=1)
    s.wash(hm, BURNT, 0.3, wob=0.8, soft=1.5, grad=s.vgrad(g[1] - 10, 1046, 0, 1))
    s.wash(hm, OCHRE, 0.2, wob=0.8, soft=2, grad=s.hgrad(g[0] - 20, g[0] + 10, 1, 0))
    contour(s, hm, 1.2, SEPIA, 1.2)
    for q in range(3):                          # knuckles
        s.ink([(g[0] - 6 + q * 4, g[1] + 4), (g[0] - 4 + q * 4, g[1] + 9)], 0.8, SEPIA, 0.8)
    s.ink([(top[0] - 9, top[1] + 16), ((top[0] + g[0]) / 2 - 3, (top[1] + g[1]) / 2 + 2)], 1.0, INK, 1.0)        # the handle over the fist
    # the right hand on the rim, fingers over its lip
    h2 = (1010, rim - 2)
    palm = s.blob(h2[0], h2[1] - 3, 13, 7, irr=0.12, n=9, seed=8)
    fs_ = [finger(s, [(h2[0] - 9 + i * 6, h2[1]), (h2[0] - 10 + i * 6, h2[1] + 7), (h2[0] - 10 + i * 6, h2[1] + 12)], 5.2) for i in range(4)]
    wrist = s.ribbon([(h2[0] + 10, h2[1] - 4), (h2[0] + 24, h2[1] - 16), (h2[0] + 30, h2[1] - 30)], 12) * vis
    hm2 = np.maximum(palm, wrist)
    for f in fs_:
        hm2 = np.maximum(hm2, f)
    s.wash(hm2, SKIN, 0.6, wob=0.5, soft=0.7, edge=0.4, ew=2, body=1)
    s.wash(hm2, BURNT, 0.28, wob=0.8, soft=1.5, grad=s.vgrad(h2[1] - 8, h2[1] + 12, 0, 1))
    contour(s, hm2, 1.1, SEPIA, 1.1)
    for i in range(1, 4):
        s.ink([(h2[0] - 12 + i * 6, h2[1] + 1), (h2[0] - 12 + i * 6, h2[1] + 10)], 0.7, SEPIA, 0.8)
    # the steam the head is lost in: a soft veil of nearly-paper, opaque enough to hide the cut
    veil = np.zeros((s.h, s.w), np.float32)
    for i in range(7):
        vx, vy = bx + J(r, 24), 966 + J(r, 10) - i * 13
        veil = np.maximum(veil, s.blob(vx, vy, U(r, 34, 54), U(r, 20, 30), irr=0.3, n=11))
    veil = cv2.GaussianBlur(veil, (0, 0), 6)
    veil = np.clip(veil * 1.6, 0, 1) * (1 - s.vgrad(984, 998, 0, 1)) * s.vgrad(850, 935, 0, 1)
    s.A *= (1 - veil * 0.95)[..., None]
    s.body = np.maximum(s.body, veil * 0.92)
    s.wash(veil, MIST, 0.06, wob=0, soft=0, edge=0)
    ax, ay = BOWL
    s.save("st_hands", crop=True, anchor=(ax - ox, ay - oy), meta={"cuff": [bx + 66 - ax, shoulder + 40 - ay], "grip": [g[0] - ax, g[1] - ay], "handle": [top[0] - ax, top[1] - ay]})


def soy_bg():
    """The counter top close: brushed stainless with the room lying soft in it, a pale blue plastic tray, and the
    kitchen behind out of focus (wet in wet, no pen): steamers and the soy pot on the left in a warm light and steam,
    the open front on the right with the lane's cool daylight."""
    SW, SH = 2560, 1440
    S = Sheet(SW, SH, 311)
    r = rs(311)
    BACK = SY_BACK
    # ---- the kitchen, out of focus
    k = Sub(S, 0, 0, SW, BACK + 2)
    F = k.full()
    k.wash(F, OCHRE, 0.55, wob=0, soft=0, edge=0, flow=0.6, fs=400, gran=0.15)
    k.wash(F, SIENNA, 0.4, wob=0, soft=0, edge=0, grad=k.rgrad(700, 300, 300, 1400, 0, 1), flow=0.6, fs=300, gran=0.15)
    k.wash(F, UMBER, 0.35, wob=0, soft=0, edge=0, grad=k.vgrad(-100, 300, 1, 0), flow=0.6, fs=300, gran=0.15)
    # the open front: cool grey-blue daylight, the lane's far wall and a scooter going by as blurs
    of = k.blob(2180, 330, 520, 420, irr=0.12, n=14)
    k.wash(of, None, 0.75, wob=10, bleed=40, edge=0, gran=0.1, lift=True)
    k.wash(of, RAIN, 0.45, wob=10, bleed=40, edge=0.05, flow=0.6, fs=200)
    k.wash(k.blob(2300, 420, 220, 200, irr=0.3), CELADON, 0.3, wob=8, bleed=30, edge=0.05)
    k.wash(k.blob(2010, 560, 180, 60, irr=0.3), SLATE, 0.35, wob=8, bleed=24, edge=0.05)
    k.wash(k.rect(1600, -20, 1700, BACK), UMBER, 0.35, wob=8, bleed=44, edge=0)                     # the pier
    # tiles, white, above the stove
    k.wash(k.blob(980, 260, 420, 220, irr=0.2), None, 0.3, wob=6, bleed=30, edge=0, lift=True)
    # the red menu boards along the top, as blurs
    for i in range(6):
        x = 120 + i * 230 + J(r, 20)
        m = k.blob(x, 70, 70, 130, irr=0.15)
        k.wash(m, None, 0.6, wob=6, bleed=22, edge=0, lift=True)
        k.wash(m, CREAM, 0.2, wob=6, bleed=22, edge=0.05)
        k.wash(k.blob(x, 60, 40, 90, irr=0.15), PERSIMMON, 0.45, wob=6, bleed=34, edge=0)
    # a lamp
    k.wash(k.rgrad(1280, 120, 20, 380, 1, 0), None, 0.6, wob=0, soft=0, edge=0, lift=True)
    k.wash(k.rgrad(1280, 120, 10, 300, 1, 0), AMBER, 0.3, wob=0, soft=0, edge=0)
    # steamer stacks and the pot
    for cx, w, top, n in ((300, 300, 170, 6), (720, 240, 330, 4)):
        st = k.blob(cx, (top + BACK) / 2, w / 2, (BACK - top) / 2 + 20, irr=0.06, n=16)
        k.wash(st, OCHRE, 0.55, wob=8, bleed=40, edge=0.1, flow=0.6, fs=100)
        k.wash(st, SIENNA, 0.35, wob=8, bleed=40, edge=0.05, grad=k.hgrad(cx - w / 2, cx + w / 2, 0, 1))
        for j in range(n):
            y = top + (BACK - top) * j / n + 30
            k.wash(k.blob(cx, y, w * 0.45, 12, irr=0.1), BURNT, 0.18, wob=6, bleed=22, edge=0)
    pt = k.blob(1060, 560, 190, 200, irr=0.08, n=16)
    k.wash(pt, None, 0.6, wob=6, bleed=24, edge=0, lift=True)
    k.wash(pt, MIST, 0.4, wob=6, bleed=24, edge=0.1)
    k.wash(pt, SLATE, 0.5, wob=6, bleed=24, edge=0.05, grad=k.hgrad(880, 1250, 0, 1))
    k.wash(k.blob(980, 560, 30, 180, irr=0.2), None, 0.5, wob=6, bleed=14, edge=0, lift=True)
    # a dark doorway at the back, the boss's apron hanging on a hook
    k.wash(k.blob(1460, 380, 110, 300, irr=0.15), UMBER, 0.3, wob=8, bleed=44, edge=0)
    for _ in range(16):                             # out-of-focus lights: soft discs
        bx, by_, br = U(r, 0, 1700), U(r, 40, BACK - 60), U(r, 24, 60)
        d = k.ellipse(bx, by_, br, br)
        k.lift(d, U(r, 0.25, 0.5), soft=br * 0.25, wob=0)
        k.wash(d, (AMBER, OCHRE, CREAM)[int(r.integers(0, 3))], U(r, 0.1, 0.25), wob=0.5, soft=br * 0.2, edge=0.35, ew=3)
    # steam, lots of it
    for cx, cy, rx, ry, a in ((300, 120, 260, 160, 0.5), (720, 260, 200, 140, 0.45), (1060, 330, 240, 200, 0.5), (500, 380, 300, 120, 0.3), (1200, 160, 300, 120, 0.3)):
        k.wash(k.blob(cx, cy, rx, ry, irr=0.4, n=13), None, a, wob=10, bleed=34, edge=0, lift=True, gran=0.1)
    k.wash(F, None, 0.25, wob=0, soft=0, edge=0, lift=True, grad=k.vgrad(BACK - 260, BACK, 0, 1))      # mist down to the counter
    # ---- the counter top: brushed steel
    c = Sub(S, 0, BACK - 20, SW, SH)
    cm = c.rect(-10, BACK, SW + 10, SH + 10)
    c.wash(cm, MIST, 0.5, wob=0.6, soft=0.8, edge=0.3, ew=6, flow=0.5, fs=400, gran=0.2, body=1)
    c.wash(cm, SLATE, 0.45, wob=0, soft=0, edge=0, grad=c.vgrad(BACK, SH, 0.1, 1), flow=0.6, fs=300, gran=0.2)
    # the room lying in it: soft vertical smears of what is behind, fading toward us
    for x, w, col, a in ((300, 300, OCHRE, 0.5), (720, 240, OCHRE, 0.45), (1060, 300, MIST, 0.0), (1280, 160, AMBER, 0.35), (2180, 700, RAIN, 0.4), (130, 200, PERSIMMON, 0.2)):
        m = c.blob(x, BACK + 160, w / 2, 220, irr=0.2)
        g = c.vgrad(BACK, BACK + 420, 1, 0)
        if a:
            c.wash(m, col, a, wob=10, bleed=30, edge=0, grad=g, clip=cm)
        else:
            c.wash(m * g, None, 0.4, wob=10, bleed=30, edge=0, lift=True, clip=cm)
    c.wash(c.blob(2180, BACK + 100, 400, 120, irr=0.2), None, 0.45, wob=10, bleed=30, edge=0, lift=True, clip=cm)
    # the brushing: long thin streaks, a little converging toward the back
    for i in range(170):
        y = U(r, BACK + 8, SH)
        xa = U(r, -200, SW)
        L = U(r, 300, 1400)
        f = (y - BACK) / (SH - BACK)
        c.stroke([(xa, y), (xa + L, y + J(r, 1.5))], U(r, 1.5, 4) * (0.5 + f), SLATE if r.random() < 0.6 else MIST, U(r, 0.05, 0.16), dry=0.5,
                 edge=0.1, wob=0.4, taper=(0.2, 0.2), gran=0.3)
        if r.random() < 0.4:
            hl(c, [(xa, y + 3), (xa + L * 0.7, y + 3)], U(r, 1, 3) * (0.5 + f), U(r, 0.08, 0.2), dry=0.4)
    # the back edge of the counter: a rolled lip catching light
    hl(c, [(-10, BACK + 4), (SW + 10, BACK + 4)], 6, 0.7)
    c.wash(c.rect(-10, BACK - 2, SW + 10, BACK + 1), SEPIA, 0.3, wob=0.3, soft=1.2)
    line(c, r, (-10, BACK + 9), (SW + 10, BACK + 9), 1.2, SEPIA, 0.6, skip=0.4)
    # ---- the tray, pale blue plastic, in perspective
    tq = [(SY_TRAY[0] + 60, SY_TRAY[1]), (SY_TRAY[2] - 60, SY_TRAY[1]), (SY_TRAY[2], SY_TRAY[3]), (SY_TRAY[0], SY_TRAY[3])]
    tm = c.poly(tq, smooth=False)
    tm = cv2.erode(tm, np.ones((3, 3), np.uint8))
    rim = c.poly([(tq[0][0] - 10, tq[0][1] - 12), (tq[1][0] + 10, tq[1][1] - 12), (tq[2][0] + 14, tq[2][1] + 6), (tq[3][0] - 14, tq[3][1] + 6)], smooth=False)
    c.wash(c.poly([(x + 30, y + 26) for x, y in [(tq[0][0] - 10, tq[0][1] - 12), (tq[1][0] + 10, tq[1][1] - 12), (tq[2][0] + 14, tq[2][1] + 6), (tq[3][0] - 14, tq[3][1] + 6)]], smooth=False),
           INKBLUE, 0.3, wob=2, bleed=14, edge=0)
    put(c, rim, (0.66, 0.78, 0.86), 0.7, lift=0.8, wob=0.8, edge=0.45, ew=5, flow=0.4, fs=200, gran=0.3)
    c.wash(tm, (0.66, 0.78, 0.86), 0.35, wob=0.8, soft=1, edge=0.4, ew=6)
    c.wash(rim - tm * 0.95, PRUSSIAN, 0.2, wob=0.5, soft=1, edge=0)
    c.wash(tm, OCHRE, 0.12, wob=6, bleed=30, edge=0, grad=c.vgrad(SY_TRAY[1], SY_TRAY[3], 1, 0.2))
    for i in range(1, 9):                          # the ribs on its floor
        q = i / 9
        a = (tq[0][0] + (tq[1][0] - tq[0][0]) * q, tq[0][1] + 8)
        b = (tq[3][0] + (tq[2][0] - tq[3][0]) * q, tq[3][1] - 8)
        hl(c, [a, b], 3, 0.12, clip=tm)
        seg(c, (a[0] + 3, a[1]), (b[0] + 4, b[1]), 0.9, PRUSSIAN, 0.2, r=r, skip=0.5)
    hl(c, [(tq[3][0] + 6, tq[3][1] + 2), (tq[2][0] - 6, tq[2][1] + 2)], 4, 0.6)
    pl(c, r, [(tq[0][0] - 10, tq[0][1] - 12), (tq[1][0] + 10, tq[1][1] - 12), (tq[2][0] + 14, tq[2][1] + 6), (tq[3][0] - 14, tq[3][1] + 6)], 2.4, SEPIA, 1.3, over=5, closed=True)
    pl(c, r, tq, 1.4, SEPIA, 0.9, over=3, closed=True, skip=0.15)
    # a few drops of soy milk on the steel, a receipt, the cup's soft reflection
    for (x, y, rx_) in ((1690, 1360, 14), (1730, 1384, 7), (560, 1300, 10)):
        d = c.ellipse(x, y, rx_, rx_ * 0.45)
        c.wash(d, SOY, 0.8, wob=0.5, soft=0.8, edge=0.6, ew=2)
        c.lift(c.ellipse(x - rx_ * 0.3, y - rx_ * 0.1, rx_ * 0.3, rx_ * 0.15), 0.7, soft=0.6)
    rp = c.poly([(140, 1120), (420, 1090), (480, 1300), (190, 1336)], smooth=False)
    put(c, rp, CREAM, 0.12, lift=0.95, wob=0.6, edge=0.35, ew=3)
    c.wash(rp, SLATE, 0.18, wob=2, bleed=10, grad=c.vgrad(1090, 1340, 0, 1))
    for i in range(7):
        y0 = 1150 + i * 22
        seg(c, (190 + i * 4, y0), (380 + i * 6, y0 - 26), 1.2, SEPIA, U(r, 0.5, 0.9), r=r, skip=0.25)
    paint_letters(c, letters(c, "豆漿 15", 285, 1125, 24, SERIF, 800, rot=6), INK, 1.0, wob=0.3)
    contour(c, rp, 1.6, SEPIA, 1.1)
    cx, cy = SY_CUP
    c.wash(c.blob(cx, cy + 70, 150, 80, irr=0.15), INKBLUE, 0.3, wob=2, bleed=18, edge=0)          # under the cup: shadow, then its reflection
    c.wash(c.blob(cx, cy + 120, 120, 110, irr=0.2), None, 0.25, wob=4, bleed=24, edge=0, lift=True)
    c.wash(c.blob(cx, cy + 110, 110, 90, irr=0.2), SOY, 0.35, wob=4, bleed=24, edge=0)
    S.save("sy_bg", opaque=True, q=90, meta={"counter_y": BACK, "tray": list(SY_TRAY), "cup": list(SY_CUP), "straw": list(SY_STRAW),
                                             "steam": [[300, 160], [720, 320], [1060, 380]]})


def soy_cup():
    """A clear plastic cup of hot soy milk with a sealed printed film lid, seen a little from above.
    Anchor = the middle of its foot on the counter."""
    s = Off(760, 940, 0, 0, 312)
    r = rs(312)
    cx, by, ty = 380, 860, 160               # foot centre, rim centre
    rt, rb, ryt, ryb = 214, 150, 60, 50
    fill = 0.86
    yf = by - (by - ty) * fill
    rf = rb + (rt - rb) * fill
    side = lambda y, sd: cx + sd * (rb + (rt - rb) * (by - y) / (by - ty))
    # its shadow and its reflection on the steel
    s.wash(s.ellipse(cx + 20, by + 18, rb * 1.25, ryb * 0.9), INKBLUE, 0.35, wob=2, soft=10, edge=0)
    body = s.poly([(side(ty, -1), ty), (side(by, -1), by), (side(by, 1), by), (side(ty, 1), ty)], smooth=False)
    body = np.maximum(np.maximum(body, s.ellipse(cx, by, rb, ryb)), s.ellipse(cx, ty, rt, ryt))
    # the milk, seen through the plastic
    milk = np.maximum(s.poly([(side(yf, -1) + 4, yf), (side(by, -1) + 5, by - 8), (side(by, 1) - 5, by - 8), (side(yf, 1) - 4, yf)], smooth=False),
                      s.ellipse(cx, by - 8, rb - 5, ryb - 6))
    milk = np.maximum(milk, s.ellipse(cx, yf, rf - 4, ryt * 0.93))
    s.lift(body, 1.0, soft=0.5, wob=0)
    s.wash(milk, SOY, 0.62, wob=0.8, soft=0.8, edge=0.5, ew=6, flow=0.4, fs=80, gran=0.3, body=1)
    s.wash(milk, CREAM, 0.25, wob=2, bleed=14, edge=0.1, grad=s.hgrad(cx - rt, cx + rt, 0.3, 1))
    s.wash(milk, RAIN, 0.3, wob=2, bleed=16, edge=0, grad=s.hgrad(cx - rt, cx - rt * 0.3, 1, 0))
    s.wash(milk, SLATE, 0.12, wob=2, bleed=16, edge=0, grad=s.vgrad(by - 160, by, 0, 1))
    s.wash(milk, OCHRE, 0.14, wob=1, soft=4, edge=0, grad=s.hgrad(cx + rt * 0.5, cx + rt, 0, 1))         # the kitchen's light through it
    s.lift(s.ellipse(cx - 30, yf + 6, rf * 0.7, ryt * 0.5), 0.25, soft=6, wob=2)                          # the skin on top, catching the light
    s.wash(s.ellipse(cx, yf, rf - 4, ryt * 0.93) - s.ellipse(cx, yf - 4, rf - 12, ryt * 0.8), OCHRE, 0.25, wob=0.5, soft=1.5, edge=0)
    # the empty band under the lid: fogged, beaded with condensation
    fog = np.clip(body - milk, 0, 1) * s.vgrad(ty - 70, ty + 10, 0, 1)
    fog = np.maximum(fog, np.clip(body - milk, 0, 1) * s.vgrad(ty, yf + 30, 1, 1))
    s.wash(fog, MIST, 0.35, wob=1, soft=2, edge=0.4, ew=4, body=0.85)
    s.wash(fog, SOY, 0.25, wob=2, bleed=8, grad=s.vgrad(ty, yf, 0, 1))
    for _ in range(140):
        x = U(r, cx - rt + 14, cx + rt - 14)
        y = U(r, ty + ryt * 0.6, yf + 20)
        rr = U(r, 2.0, 7.5) ** 1.0
        if abs(x - cx) > (side(y, 1) - cx) - 10:
            continue
        d = s.ellipse(x, y, rr, rr * 1.1)
        s.wash(d, SLATE, 0.45, wob=0.3, soft=0.5, edge=0.8, ew=1.2)
        s.lift(s.ellipse(x - rr * 0.3, y - rr * 0.35, rr * 0.35, rr * 0.3), 0.9, soft=0.4, wob=0)
    for _ in range(6):                               # beads that ran
        x = U(r, cx - rt * 0.8, cx + rt * 0.8)
        y0 = U(r, yf - 10, yf + 60)
        L = U(r, 60, 220)
        s.stroke([(x, y0), (x + J(r, 3), y0 + L * 0.5), (x + J(r, 4), y0 + L)], U(r, 4, 7), SLATE, 0.28, edge=0.7, wob=0.5, taper=(0.1, 0.05), gran=0.1)
        s.lift(s.ribbon([(x - 1.5, y0 + 4), (x - 1.5 + J(r, 2), y0 + L - 4)], 1.6), 0.7, soft=0.5, wob=0.3)
        d = s.ellipse(x + J(r, 3), y0 + L + 2, 5, 6)
        s.wash(d, SLATE, 0.5, wob=0.3, soft=0.5, edge=0.8, ew=1.2)
        s.lift(s.ellipse(x - 1, y0 + L, 1.8, 1.8), 0.9, soft=0.3)
    # the plastic itself: two upright highlights, the shoulder ridge, the thick foot
    for q, w_, a in ((-0.62, 26, 0.75), (-0.48, 8, 0.6), (0.7, 12, 0.55)):
        pts = [(side(ty + 30, 1 if q > 0 else -1) * 0 + cx + q * rt * 0.98, ty + ryt + 10), (cx + q * (rb + (rt - rb) * 0.5) * 0.98, (ty + by) / 2), (cx + q * rb * 0.98, by - 30)]
        s.lift(s.ribbon(pts, w_, taper=(0.25, 0.35)), a, soft=1.5, wob=1)
    s.wash(s.ribbon([(side(by, -1) + 6, by - 10), (side(ty, -1) + 4, ty + 10)], 12), SLATE, 0.3, wob=1, soft=2, edge=0.2)
    s.wash(s.ribbon([(side(by, 1) - 6, by - 10), (side(ty, 1) - 4, ty + 10)], 10), UMBER, 0.25, wob=1, soft=2, edge=0.2)
    yr = ty + 52
    s.ink(ell(cx, yr, side(yr, 1) - cx, ryt * 0.95, 0.08, math.pi - 0.08, 40), 1.3, SEPIA, 0.8, skip=0.15)
    s.lift(s.ribbon(list(ell(cx, yr + 6, side(yr, 1) - cx - 6, ryt * 0.92, 0.5, 2.6, 20)), 3), 0.6, soft=1, wob=0.5)
    foot = s.ellipse(cx, by, rb, ryb) * (1 - s.ellipse(cx, by - 12, rb - 8, ryb - 10))
    s.wash(foot, MIST, 0.18, wob=0.4, soft=1, edge=0.5, ew=2, body=0.9)
    s.ink(ell(cx, by - 12, rb - 8, ryb - 10, 0.15, math.pi - 0.15, 30), 1.2, SEPIA, 0.8)
    # the sealed lid: a printed film over the rolled rim
    lid = s.ellipse(cx, ty, rt + 4, ryt + 2)
    put(s, lid, CREAM, 0.08, lift=1.0, wob=0.4, edge=0.3, ew=3)
    ring = lid * (1 - s.ellipse(cx, ty, (rt + 4) * 0.84, (ryt + 2) * 0.84))
    s.wash(ring, PERSIMMON, 0.9, wob=0.5, soft=0.6, edge=0.5, ew=2, gran=0.2)
    s.wash(ring, ROSE, 0.25, wob=0.5, soft=0.6)
    thin = s.ellipse(cx, ty, (rt + 4) * 0.79, (ryt + 2) * 0.79) * (1 - s.ellipse(cx, ty, (rt + 4) * 0.76, (ryt + 2) * 0.76))
    s.wash(thin, INKBLUE, 0.9, wob=0.4, soft=0.5)
    for i in range(16):                              # a ring of little blue beans
        a = i * 2 * math.pi / 16 + 0.1
        bx, by_ = cx + math.cos(a) * rt * 0.66, ty + math.sin(a) * ryt * 0.66
        bm = s.ellipse(bx, by_, 9, 4.5, math.degrees(a) + 90)
        s.wash(bm, INKBLUE, 0.85, wob=0.3, soft=0.5, edge=0.5, ew=1.2)
    for i in range(24):                              # and white dashes in the red
        a = i * 2 * math.pi / 24
        s.lift(s.ribbon([(cx + math.cos(a) * rt * 0.88, ty + math.sin(a) * ryt * 0.88), (cx + math.cos(a + 0.12) * rt * 0.88, ty + math.sin(a + 0.12) * ryt * 0.88)], 2.4, smooth=False), 0.85, soft=0.4, wob=0)
    s.lift(s.ribbon([(cx - rt * 0.5, ty - ryt * 0.35), (cx - rt * 0.1, ty - ryt * 0.62), (cx + rt * 0.3, ty - ryt * 0.7)], 10, taper=(0.4, 0.5)), 0.6, soft=3, wob=1)
    s.wash(lid, SLATE, 0.12, wob=1, soft=4, grad=s.hgrad(cx - rt, cx + rt, 1, 0))
    hl(s, list(ell(cx, ty, rt + 4, ryt + 2, 0.15, math.pi - 0.15, 30)), 3, 0.7)                   # the rolled rim
    # pen, loose
    oval(s, cx, ty, rt + 4, ryt + 2, 2.4, SEPIA, 1.4, twice=True)
    s.ink([(side(ty, -1), ty + 2), (side(by, -1) + J(r, 2), by + 6)], 2.6, SEPIA, 1.5, taper=(0.05, 0.3))
    s.ink([(side(ty, 1), ty + 2), (side(by, 1) + J(r, 2), by + 4)], 2.4, SEPIA, 1.4, taper=(0.05, 0.3))
    s.ink(ell(cx, by, rb, ryb, 0.05, math.pi - 0.05, 40), 2.4, SEPIA, 1.4)
    s.ink(ell(cx, yf, rf - 4, ryt * 0.93, 0.15, math.pi - 0.15, 40), 1.2, SIENNA, 0.8, skip=0.2)
    s.save("sy_cup", crop=True, anchor=(cx, by), q=92, meta={"lid": [0, ty - by, rt + 4, ryt + 2], "steam": [-int(rt * 0.75), int(ty - by - 10)]})


def soy_straw():
    """A straw in its paper wrapper, lying flat. Anchor = its middle."""
    s = Off(960, 120, 0, 0, 313)
    r = rs(313)
    cy, x0, x1, th = 54, 50, 910, 30
    s.wash(s.ribbon([(x0 + 30, cy + 20), (x1 - 20, cy + 22)], 22, smooth=False), INKBLUE, 0.3, wob=1, soft=6, edge=0)
    pts = [(x0 + 22, cy), (x0 + 300, cy - 1), (x1 - 300, cy + 1), (x1 - 22, cy)]
    m = s.ribbon(pts, th)
    for xe, sd in ((x0, -1), (x1, 1)):          # the twisted paper ends
        tw = s.poly([(xe - sd * 28, cy - th / 2), (xe + sd * 4, cy - 5), (xe + sd * 14, cy - 2 + J(r, 2)), (xe + sd * 14, cy + 3), (xe + sd * 4, cy + 6), (xe - sd * 28, cy + th / 2)])
        m = np.maximum(m, tw)
    put(s, m, CREAM, 0.12, lift=1.0, wob=0.5, edge=0.35, ew=3)
    s.wash(m, SLATE, 0.3, wob=1, soft=3, grad=s.vgrad(cy - th / 2, cy + th / 2, 0, 1))
    s.wash(m, OCHRE, 0.12, wob=1, soft=3, grad=s.vgrad(cy - th / 2, cy, 1, 0))
    x = x0 + 30
    while x < x1 - 40:                          # the printed spiral
        s.wash(s.ribbon([(x, cy + th / 2 - 2), (x + 26, cy - th / 2 + 2)], 3.2, smooth=False), PERSIMMON, 0.85, wob=0.4, soft=0.5, clip=m)
        x += 46
    for xe, sd in ((x0, -1), (x1, 1)):
        for q in range(4):
            s.ink([(xe - sd * (26 - q * 6), cy - th / 2 + 2), (xe + sd * 6, cy + J(r, 3))], 0.9, SEPIA, 0.8)
    s.lift(s.ribbon([(x0 + 40, cy - 8), (x1 - 40, cy - 8)], 4, smooth=False), 0.6, soft=1)
    contour(s, m, 1.8, SEPIA, 1.3)
    s.save("sy_straw", crop=True, anchor=((x0 + x1) / 2, cy), q=92, meta={"len": x1 - x0})


def soy_hand():
    """The boss's left hand reaching across from the upper right, back of the hand to us, a white sleeve cover (袖套)
    gathered at the wrist; the straw pinched between thumb and index finger as it is set down, the other three fingers
    curled loosely under. Anchor = the pinch (the page draws sy_straw there first, then this)."""
    s = Off(1400, 1100, 0, 0, 314)
    r = rs(314)
    P = np.array([360.0, 860.0])
    Wc = np.array([840.0, 300.0])
    u = (P - Wc) / np.linalg.norm(P - Wc)      # along the hand, toward the fingers
    v = np.array([-u[1], u[0]])
    if v[0] < 0:
        v = -v                                   # toward the thumb: the lower right of the fingers' line
    L = np.linalg.norm(P - Wc)                   # wrist to pinch, ~740 px
    at = lambda a, b: tuple(Wc + u * a * L + v * b * L)
    deg = math.degrees(math.atan2(u[1], u[0]))
    # forearm in its sleeve cover, off to the upper right
    sleeve = s.ribbon([at(-0.06, 0), at(-0.3, 0.01), at(-0.6, 0.02), at(-1.0, 0.03)], 0.40 * L, profile=lambda t: 1 + 0.1 * t)
    wrist = s.ribbon([at(-0.1, 0), at(0.08, 0.0)], 0.31 * L)
    # the hand: back of the hand and four fingers together (one outline round them all), the thumb under them
    dorsum = s.poly([at(0.0, -0.16), at(0.02, 0.16), at(0.18, 0.21), at(0.42, 0.15), at(0.47, 0.0), at(0.44, -0.16), at(0.36, -0.22), at(0.15, -0.2)])
    FING = (([at(0.40, 0.09), at(0.62, 0.08), at(0.82, 0.05), at(0.99, 0.022)], 0.108),
            ([at(0.43, -0.01), at(0.64, -0.02), at(0.80, -0.03), at(0.905, -0.02)], 0.112),
            ([at(0.41, -0.10), at(0.60, -0.11), at(0.72, -0.11), at(0.795, -0.09)], 0.10),
            ([at(0.36, -0.18), at(0.50, -0.20), at(0.60, -0.19), at(0.655, -0.16)], 0.086))
    fingers = [s.ribbon(p_, w * L, profile=lambda t: 1 - 0.14 * t) for p_, w in FING]
    idx = fingers[0]
    thumb = s.ribbon([at(0.10, 0.16), at(0.40, 0.225), at(0.70, 0.165), at(0.975, 0.06)], 0.125 * L, profile=lambda t: 1 - 0.3 * t)
    # sleeve first
    put(s, sleeve, CREAM, 0.16, lift=1.0, wob=1, edge=0.4, ew=6, flow=0.4, fs=120)
    s.wash(sleeve, RAIN, 0.4, wob=2, bleed=16, edge=0.1, grad=s.vgrad(0, 640, 0.15, 1))
    s.wash(sleeve, OCHRE, 0.18, wob=2, bleed=14, edge=0.1, grad=s.hgrad(800, 1400, 0, 1))
    s.wash(s.ribbon([at(-0.02, -0.17), at(-0.6, -0.2), at(-1.0, -0.2)], 0.1 * L), SLATE, 0.3, wob=2, bleed=8, clip=sleeve)
    for i in range(18):                           # the elastic gathered at the wrist, softer folds further up
        a = U(r, -0.07, -0.16)
        b = U(r, -0.17, 0.17)
        s.ink([at(a, b), at(a - U(r, 0.04, 0.09), b + J(r, 0.02))], 1.3, SEPIA, 0.6, skip=0.2)
    for i in range(5):
        a = U(r, -0.3, -0.9)
        s.stroke([at(a, -0.19), at(a - 0.06, J(r, 0.04)), at(a - 0.03, 0.19)], U(r, 12, 26), SLATE, 0.22, dry=0.3, edge=0.2, wob=1, clip=sleeve)
        s.ink([at(a, -0.18), at(a - 0.06, J(r, 0.04)), at(a - 0.03, 0.18)], 1.1, SEPIA, 0.4, skip=0.35)
    hem = s.ribbon([at(-0.07, -0.205), at(-0.07, 0.205)], 0.035 * L) * sleeve
    s.wash(hem, SLATE, 0.3, wob=0.6, soft=1.5)
    contour(s, sleeve, 2.6, SEPIA, 1.4)
    # the thumb first: the fingers lie over its upper edge
    base(s, thumb, SKIN, 0.6, wob=1.0, edge=0.35, ew=4)
    s.wash(thumb, BURNT, 0.34, wob=1.5, bleed=10, edge=0.15, grad=s.vgrad(500, 900, 0.2, 1))
    s.wash(thumb, ROSE, 0.2, wob=1.5, bleed=8, grad=s.rgrad(*at(0.95, 0.07), 10, 90, 1, 0))
    contour(s, thumb, 1.9, SEPIA, 1.3)
    nm = s.ellipse(*at(0.94, 0.075), 0.05 * L * 0.5, 0.05 * L * 0.36, deg)
    s.lift(nm, 0.45, soft=1, wob=0.4)
    contour(s, nm, 1.2, SEPIA, 0.9)
    hand = np.maximum(dorsum, wrist * (1 - sleeve))
    for f in fingers:
        hand = np.maximum(hand, f)
    base(s, hand, SKIN, 0.58, wob=1.2, edge=0.35, ew=5)
    s.wash(hand, BURNT, 0.22, wob=2, bleed=14, grad=s.vgrad(300, 900, 0, 1), edge=0.1)
    for f, (p_, w) in zip(fingers, FING):        # each finger turning away from us toward its tip
        s.wash(f, BURNT, 0.1, wob=1.5, bleed=10, edge=0, grad=s.rgrad(*p_[-1], 0, w * L * 2.0, 1, 0))
    s.wash(hand, ROSE, 0.14, wob=2, bleed=12, grad=s.rgrad(*at(0.45, -0.04), 20, 0.25 * L, 1, 0), edge=0)
    for b in (0.07, -0.02, -0.11, -0.19):        # tendons fanning to the knuckles
        s.ink([at(0.06, b * 0.5), at(0.25, b * 0.8), at(0.41, b)], 1.1, SEPIA, 0.35, skip=0.35)
    for (a, b) in ((0.44, 0.085), (0.47, -0.015), (0.445, -0.105), (0.39, -0.185)):     # knuckles
        s.ink(ell(*at(a, b), 0.04 * L, 0.022 * L, 2.3, 4.0, 12, rot=math.radians(deg)), 1.3, SEPIA, 0.75)
    for (a, b) in ((0.72, 0.068), (0.73, -0.026), (0.66, -0.112)):                       # middle joints
        s.ink(ell(*at(a, b), 0.03 * L, 0.018 * L, 2.6, 3.8, 10, rot=math.radians(deg)), 1.1, SEPIA, 0.55)
    for sep in ([at(0.56, 0.038), at(0.74, 0.022), at(0.86, 0.01), at(0.9, 0.012)],
                [at(0.56, -0.062), at(0.68, -0.07), at(0.77, -0.065), at(0.80, -0.05)],
                [at(0.52, -0.148), at(0.6, -0.15), at(0.65, -0.135)]):                    # where the fingers lie against each other
        s.ink(sep, 1.5, SEPIA, 1.0, taper=(0.3, 0.1))
    for (a, b, w) in ((0.955, 0.024, 0.05), (0.875, -0.018, 0.048)):                       # nails
        nm = s.ellipse(*at(a, b), w * L * 0.5, w * L * 0.36, deg)
        s.lift(nm, 0.45, soft=1, wob=0.4)
        s.wash(nm, ROSE, 0.18, wob=0.4, soft=1)
        contour(s, nm, 1.2, SEPIA, 0.9)
    # the kitchen's warm light catching the upper edges
    allm = np.maximum(hand, thumb)
    rimm = np.clip(cv2.GaussianBlur(allm, (0, 0), 2.5) - cv2.GaussianBlur(np.roll(np.roll(allm, 9, axis=0), -5, axis=1), (0, 0), 2.5), 0, 1)
    s.wash(rimm * allm, AMBER, 0.55, wob=0, soft=1, edge=0)
    contour(s, hand, 2.3, SEPIA, 1.4)
    cuff = at(-0.07, 0)
    th = at(0.975, 0.06)
    s.save("sy_hand", crop=True, anchor=tuple(P), q=92, meta={"cuff": [int(cuff[0] - P[0]), int(cuff[1] - P[1])],
                                                             "thumb": [int(th[0] - P[0]), int(th[1] - P[1])]})


ALL = {"stall_bg": stall_bg, "stall_fg": stall_fg, "stall_hands": stall_hands, "soy_bg": soy_bg, "soy_cup": soy_cup, "soy_straw": soy_straw,
       "soy_hand": soy_hand}
