"""Drawing habits shared by every painting in the film: loose urban-sketch pen work over transparent washes.
The pen overshoots corners and never quite closes a shape; the wash never quite fills the pen line."""
import math

import cv2
import numpy as np

from wc import *


def Q(s, pts):
    """Hard-cornered polygon mask."""
    return s.poly(pts, smooth=False)


def R(s, x0, y0, x1, y1):
    return Q(s, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def ell(cx, cy, rx, ry, a0=0.0, a1=2 * math.pi, n=48, rot=0.0):
    """Points on an ellipse arc, for the pen."""
    a = np.linspace(a0, a1, n, endpoint=abs(a1 - a0) < 2 * math.pi - 1e-6)
    x, y = np.cos(a) * rx, np.sin(a) * ry
    c, sn = math.cos(rot), math.sin(rot)
    return np.stack([cx + x * c - y * sn, cy + x * sn + y * c], 1)


def base(s, m, color, k=0.8, **kw):
    """An opaque object: clear what is behind it, lay its first wash, mark it as body."""
    s.lift(m, 1.0, soft=0.8, wob=1.0)
    return s.wash(m, color, k, body=1, **kw)


def seg(s, a, b, w=2.2, color=INK, k=1.6, bow=0.006, over=0.0, r=None, **kw):
    """A ruled-by-hand pen line: a slight bow, optional overshoot past both ends."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.hypot(*d)) + 1e-6
    u = d / L
    n = np.array([-u[1], u[0]])
    if over:
        a, b = a - u * over, b + u * over
    j = (r.random() - 0.5) * 2 if r is not None else 0.6
    mid = (a + b) / 2 + n * L * bow * j
    kw.setdefault("taper", (0.06, 0.10))
    s.ink([a, mid, b], w, color, k, **kw)


def path(s, pts, w=2.2, color=INK, k=1.6, closed=False, over=4.0, r=None, **kw):
    """A polygonal outline drawn edge by edge, each edge its own stroke (corners overshoot, lines double up)."""
    P = list(pts) + ([pts[0]] if closed else [])
    for a, b in zip(P, P[1:]):
        seg(s, a, b, w, color, k, over=over, r=r, **kw)


def oval(s, cx, cy, rx, ry, w=2.2, color=INK, k=1.6, rot=0.0, a0=0.0, a1=2 * math.pi, twice=False, **kw):
    """A pen ellipse; `twice` goes round again slightly off, the way a quick hand does."""
    full = abs(a1 - a0) >= 2 * math.pi - 1e-6
    if full:      # leave the loop a little open and let the ends cross
        s.ink(ell(cx, cy, rx, ry, a0 + 0.3, a0 + 0.3 + 2 * math.pi + 0.25, 60, rot), w, color, k, **kw)
    else:
        s.ink(ell(cx, cy, rx, ry, a0, a1, 40, rot), w, color, k, **kw)
    if twice:
        s.ink(ell(cx + 1.5, cy - 1.2, rx * 1.012, ry * 0.985, a0 + 2.0, a0 + 2.0 + 3.6, 40, rot), w * 0.7, color, k * 0.7, **kw)


def wood(s, m, color=SIENNA, dark=BURNT, k=0.75, ang=0.0, n=26, seed=1, lines=True, lw=1.3, streak=1.0, clear=True, wob=2):
    """Wood: a light wash, long dry streaks along the grain, a few pen grain lines with gaps."""
    r = rs(seed)
    if clear:
        base(s, m, color, k, wob=wob, flow=0.45, fs=260, gran=0.4, edge=0.35)
    else:
        s.wash(m, color, k, wob=wob, flow=0.45, fs=260, gran=0.4, edge=0.35)
    roi = s._roi(m, 0)
    if roi is None:
        return
    y0, y1, x0, x1 = roi[0].start, roi[0].stop, roi[1].start, roi[1].stop
    cx, cy, Rr = (x0 + x1) / 2, (y0 + y1) / 2, math.hypot(x1 - x0, y1 - y0) / 2
    ca, sa = math.cos(ang), math.sin(ang)
    mm = cv2.GaussianBlur(m, (0, 0), 1.2)
    for i in range(n):
        o = (r.random() * 2 - 1) * Rr
        l0, l1 = -Rr, Rr
        wv = (r.random() - 0.5) * Rr * 0.04
        pts = [(cx + ca * (l0 + (l1 - l0) * q) - sa * (o + wv * math.sin(q * 5 + i)), cy + sa * (l0 + (l1 - l0) * q) + ca * (o + wv * math.sin(q * 5 + i))) for q in np.linspace(0, 1, 7)]
        if i < n * 0.55 * streak:
            s.stroke(pts, 10 + 46 * r.random(), dark, 0.10 + 0.20 * r.random(), dry=0.35 + 0.3 * r.random(), edge=0.2, wob=1.5, clip=mm, taper=(0.05, 0.05), gran=0.5)
        elif lines:
            s.ink(pts, lw, dark, 0.55 + 0.5 * r.random(), tremor=1.1, skip=0.45, clip=mm, taper=(0.03, 0.03))


def shade(s, m, color=UMBER, k=0.35, bleed=10, **kw):
    """A soft shadow glaze (wet edge)."""
    kw.setdefault("edge", 0.15)
    kw.setdefault("gran", 0.35)
    return s.wash(m, color, k, wob=2.5, bleed=bleed, **kw)


def cast(s, m, dx, dy, color=UMBER, k=0.4, blur=14, clip=None):
    """A cast shadow: the object's mask shifted and softened."""
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    sh = cv2.warpAffine(m, M, (s.w, s.h))
    sh = cv2.GaussianBlur(sh, (0, 0), blur)
    if clip is not None:
        sh = sh * clip
    return s.wash(sh, color, k, wob=0, soft=0, edge=0, gran=0.4, flow=0.2)


def glow(s, cx, cy, r, amt=0.6, sy=1.0):
    """Light: lift the pigment in a soft disc (the paper is the lamp)."""
    return s.lift(s.rgrad(cx, cy, r * 0.15, r, 1, 0, sy), amt, soft=0, wob=0)


def text_mask(s, txt, x, y, size, font, anchor="mm", spacing=0, rot=0.0):
    """Rasterise text with Pillow into a mask (for signs)."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("L", (s.w, s.h), 0)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(font, int(size))
    if spacing:
        cx = x - (len(txt) - 1) * (size + spacing) / 2
        for ch in txt:
            d.text((cx, y), ch, 255, f, anchor="mm")
            cx += size + spacing
    else:
        d.text((x, y), txt, 255, f, anchor=anchor)
    m = np.asarray(im, np.float32) / 255
    if rot:
        M = cv2.getRotationMatrix2D((x, y), rot, 1)
        m = cv2.warpAffine(m, M, (s.w, s.h))
    return m


def contour(s, m, w=2.2, color=INK, k=1.5, level=0.5, step=5.0, **kw):
    """Pen the outline of a mask (so a union of shapes gets one line round it, like a hand round its fingers)."""
    cs, _ = cv2.findContours((m > level).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cs:
        P = c[:, 0, :].astype(np.float64)
        if len(P) < 12:
            continue
        P = P[::max(1, int(step))]
        P = np.vstack([P, P[:2]])
        # run a small moving average so the pixel staircase does not show
        Pm = (np.roll(P, 1, 0) + P + np.roll(P, -1, 0)) / 3
        Pm[0], Pm[-1] = P[0], P[-1]
        s.ink(Pm, w, color, k, smooth=False, taper=(0.02, 0.04), **kw)


def finger(s, pts, w, tip=0.72):
    """A finger (or a thumb) as a tapering capsule along a path; root to tip."""
    return s.ribbon(pts, w, profile=lambda t: 1.0 - (1 - tip) * t)
