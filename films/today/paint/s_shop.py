"""A little old shop of osmanthus things (桂花釀, 桂花糕, tea) on the lane, at blue hour. Sheet 3072x1728.
Seen from across the lane at eye level (y=1000), the front a plane at about 300 px/m, the lane's floor line y=1440.
The camera trucks along the shop front, so the sheet is a long frieze: a neighbour's house on the left, the shop in the
middle (a grey tile eave, a hand-lettered sign, an indigo cloth awning, a warm window full of jars, a bench, a door
with a noren), a garden wall with an osmanthus tree over it on the right.

  os_bg    opaque. meta: window [x0,y0,x1,y1] the lit window's glass (warm glow); sign [x,y] the middle of the "桂花"
           board; door [x0,y0,x1,y1] the lit doorway behind the noren; lamps [[x,y],...] the warm lamps (over the sign,
           over the door, the window upstairs on the left); flowers [[x,y],...] osmanthus blossom clusters on the tree over
           the garden wall (falling-petal origins); bench [x,y] the middle of the bench seat; ground_y the floor line.
  os_tree  sprite, an osmanthus shrub in a big glazed pot, near, at the right edge. Anchor = the middle of the pot's
           foot on the ground; place at (2740, 1728) for the painted composition (parallax ~1.3x). meta: flowers
           [[dx,dy],...] blossom clusters relative to the anchor; lit [dx,dy] the side the window lights.
  os_fg    sprite, a near wall corner on the left edge with a bracket lamp. Anchor = its foot (x, 1728); place at its own
           anchor (130, 1728) for the composition (parallax ~1.6x). meta: lamp [dx,dy] the lamp's bulb relative to anchor.
"""
import math

import cv2
import numpy as np

from kit import *
from s_stall import (KAI, SERIF, Off, Sub, J, U, box, cable, contour, drags, hl, letters, line, mottle, paint_letters, pl, put,
                     streaks, tiles)

W, H = 3072, 1728
HZ = 1000
G = 1440                                  # floor line of the shop front
SX0, SX1 = 760, 2330                      # the shop's walls
EAVE = 528
WIN = (930, 858, 1910, 1238)              # the window glass
DOOR = (1972, 772, 2228, G)
SIGN = (1420, 612)
LAMPS = [(1260, 548), (1580, 548), (2100, 742), (430, 520)]
FLOWERS = []
BENCH = (1290, 1316)


def leafmass(s, r, cx, cy, rx, ry, n, size=(9, 16), cols=(LEAF, PRUSSIAN, MOSS), k=(0.5, 0.9), light=None, seed=0, clip=None):
    """A mass of small leaves: first a wet underpainting, then leaves laid in three layers from dark to light, each layer
    one wash of many little pointed ellipses (osmanthus leaves are small, leathery, dark)."""
    g = rs(seed)
    m0 = s.blob(cx, cy, rx, ry, irr=0.25, n=15, seed=seed + 1)
    s.lift(m0 * 0.75 if clip is None else m0 * clip * 0.75, 0.75, soft=2, wob=2)
    s.wash(m0, cols[1], 0.3, wob=4, bleed=10, edge=0.2, flow=0.6, fs=40, gran=0.4, body=0.95, clip=clip)
    s.wash(m0, cols[0], 0.25, wob=4, bleed=14, edge=0.1, flow=0.8, fs=30, gran=0.4, clip=clip)
    ox, oy = getattr(s, "ox", 0), getattr(s, "oy", 0)
    for layer, (c, kk, frac) in enumerate(((cols[1], k[1], 0.9), (cols[0], k[0], 0.75), (cols[2], k[0] * 0.8, 0.5))):
        m = np.zeros((s.h * 2, s.w * 2), np.uint8)
        for _ in range(int(n * frac)):
            a = g.random() * 2 * math.pi
            d = g.random() ** 0.6
            x = cx + math.cos(a) * rx * d * 0.95
            y = cy + math.sin(a) * ry * d * 0.95 - layer * ry * 0.06
            L = U(g, *size)
            ang = g.random() * 180
            cv2.ellipse(m, (int((x - ox) * 2 * 8), int((y - oy) * 2 * 8)), (int(L * 2 * 8), int(L * 0.42 * 2 * 8)), ang, 0, 360, 255, -1, cv2.LINE_AA, 3)
        mm = cv2.resize(m, (s.w, s.h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        if clip is not None:
            mm *= clip
        s.wash(mm, c, kk, wob=0.6, soft=0.5, edge=0.6, ew=1.5, gran=0.4, flow=0.4, fs=30, body=0.95)
        if layer == 2:                      # the top leaves catch what light there is
            s.lift(mm * s.noise(30, 2), 0.35, soft=0.5, wob=0)
    if light is not None:
        lx, ly, lr, lc, lk = light
        s.wash(m0, lc, lk, wob=2, soft=4, edge=0, grad=s.rgrad(lx, ly, lr * 0.2, lr, 1, 0))
    return m0


def blossoms(s, r, pts, n=(6, 14), spread=10, k=1.1, seed=0):
    """Clusters of tiny four-petalled orange flowers, tucked against the leaves."""
    g = rs(seed)
    ox, oy = getattr(s, "ox", 0), getattr(s, "oy", 0)
    m = np.zeros((s.h * 2, s.w * 2), np.uint8)
    for (x, y) in pts:
        for _ in range(int(g.integers(*n))):
            fx, fy = x + g.normal() * spread, y + g.normal() * spread * 0.7
            rr = U(g, 1.6, 3.4)
            for q in range(4):
                a = q * math.pi / 2 + g.random()
                cv2.circle(m, (int((fx - ox + math.cos(a) * rr * 0.7) * 16), int((fy - oy + math.sin(a) * rr * 0.7) * 16)), int(rr * 0.75 * 16), 255, -1, cv2.LINE_AA, 3)
    mm = cv2.resize(m, (s.w, s.h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    s.lift(mm, 0.85, soft=0.4, wob=0)
    s.wash(mm, OSMAN, k, wob=0.3, soft=0.4, edge=0.6, ew=1.0, gran=0.2, flow=0, body=0.9)
    s.wash(mm, AMBER, 0.3, wob=0.3, soft=0.6, edge=0.3, flow=0.6, fs=8)
    return mm


# ================================================================================================ the background
def os_bg():
    S = Sheet(W, H, 401)
    r = rs(401)
    sky(S, r)
    far(S, r)
    left_house(S, r)
    garden(S, r)
    shop(S, r)
    ground(S, r)
    dusk(S, r)
    meta = {"window": list(WIN), "sign": list(SIGN), "door": list(DOOR), "lamps": [list(p) for p in LAMPS],
            "flowers": [[int(x), int(y)] for x, y in FLOWERS], "bench": list(BENCH), "ground_y": G}
    S.save("os_bg", opaque=True, q=90, meta=meta)


def sky(S, r):
    s = Sub(S, 0, 0, W, 820)
    F = s.full()
    s.wash(F, DUSK, 0.6, wob=0, soft=0, edge=0, flow=0.4, fs=500, gran=0.15)
    s.wash(F, INDIGO, 0.9, wob=0, soft=0, edge=0, grad=s.vgrad(-120, 420, 1, 0), flow=0.5, fs=400, gran=0.2)
    s.wash(F, PAYNE, 0.4, wob=0, soft=0, edge=0, grad=s.vgrad(-60, 240, 1, 0), flow=0.6, fs=300, gran=0.2)
    s.wash(F, ROSE, 0.3, wob=0, soft=0, edge=0, grad=s.vgrad(250, 520, 0, 1) * s.vgrad(560, 820, 1, 0.4), flow=0.7, fs=300, gran=0.15)
    s.wash(F, OCHRE, 0.18, wob=0, soft=0, edge=0, grad=s.vgrad(400, 700, 0, 1), flow=0.7, fs=250)
    # long thin clouds, wet in wet, catching the last light underneath
    for cx, cy, rx, ry, k in ((600, 210, 700, 22, 0.35), (1900, 150, 900, 26, 0.32), (2700, 290, 500, 18, 0.28), (1200, 330, 520, 14, 0.24)):
        s.wash(s.blob(cx, cy, rx, ry, irr=0.4, n=17), VIOLET, k, wob=8, bleed=20, edge=0.3, ew=8, flow=0.5, fs=120, gran=0.25)
        s.wash(s.blob(cx + J(r, 80), cy + ry * 0.7, rx * 0.7, ry * 0.4, irr=0.4), ROSE, k * 0.6, wob=6, bleed=10, edge=0.2)
    for cx, cy, a in ((1300, 260, 0.2), (2300, 340, 0.18)):
        s.lift(s.blob(cx, cy, 380, 28, irr=0.4), a, bleed=20, wob=6)
    # the first star
    st = s.ellipse(2380, 92, 3.5, 3.5)
    s.lift(st, 1.0, soft=0.6, wob=0)
    s.lift(s.rgrad(2380, 92, 2, 16, 1, 0), 0.25, soft=0, wob=0)


def far(S, r):
    """Roofs and blocks farther off: flat dusk shapes with a few lit windows, tanks, aerials."""
    s = Sub(S, 0, 180, W, 820)
    x = -40.0
    while x < W:
        w_ = U(r, 140, 380)
        top = U(r, 330, 470) if r.random() < 0.7 else U(r, 250, 330)
        m = s.rect(x, top, x + w_, 820)
        s.wash(m, PAYNE, U(r, 0.45, 0.65), wob=0.8, soft=1.0, edge=0.4, ew=5, flow=0.5, fs=90, gran=0.3, body=1)
        s.wash(m, INDIGO, 0.25, wob=3, bleed=8, edge=0.1, grad=s.vgrad(top, top + 200, 1, 0.3))
        line(s, r, (x - 2, top), (x + w_ + 2, top + J(r, 1)), 1.4, INKBLUE, 1.1)
        for _ in range(int(r.integers(1, 6))):
            wx, wy = U(r, x + 10, x + w_ - 30), U(r, top + 20, 760)
            wm = s.rect(wx, wy, wx + U(r, 12, 20), wy + U(r, 16, 24))
            if r.random() < 0.55:
                s.lift(wm, 0.9, soft=0.6, wob=0.4)
                s.wash(wm, OCHRE, U(r, 0.35, 0.6), wob=0.4, soft=0.6, edge=0.5, ew=2)
            else:
                s.wash(wm, INKBLUE, 0.5, wob=0.4, soft=0.6)
        if r.random() < 0.35:            # a water tank on legs
            tx = x + U(r, 20, w_ - 70)
            tm = s.rect(tx, top - 50, tx + 50, top - 18)
            s.wash(tm, PAYNE, 0.7, wob=0.5, soft=0.8, edge=0.4, body=1)
            for lx in (tx + 6, tx + 44):
                line(s, r, (lx, top - 18), (lx, top), 1.2, INKBLUE, 1.0, over=0)
            box(s, r, tx, top - 50, tx + 50, top - 18, 1.2, INKBLUE, 1.0, over=1)
        if r.random() < 0.3:             # an aerial
            ax = x + U(r, 10, w_ - 10)
            s.ink([(ax, top), (ax + J(r, 2), top - U(r, 40, 80))], 1.3, INKBLUE, 1.0)
            for i in range(3):
                line(s, r, (ax - 12 + i * 2, top - 40 - i * 9), (ax + 12 - i * 2, top - 40 - i * 9), 1.0, INKBLUE, 0.9, over=0)
        x += w_ * U(r, 0.7, 1.05)
    s.wash(s.full(), None, 0.25, wob=0, soft=0, edge=0, lift=True, grad=s.vgrad(600, 820, 0, 1))      # the evening haze low down


def left_house(S, r):
    """The neighbour on the left: three storeys of dusky old plaster, one lit window upstairs with a cage and a plant,
    a shuttered door at the street."""
    s = Sub(S, 0, 100, SX0 + 20, G + 30)
    fm = s.rect(-10, 130, SX0, G + 10)
    s.wash(fm, DUSK, 0.6, wob=1.5, soft=0.9, edge=0.4, ew=8, flow=0.5, fs=220, gran=0.45, body=1)
    s.wash(fm, VIOLET, 0.25, wob=3, bleed=12, edge=0.1, flow=0.6, fs=180, clip=fm)
    s.wash(fm, PAYNE, 0.35, wob=3, bleed=14, edge=0.1, grad=s.vgrad(130, 900, 0.2, 1), clip=fm)
    mottle(s, r, 20, 160, SX0 - 20, G - 20, [PAYNE, ROSE, UMBER, SLATE], 7, 0.13, clip=fm)
    drags(s, r, 20, 160, SX0 - 20, G - 20, PAYNE, 8, clip=fm)
    streaks(s, r, 20, SX0 - 20, 160, G, 16, PAYNE, 0.22, (6, 16), clip=fm)
    s.wash(s.rect(-10, 130, SX0 + 6, 152), SLATE, 0.7, wob=0.8, soft=0.8, edge=0.4, body=1)
    line(s, r, (-5, 130), (SX0 + 8, 129), 2.2, INKBLUE, 1.5, dbl=0.5)
    line(s, r, (-5, 152), (SX0 + 6, 152), 1.4, INKBLUE, 1.2)
    # floor bands
    for y in (520, 860):
        s.lift(s.rect(-10, y, SX0, y + 14), 0.2, soft=0.8, wob=0.8)
        line(s, r, (-5, y), (SX0 + 5, y), 1.2, INKBLUE, 1.0, skip=0.1)
        line(s, r, (-5, y + 14), (SX0 + 5, y + 14), 1.8, INKBLUE, 1.3, dbl=0.4)
        s.wash(s.rect(-10, y + 14, SX0, y + 34), PAYNE, 0.3, wob=0, soft=0, edge=0, grad=s.vgrad(y + 14, y + 34, 1, 0))
    # the lit window upstairs (third floor) and a dark one below it
    wx0, wy0, wx1, wy1 = 300, 300, 560, 480
    gm = s.rect(wx0, wy0, wx1, wy1)
    s.lift(gm, 1.0, soft=0.6, wob=0)
    s.wash(gm, OCHRE, 0.45, wob=0.6, soft=0.8, edge=0.4, ew=3, flow=0.4, fs=40)
    s.wash(s.rect(wx0, wy0, wx0 + 90, wy1), SIENNA, 0.45, wob=0.6, soft=0.8, edge=0.5, ew=2)             # a curtain half drawn
    s.wash(s.rect(wx1 - 70, wy0, wx1, wy1), SIENNA, 0.35, wob=0.6, soft=0.8, edge=0.5, ew=2)
    s.wash(s.rect(wx0 + 140, wy0 + 120, wx0 + 170, wy1), UMBER, 0.6, wob=0.4, soft=0.8)                   # a lamp's stem
    s.wash(s.ellipse(wx0 + 155, wy0 + 112, 34, 16), OCHRE, 0.5, wob=0.4, soft=0.8)
    halo = s.rgrad(430, 390, 60, 300, 1, 0) * (1 - gm)
    s.wash(halo, None, 0.2, wob=0, soft=0, edge=0, lift=True)
    s.wash(halo, AMBER, 0.15, wob=0, soft=0, edge=0)
    line(s, r, (wx0 + 130, wy0), (wx0 + 130, wy1), 1.6, SEPIA, 1.2)
    box(s, r, wx0, wy0, wx1, wy1, 2.0, SEPIA, 1.4, over=4, dbl=0.4)
    # its cage, with pots
    x = wx0 - 18
    while x < wx1 + 20:
        line(s, r, (x, wy0 - 16), (x + J(r, 1), wy1 + 24), 1.4, INKBLUE, 1.2, over=1)
        x += 21
    box(s, r, wx0 - 22, wy0 - 18, wx1 + 22, wy1 + 26, 2.2, INKBLUE, 1.4, over=4, dbl=0.4)
    s.wash(s.rect(wx0 - 22, wy1 + 26, wx1 + 22, wy1 + 50), PAYNE, 0.6, wob=0.6, soft=0.8, body=1)
    for px in (wx0 + 10, wx0 + 110, wx1 - 40):
        for _ in range(10):
            a = U(r, -2.7, -0.4)
            Lf = U(r, 22, 60)
            tip = (px + math.cos(a) * Lf, wy1 + 20 + math.sin(a) * Lf)
            s.stroke([(px, wy1 + 22), tip], U(r, 6, 11), (LEAF, MOSS, PRUSSIAN)[int(r.integers(0, 3))], U(r, 0.6, 0.9), taper=(0.1, 0.7), wob=0.8, edge=0.5, body=0.9)
    # second floor: a dark window with its blind down, an AC box
    window_dark(s, r, 120, 620, 400, 800)
    s.wash(s.rect(470, 660, 610, 750), SLATE, 0.6, wob=0.6, soft=0.8, edge=0.4, ew=2, body=1)
    box(s, r, 470, 660, 610, 750, 1.8, INKBLUE, 1.3, over=2)
    for q in np.linspace(0.2, 0.8, 6):
        line(s, r, (480, 660 + 90 * q), (550, 660 + 90 * q), 0.9, INKBLUE, 0.8, over=0)
    s.stroke([(590, 752), (592, 900)], 8, PAYNE, 0.3, dry=0.25, taper=(0, 0.7))
    # the street door: old wood, shut, a red spring couplet half peeled
    dx0, dy0, dx1, dy1 = 180, 980, 480, G - 4
    dm = s.rect(dx0, dy0, dx1, dy1)
    put(s, dm, UMBER, 0.75, lift=0.8, wob=1, edge=0.45, ew=5, flow=0.5, fs=60)
    s.wash(dm, INDIGO, 0.25, wob=2, bleed=10, edge=0.1)
    for xx in np.arange(dx0 + 30, dx1, 30):
        line(s, r, (xx, dy0 + 4), (xx + J(r, 1), dy1 - 4), 1.1, SEPIA, 0.8, over=0, skip=0.2)
    box(s, r, dx0, dy0, dx1, dy1, 2.2, SEPIA, 1.5, over=5, dbl=0.4)
    for cx_ in (dx0 - 40, dx1 + 40):
        cm = s.rect(cx_ - 22, dy0 + 20, cx_ + 22, dy0 + 300)
        put(s, cm, PERSIMMON, 0.55, lift=0.7, wob=0.8, edge=0.5, ew=3, dry=0.25)
        s.wash(cm, PAYNE, 0.35, wob=1, soft=2)
        s.lift(s.rect(cx_ - 22, dy0 + 200, cx_ + 22, dy0 + 300) * s.noise(14, 2), 0.6, soft=1)       # peeled
    lt = letters(s, "平安", dx0 - 40, dy0 + 100, 30, KAI, 400, vertical=True)
    paint_letters(s, lt, INK, 0.8, wob=0.3)
    lt = letters(s, "如意", dx1 + 40, dy0 + 100, 30, KAI, 400, vertical=True)
    paint_letters(s, lt, INK, 0.8, wob=0.3)
    # a bicycle leaning on the wall: dark, simple
    by = G - 6
    for wx in (560, 720):
        oval(s, wx, by - 58, 58, 58, 2.2, INKBLUE, 1.3)
        oval(s, wx, by - 58, 6, 6, 1.4, INKBLUE, 1.2)
    pl(s, r, [(560, by - 58), (630, by - 150), (700, by - 150), (720, by - 58), (650, by - 58), (560, by - 58)], 2.4, INKBLUE, 1.3, over=1)
    pl(s, r, [(630, by - 150), (650, by - 58)], 2.2, INKBLUE, 1.3, over=1)
    pl(s, r, [(700, by - 150), (708, by - 180), (690, by - 186)], 2.2, INKBLUE, 1.3, over=1)
    s.wash(s.ellipse(625, by - 162, 22, 7), INK, 0.8, wob=0.4, soft=0.6, body=1)
    bk = s.rect(726, by - 196, 778, by - 158)
    s.wash(bk, SLATE, 0.5, wob=0.5, soft=0.8, body=0.8)
    s.hatch(bk, 0.0, 6, 0.9, INKBLUE, 0.8, skip=0.1)
    s.hatch(bk, 1.57, 6, 0.9, INKBLUE, 0.8, skip=0.1)
    s.wash(s.rect(-10, G - 50, SX0, G + 10), PAYNE, 0.3, wob=3, bleed=10, edge=0.1, clip=fm)
    line(s, r, (SX0, 130), (SX0 + J(r, 2), G), 2.4, SEPIA, 1.5, dbl=0.5)


def window_dark(s, r, x0, y0, x1, y1):
    m = s.rect(x0, y0, x1, y1)
    put(s, m, INKBLUE, 0.7, lift=0.8, wob=0.8, edge=0.5, ew=4, flow=0.5, fs=40)
    s.lift(Q(s, [(x0 + 6, y0 + 6), (x0 + (x1 - x0) * 0.5, y0 + 6), (x0 + 6, y0 + (y1 - y0) * 0.6)]), 0.25, soft=3)       # the sky in it
    s.wash(s.rect(x0 + 4, y0 + 4, x1 - 4, y0 + (y1 - y0) * 0.45), SLATE, 0.4, wob=0.6, soft=0.8, edge=0.4)
    for q in np.linspace(0.1, 0.42, 6):
        line(s, r, (x0 + 6, y0 + (y1 - y0) * q), (x1 - 6, y0 + (y1 - y0) * q), 0.9, INKBLUE, 0.7, over=0)
    line(s, r, ((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1), 1.6, SEPIA, 1.2)
    box(s, r, x0, y0, x1, y1, 2.0, SEPIA, 1.4, over=4, dbl=0.4)
    s.wash(s.rect(x0 - 8, y1, x1 + 8, y1 + 9), SLATE, 0.6, wob=0.5, soft=0.8, body=1)
    line(s, r, (x0 - 10, y1 + 9), (x1 + 10, y1 + 9), 1.8, SEPIA, 1.3)


def garden(S, r):
    """On the right: a two-storey house set back, a garden wall of old brick with a tile cap, a gate, and the osmanthus
    tree leaning over the wall, in flower."""
    s = Sub(S, SX1 - 20, 150, W, G + 30)
    # the house behind
    hm = s.rect(SX1 + 120, 330, W + 10, 900)
    s.wash(hm, PAYNE, 0.55, wob=1.2, soft=0.9, edge=0.4, ew=6, flow=0.5, fs=150, gran=0.4, body=1)
    s.wash(hm, DUSK, 0.3, wob=3, bleed=10, edge=0.1, flow=0.6, fs=120)
    rf = Q(s, [(SX1 + 90, 340), (W + 20, 340), (W + 20, 290), (SX1 + 160, 290)])
    s.wash(rf, INKBLUE, 0.7, wob=0.8, soft=0.8, edge=0.4, body=1)
    line(s, r, (SX1 + 88, 340), (W + 10, 340), 2.0, INKBLUE, 1.4, dbl=0.4)
    lw = s.rect(2850, 430, 2990, 560)
    s.lift(lw, 1.0, soft=0.6, wob=0)
    s.wash(lw, OCHRE, 0.5, wob=0.5, soft=0.8, edge=0.5, ew=2)
    s.wash(s.rect(2850, 430, 2990, 480), SIENNA, 0.4, wob=0.5, soft=0.8)
    line(s, r, (2920, 430), (2920, 560), 1.4, SEPIA, 1.2)
    box(s, r, 2850, 430, 2990, 560, 1.8, SEPIA, 1.3, over=3)
    s.wash(s.rgrad(2920, 495, 40, 200, 1, 0) * (1 - lw), AMBER, 0.12, wob=0, soft=0, edge=0)
    LAMPS.append((2920, 495))
    # the tree: a trunk going up behind the wall, then the crown in masses
    for pts, w_ in (([(2700, 900), (2690, 760), (2650, 620), (2600, 520)], 34), ([(2690, 760), (2780, 640), (2860, 560)], 20), ([(2650, 620), (2540, 560), (2470, 520)], 16)):
        s.stroke(pts, w_, SEPIA, 0.9, taper=(0.0, 0.5), wob=1, edge=0.5, body=1)
        s.ink(pts, 1.4, INK, 1.0, skip=0.2)
    crown = s.zeros()
    for (cx, cy, rx, ry, sd) in ((2560, 520, 200, 120, 11), (2760, 470, 220, 140, 12), (2950, 560, 190, 150, 13), (2440, 640, 150, 100, 14),
                                 (2660, 650, 200, 110, 15), (2880, 720, 170, 90, 16)):
        m0 = leafmass(s, r, cx, cy, rx, ry, int(rx * ry / 55), (8, 15), (LEAF, INDIGO, MOSS), (0.32, 0.5),
                      light=(2300, 900, 500, OCHRE, 0.25), seed=sd)
        crown = np.maximum(crown, m0)
    s.wash(crown, INDIGO, 0.12, wob=2, soft=4, edge=0, grad=s.vgrad(380, 800, 1, 0.2))
    s.wash(crown, OCHRE, 0.18, wob=2, soft=6, edge=0, grad=s.hgrad(2380, 2700, 1, 0))
    g = rs(17)
    pts = []
    for _ in range(46):
        a = g.random() * 2 * math.pi
        cx, cy = 2700 + math.cos(a) * 330 * g.random() ** 0.5, 590 + math.sin(a) * 200 * g.random() ** 0.5
        pts.append((cx, cy))
    blossoms(s, r, pts, (5, 12), 9, 1.0, seed=18)
    FLOWERS.extend(sorted(pts, key=lambda p: p[1])[:30])
    contour(s, crown, 1.4, INKBLUE, 0.9, skip=0.35)
    # the garden wall: brick, a tile cap, a gate of iron
    wm = s.rect(SX1, 900, W + 10, G + 6)
    s.wash(wm, BURNT, 0.5, wob=1.2, soft=0.9, edge=0.4, ew=6, flow=0.5, fs=150, gran=0.5, body=1)
    s.wash(wm, PAYNE, 0.5, wob=3, bleed=10, edge=0.1, flow=0.6, fs=120, clip=wm)
    s.wash(wm, VIOLET, 0.15, wob=3, bleed=10, edge=0.1, clip=wm)
    for _ in range(14):                         # brickwork said in patches
        cx, cy = U(r, SX1 + 40, W - 40), U(r, 940, G - 30)
        j0 = round((cy - 900) / 16)
        for j in range(int(r.integers(2, 5))):
            y = 900 + (j0 + j) * 16
            seg(s, (cx - U(r, 30, 70), y), (cx + U(r, 30, 70), y + J(r, 0.6)), 0.9, SEPIA, 0.6, r=r, skip=0.25)
            off = 18 if (j0 + j) % 2 else 0
            for i in range(-2, 3):
                if r.random() < 0.7:
                    seg(s, (cx + i * 36 + off, y + 1), (cx + i * 36 + off, y + 15), 0.8, SEPIA, 0.5, r=r)
    cap = s.rect(SX1, 876, W + 10, 904)
    put(s, cap, SLATE, 0.7, lift=0.85, wob=0.8, edge=0.45, ew=3)
    for xx in np.arange(SX1 + 6, W, 28):
        s.ink(ell(xx + 14, 878, 14, 8, math.pi, 2 * math.pi, 12), 1.1, INKBLUE, 1.0)
    line(s, r, (SX1, 876), (W + 10, 876), 2.0, INKBLUE, 1.4, dbl=0.4)
    line(s, r, (SX1, 904), (W + 10, 904), 1.8, INKBLUE, 1.3)
    s.wash(s.rect(SX1, 904, W + 10, 940), PAYNE, 0.45, wob=0, soft=0, edge=0, grad=s.vgrad(904, 940, 1, 0))
    for _ in range(18):                         # flowers fallen on the cap
        x = U(r, SX1 + 300, W - 20)
        s.wash(s.ellipse(x, 874 + J(r, 2), 3, 2), OSMAN, 1.0, wob=0.2, soft=0.4, body=0.9)
    gx0, gx1 = 2520, 2760
    gm = s.rect(gx0, 960, gx1, G - 2)
    put(s, gm, INKBLUE, 0.65, lift=0.9, wob=0.8, edge=0.45, ew=4)
    s.lift(s.rect(gx0 + 12, 980, gx1 - 12, G - 20), 0.15, soft=3, wob=2)
    xx = gx0 + 14
    while xx < gx1 - 8:
        seg(s, (xx, 966), (xx + J(r, 0.8), G - 8), 1.3, INK, 1.2, r=r)
        xx += 18
    for yy in (1000, 1240, G - 30):
        line(s, r, (gx0, yy), (gx1, yy + J(r, 1)), 2.2, INK, 1.3)
    box(s, r, gx0, 960, gx1, G - 2, 2.4, INK, 1.5, over=4, dbl=0.4)
    for px in (gx0 - 40, gx1):
        pm = s.rect(px, 930, px + 40, G + 4)
        put(s, pm, SLATE, 0.6, lift=0.8, wob=0.8, edge=0.45, ew=3)
        s.wash(pm, PAYNE, 0.4, wob=1, soft=2, grad=s.hgrad(px, px + 40, 0.2, 1))
        box(s, r, px, 930, px + 40, G + 4, 1.8, INKBLUE, 1.4, over=3)
        put(s, s.rect(px - 6, 916, px + 46, 934), SLATE, 0.7, lift=0.6, wob=0.6, edge=0.4)
    nm = s.rect(gx1 + 4, 1020, gx1 + 36, 1060)
    put(s, nm, CREAM, 0.2, lift=0.9, wob=0.4)
    s.wash(letters(s, "7", gx1 + 20, 1040, 26, SERIF, 800), INK, 1.2, wob=0.3, soft=0.4)
    box(s, r, gx1 + 4, 1020, gx1 + 36, 1060, 1.2, SEPIA, 1.1, over=1)
    streaks(s, r, SX1 + 10, W - 10, 940, G, 14, PAYNE, 0.2, (6, 14), clip=wm)
    s.wash(s.rect(SX1, G - 60, W + 10, G + 6), PAYNE, 0.3, wob=3, bleed=10, edge=0.1, clip=wm)


def shop(S, r):
    s = Sub(S, SX0 - 40, 380, SX1 + 40, G + 40)
    # ---- the facade: dark timber boards, the eave's shadow
    fm = s.rect(SX0, EAVE, SX1, G + 6)
    s.wash(fm, UMBER, 0.7, wob=1.2, soft=0.9, edge=0.4, ew=8, flow=0.5, fs=180, gran=0.45, body=1)
    s.wash(fm, INDIGO, 0.3, wob=3, bleed=12, edge=0.1, flow=0.6, fs=150, clip=fm)
    for _ in range(14):
        x = U(r, SX0 + 10, SX1 - 10)
        s.stroke([(x, EAVE + U(r, 20, 200)), (x + J(r, 3), G - U(r, 0, 200))], U(r, 10, 28), SEPIA, U(r, 0.08, 0.16), dry=0.5, edge=0.2, wob=1.5,
                 taper=(0.1, 0.3), clip=fm)
    xx = SX0 + 40
    while xx < SX1:                             # the boards
        seg(s, (xx, EAVE + 40), (xx + J(r, 1), G - 8), 1.0, SEPIA, U(r, 0.4, 0.8), r=r, skip=0.35)
        xx += U(r, 34, 44)
    # ---- the roof: grey tiles in rows, round tile-ends along the eave
    rm = Q(s, [(SX0 - 30, EAVE), (SX1 + 30, EAVE), (SX1 + 10, 428), (SX0 - 10, 428)])
    s.wash(rm, SLATE, 0.75, wob=1, soft=0.8, edge=0.4, ew=5, flow=0.5, fs=90, gran=0.5, body=1)
    s.wash(rm, INDIGO, 0.4, wob=2, bleed=8, edge=0.1, grad=s.vgrad(428, EAVE, 0.2, 1))
    xx = SX0 - 20
    while xx < SX1 + 20:                        # the tile channels running up the slope
        s.ink([(xx, EAVE - 4), (xx + J(r, 1) - (xx - 1540) * 0.012, 432)], 1.2, INKBLUE, 0.9, skip=0.15)
        hl(s, [(xx + 9, EAVE - 8), (xx + 9 - (xx - 1540) * 0.012, 436)], 3, 0.18)
        xx += 26
    for yy in (452, 478, 502):
        line(s, r, (SX0 - 14, yy), (SX1 + 14, yy + J(r, 1)), 1.0, INKBLUE, 0.7, over=0, skip=0.3)
    ridge = s.rect(SX0 - 14, 414, SX1 + 14, 432)
    put(s, ridge, SLATE, 0.8, lift=0.6, wob=0.6, edge=0.45, ew=3)
    box(s, r, SX0 - 14, 414, SX1 + 14, 432, 2.0, INKBLUE, 1.4, over=5)
    for xx in np.arange(SX0 - 18, SX1 + 30, 26):
        e = s.ellipse(xx, EAVE + 4, 11, 11)
        put(s, e, SLATE, 0.7, lift=0.7, wob=0.4, edge=0.5, ew=2)
        oval(s, xx, EAVE + 4, 11, 11, 1.2, INKBLUE, 1.1)
        s.wash(s.ellipse(xx, EAVE + 4, 4, 4), INKBLUE, 0.5, wob=0.2, soft=0.5)
    line(s, r, (SX0 - 32, EAVE), (SX1 + 32, EAVE), 2.2, INKBLUE, 1.5, dbl=0.5)
    s.wash(s.rect(SX0, EAVE + 15, SX1, EAVE + 70), INK, 0.45, wob=0, soft=0, edge=0, grad=s.vgrad(EAVE + 15, EAVE + 70, 1, 0))
    # ---- two small lamps over the sign, and the sign: old wood, "桂花" carved and inked, the smaller line under it
    bx0, by0, bx1, by1 = 1150, 548, 1690, 712
    for lx, ly in LAMPS[:2]:
        s.ink([(lx, EAVE + 14), (lx + 4, ly - 30), (lx + 22, ly - 20)], 2.2, INK, 1.3)
        shade = Q(s, [(lx + 10, ly - 28), (lx + 34, ly - 26), (lx + 44, ly - 6), (lx + 2, ly - 8)])
        put(s, shade, INK, 0.8, lift=0.5, wob=0.4)
        s.lift(s.ellipse(lx + 23, ly - 6, 14, 5), 1.0, soft=0.8)
        s.wash(s.ellipse(lx + 23, ly - 6, 14, 5), OCHRE, 0.3, wob=0.3, soft=0.6)
        cone = Q(s, [(lx + 8, ly - 6), (lx + 38, ly - 6), (lx + 140, by1 + 20), (lx - 90, by1 + 20)])
        cone = cv2.GaussianBlur(cone, (0, 0), 18)
        s.wash(cone, None, 0.35, wob=0, soft=0, edge=0, lift=True)
        s.wash(cone, AMBER, 0.15, wob=0, soft=0, edge=0)
    bm = s.rect(bx0, by0, bx1, by1)
    put(s, bm, SIENNA, 0.55, lift=0.95, wob=0.8, edge=0.45, ew=6, flow=0.5, fs=90, gran=0.5)
    s.wash(bm, UMBER, 0.3, wob=2, bleed=10, edge=0.1, flow=0.8, fs=60)
    s.wash(bm, OCHRE, 0.35, wob=0, soft=0, edge=0, grad=s.vgrad(by0, by1, 1, 0.3))
    for _ in range(8):
        y = U(r, by0 + 8, by1 - 8)
        a = U(r, bx0, bx1 - 200)
        s.stroke([(a, y), (a + U(r, 120, 400), y + J(r, 2))], U(r, 4, 12), BURNT, U(r, 0.1, 0.2), dry=0.5, edge=0.2, wob=1, taper=(0.1, 0.2), clip=bm)
        if r.random() < 0.5:
            s.ink([(a, y), (a + U(r, 80, 300), y + J(r, 2))], 0.9, BURNT, 0.6, skip=0.4)
    big = letters(s, "桂花", SIGN[0], SIGN[1] + 2, 108, KAI, 400, sx=1.08)
    s.wash(big, None, 0.3, wob=0.6, soft=0.8, edge=0, lift=True)
    sh = cv2.warpAffine(big, np.float32([[1, 0, 3], [0, 1, 4]]), (s.w, s.h))
    s.wash(sh * (1 - big), OCHRE, 0.5, wob=0.4, soft=0.8)                     # the carved edge catching the lamps
    paint_letters(s, big, INK, 1.6, wob=0.6)
    paint_letters(s, big, BURNT, 0.3, wob=0.6)
    small = letters(s, "桂花釀 · 桂花糕", SIGN[0], 686, 34, KAI, 400)
    paint_letters(s, small, INK, 1.3, wob=0.3)
    box(s, r, bx0, by0, bx1, by1, 2.4, SEPIA, 1.5, over=6, dbl=0.5)
    box(s, r, bx0 + 9, by0 + 8, bx1 - 9, by1 - 8, 1.1, SEPIA, 0.9, over=2, skip=0.2)
    for xx in (bx0 + 40, bx1 - 40):
        s.ink([(xx, EAVE + 10), (xx, by0)], 1.6, INK, 1.2)
    # ---- the window: the room behind it first (it is the light of the whole picture)
    wx0, wy0, wx1, wy1 = WIN
    gm = s.rect(wx0, wy0, wx1, wy1)
    s.lift(gm, 1.0, soft=0.5, wob=0)
    s.wash(gm, OCHRE, 0.38, wob=0, soft=0.5, edge=0, flow=0.5, fs=120, body=1)
    s.wash(gm, SIENNA, 0.35, wob=0, soft=0, edge=0, grad=s.rgrad(1420, 900, 200, 700, 0, 1, sy=0.6), flow=0.5, fs=100)
    s.wash(gm, AMBER, 0.25, wob=0, soft=0, edge=0, grad=s.rgrad(1420, 880, 40, 420, 1, 0, sy=0.7))
    s.lift(s.rgrad(1420, 876, 10, 160, 1, 0, sy=0.8), 0.5, soft=0, wob=0)
    s.wash(s.ellipse(1420, 868, 34, 12), SIENNA, 0.6, wob=0.4, soft=0.8)                                   # a pendant shade inside
    s.ink([(1420, wy0), (1420, 860)], 1.4, SEPIA, 1.1)
    # three shelves of jars, plates of cakes at the bottom
    shelves = (968, 1088, 1200)
    for sy_ in shelves:
        sm = s.rect(wx0, sy_, wx1, sy_ + 12)
        put(s, sm, SIENNA, 0.65, lift=0.5, wob=0.6, edge=0.45, ew=2)
        hl(s, [(wx0, sy_ + 1), (wx1, sy_ + 1)], 2.4, 0.4)
        line(s, r, (wx0, sy_ + 12), (wx1, sy_ + 12), 1.4, SEPIA, 1.2)
        s.wash(s.rect(wx0, sy_ + 12, wx1, sy_ + 34), UMBER, 0.3, wob=0, soft=0, edge=0, grad=s.vgrad(sy_ + 12, sy_ + 34, 1, 0))
    for si, sy_ in enumerate(shelves[:2]):
        xx = wx0 + 22
        while xx < wx1 - 60:
            jw = U(r, 44, 66) if si == 0 else U(r, 54, 80)
            jh = U(r, 80, 100) if si == 0 else U(r, 70, 92)
            jar(s, r, xx + jw / 2, sy_, jw, jh)
            xx += jw + U(r, 10, 22)
    # bottom shelf: cakes on plates, a tea tin, a stack of boxes tied with string
    xx = wx0 + 30
    while xx < wx1 - 100:
        kind = r.random()
        if kind < 0.55:
            cake_plate(s, r, xx + 60, 1200, 120)
            xx += 140
        elif kind < 0.8:
            tw, th = 52, 76
            tm = s.rect(xx, 1200 - th, xx + tw, 1200)
            put(s, tm, (PERSIMMON, TEAL, OCHRE)[int(r.integers(0, 3))], 0.7, lift=0.8, wob=0.5, edge=0.45, ew=2)
            s.wash(s.rect(xx, 1200 - th, xx + tw, 1200 - th + 12), SEPIA, 0.5, wob=0.3, soft=0.6)
            s.wash(s.rect(xx + 10, 1200 - th + 24, xx + tw - 10, 1200 - 18), CREAM, 0.15, wob=0.3, soft=0.6)
            paint_letters(s, letters(s, "茶", xx + tw / 2, 1200 - 40, 22, KAI, 400), INK, 1.1, wob=0.2)
            box(s, r, xx, 1200 - th, xx + tw, 1200, 1.3, SEPIA, 1.2, over=1)
            xx += tw + 20
        else:
            for j in range(3):
                bw = 110 - j * 10
                bm2 = s.rect(xx + j * 5, 1200 - 30 * (j + 1), xx + j * 5 + bw, 1200 - 30 * j)
                put(s, bm2, CREAM, 0.3, lift=0.8, wob=0.5, edge=0.4, ew=2)
                s.wash(bm2, OCHRE, 0.2, wob=0.5, soft=1)
                box(s, r, xx + j * 5, 1200 - 30 * (j + 1), xx + j * 5 + bw, 1200 - 30 * j, 1.2, SEPIA, 1.1, over=1)
                s.ink([(xx + j * 5 + bw / 2, 1200 - 30 * (j + 1)), (xx + j * 5 + bw / 2, 1200 - 30 * j)], 1.0, PERSIMMON, 1.0)
            xx += 130
    # a hand-written price card propped against the glass
    pc = Q(s, [(1640, 1150), (1760, 1146), (1764, 1196), (1642, 1200)])
    put(s, pc, CREAM, 0.08, lift=1.0, wob=0.4)
    paint_letters(s, letters(s, "桂花釀 180", 1702, 1164, 18, KAI, 400), INK, 1.2, wob=0.2)
    paint_letters(s, letters(s, "桂花糕 一盒 120", 1702, 1186, 15, KAI, 400), PERSIMMON, 1.1, wob=0.2)
    contour(s, pc, 1.0, SEPIA, 0.9)
    # the glass: the dusk lying on it in long diagonal sheets, the frame and its muntins
    for x0_, w_ in ((1000, 90), (1300, 40), (1590, 130), (1820, 50)):
        rf = Q(s, [(x0_, wy0), (x0_ + w_, wy0), (x0_ + w_ - 200, wy1), (x0_ - 200, wy1)]) * gm
        s.wash(rf, RAIN, 0.14, wob=2, soft=6, edge=0, flow=0.5, fs=60)
    mx = [wx0 + (wx1 - wx0) * q for q in (1 / 3, 2 / 3)]
    frame = s.zeros()
    for x in mx:
        frame = np.maximum(frame, s.rect(x - 7, wy0, x + 7, wy1))
    frame = np.maximum(frame, s.rect(wx0 - 22, wy0 - 22, wx1 + 22, wy0) + s.rect(wx0 - 22, wy1, wx1 + 22, wy1 + 26) + s.rect(wx0 - 22, wy0, wx0, wy1) + s.rect(wx1, wy0, wx1 + 22, wy1))
    frame = np.clip(frame, 0, 1)
    put(s, frame, UMBER, 0.85, lift=0.9, wob=0.6, edge=0.45, ew=3, flow=0.5, fs=40)
    s.wash(frame, INDIGO, 0.2, wob=1, soft=2)
    for x in mx:
        hl(s, [(x - 7, wy0 + 4), (x - 7, wy1 - 4)], 2.5, 0.35)
        s.wash(s.ribbon([(x - 6, wy0 + 4), (x - 6, wy1 - 4)], 2.5, smooth=False), AMBER, 0.4, wob=0.3, soft=0.6)
        box(s, r, x - 7, wy0, x + 7, wy1, 1.4, SEPIA, 1.2, over=1, dbl=0.2)
    box(s, r, wx0, wy0, wx1, wy1, 1.8, SEPIA, 1.4, over=3, dbl=0.4)
    box(s, r, wx0 - 22, wy0 - 22, wx1 + 22, wy1 + 26, 2.4, SEPIA, 1.5, over=6, dbl=0.5)
    hl(s, [(wx0 - 20, wy1 + 24), (wx1 + 20, wy1 + 24)], 3, 0.4)
    s.wash(s.ribbon([(wx0 - 20, wy1 + 4), (wx1 + 20, wy1 + 4)], 4, smooth=False), AMBER, 0.45, wob=0.4, soft=0.8)
    # under the window: a panelled base
    for i in range(3):
        a0 = wx0 + i * (wx1 - wx0) / 3 + 16
        a1 = wx0 + (i + 1) * (wx1 - wx0) / 3 - 16
        pm = s.rect(a0, wy1 + 48, a1, G - 22)
        s.wash(pm, SEPIA, 0.35, wob=1, soft=1.5, edge=0.5, ew=4)
        s.lift(s.rect(a0 + 10, wy1 + 56, a1 - 10, G - 30), 0.1, soft=3, wob=1.5)
        box(s, r, a0, wy1 + 48, a1, G - 22, 1.5, SEPIA, 1.3, over=3, dbl=0.3)
        hl(s, [(a0 + 4, wy1 + 50), (a1 - 4, wy1 + 50)], 2.0, 0.3)
    # ---- the awning: indigo cloth on a frame, sagging between ribs, its valance lettered
    ax0, ax1, at_, af, av = 900, 1940, 742, 812, 850
    slope = Q(s, [(ax0 + 20, at_), (ax1 - 20, at_), (ax1 + 10, af), (ax0 - 10, af)])
    val = s.rect(ax0 - 10, af, ax1 + 10, av)
    aw = np.maximum(slope, val)
    put(s, aw, NAVY, 0.7, lift=1.0, wob=0.8, edge=0.45, ew=5, flow=0.6, fs=90, gran=0.45)
    s.wash(aw, PRUSSIAN, 0.3, wob=2, bleed=8, edge=0.1, flow=0.8, fs=60)
    s.wash(slope, INK, 0.4, wob=0, soft=0, edge=0, grad=s.vgrad(at_, af, 1, 0.1))
    s.wash(val, AMBER, 0.2, wob=0, soft=0, edge=0, grad=s.vgrad(af, av, 0, 1))                          # lit from the window under it
    for i in range(1, 8):                        # ribs, and the cloth sagging between them
        xt = ax0 + 20 + (ax1 - ax0 - 40) * i / 8
        xb = ax0 - 10 + (ax1 - ax0 + 20) * i / 8
        s.ink([(xt, at_), (xb, af)], 1.3, INK, 1.0)
        hl(s, [(xt - 40, at_ + 6), (xb - 40, af - 4)], 6, 0.12, clip=slope)
    s.lift(letters(s, "桂 花 · 茶 · 甜 點", (ax0 + ax1) / 2, (af + av) / 2 + 2, 28, KAI, 400), 1.0, soft=0.5, wob=0.3)
    s.wash(letters(s, "桂 花 · 茶 · 甜 點", (ax0 + ax1) / 2, (af + av) / 2 + 2, 26, KAI, 400), CREAM, 0.2, wob=0.3, soft=0.5)
    pl(s, r, [(ax0 + 20, at_), (ax1 - 20, at_)], 2.0, INK, 1.4, over=4)
    pl(s, r, [(ax0 - 10, af), (ax1 + 10, af)], 1.8, INK, 1.3, over=4)
    pl(s, r, [(ax0 - 10, av), (ax1 + 10, av)], 2.0, INK, 1.4, over=4)
    pl(s, r, [(ax0 + 20, at_), (ax0 - 10, af), (ax0 - 10, av)], 1.8, INK, 1.3, over=2)
    pl(s, r, [(ax1 - 20, at_), (ax1 + 10, af), (ax1 + 10, av)], 1.8, INK, 1.3, over=2)
    for xx in np.arange(ax0, ax1, 70):            # its scalloped hem
        s.ink(ell(xx + 35, av, 35, 9, 0.1, math.pi - 0.1, 14), 1.3, INK, 1.1)
    # ---- the door: a lit doorway, a noren of indigo cloth split in three, a small "營業中" board
    dx0, dy0, dx1, dy1 = DOOR
    dm = s.rect(dx0, dy0, dx1, dy1)
    s.lift(dm, 1.0, soft=0.5, wob=0)
    s.wash(dm, OCHRE, 0.45, wob=0, soft=0.5, edge=0, flow=0.5, fs=60, body=1)
    s.wash(dm, SIENNA, 0.4, wob=2, bleed=12, edge=0, grad=s.vgrad(dy0, dy1, 0.2, 1))
    s.wash(s.rect(dx0 + 40, 1100, dx1 - 30, dy1), UMBER, 0.4, wob=3, bleed=16, edge=0.1)                 # a counter inside
    s.wash(s.rect(dx0, dy1 - 30, dx1, dy1), SEPIA, 0.5, wob=0.5, soft=1)
    nb = dy0 + 300
    for i in range(3):
        a0 = dx0 + i * (dx1 - dx0) / 3 + 2
        a1 = dx0 + (i + 1) * (dx1 - dx0) / 3 - 2
        pn = Q(s, [(a0, dy0 + 6), (a1, dy0 + 6), (a1 + J(r, 3), nb + J(r, 6)), (a0 + J(r, 3), nb + J(r, 6))])
        put(s, pn, NAVY, 0.7, lift=0.9, wob=0.8, edge=0.45, ew=4, flow=0.6, fs=40, gran=0.45)
        s.wash(pn, AMBER, 0.15, wob=1, soft=3, edge=0, grad=s.vgrad(dy0 + 100, nb, 0, 1))
        s.ink([((a0 + a1) / 2 + J(r, 4), dy0 + 30), ((a0 + a1) / 2 + J(r, 6), nb - 10)], 1.0, INK, 0.6, skip=0.3)
        contour(s, pn, 1.5, INK, 1.2)
    mc = s.ellipse((dx0 + dx1) / 2, dy0 + 150, 52, 52)
    s.lift(mc, 0.85, soft=0.6, wob=0.4)
    s.wash(mc, CREAM, 0.15, wob=0.4, soft=0.6)
    s.wash(letters(s, "桂", (dx0 + dx1) / 2, dy0 + 152, 62, KAI, 400), NAVY, 1.1, wob=0.4, soft=0.5)
    s.ink([(dx0 - 10, dy0 + 6), (dx1 + 10, dy0 + 6)], 4, SEPIA, 1.4)
    fr = np.clip(s.rect(dx0 - 22, dy0 - 22, dx1 + 22, dy1 + 4) - s.rect(dx0, dy0, dx1, dy1 + 4), 0, 1)
    put(s, fr, UMBER, 0.85, lift=0.9, wob=0.6, edge=0.45, ew=3)
    for x in (dx0 - 2, dx1 + 2):
        hl(s, [(x, dy0 + 4), (x, dy1 - 4)], 3, 0.4)
        s.wash(s.ribbon([(x, dy0 + 4), (x, dy1 - 4)], 3, smooth=False), AMBER, 0.45, wob=0.3, soft=0.6)
    box(s, r, dx0 - 22, dy0 - 22, dx1 + 22, dy1 + 4, 2.4, SEPIA, 1.5, over=5, dbl=0.5)
    box(s, r, dx0, dy0, dx1, dy1, 1.6, SEPIA, 1.3, over=2)
    om = Q(s, [(dx1 + 34, 940), (dx1 + 92, 942), (dx1 + 90, 1100), (dx1 + 32, 1098)])
    put(s, om, CREAM, 0.25, lift=0.9, wob=0.5, edge=0.4, ew=2)
    s.wash(om, OCHRE, 0.25, wob=0.5, soft=1)
    paint_letters(s, letters(s, "營業中", dx1 + 62, 1020, 34, KAI, 400, vertical=True), PERSIMMON, 1.3, wob=0.3)
    s.ink([(dx1 + 40, 940), (dx1 + 62, 912), (dx1 + 84, 940)], 1.0, SEPIA, 1.0)
    contour(s, om, 1.4, SEPIA, 1.2)
    # the lamp over the door and its pool of light
    lx, ly = LAMPS[2]
    s.ink([(lx - 40, ly - 50), (lx - 20, ly - 56), (lx, ly - 40), (lx, ly - 26)], 2.6, INK, 1.4)
    shade = Q(s, [(lx - 10, ly - 28), (lx + 10, ly - 28), (lx + 30, ly), (lx - 30, ly)])
    put(s, shade, CELADON, 0.6, lift=0.8, wob=0.4, edge=0.45, ew=2)
    s.wash(shade, INK, 0.3, wob=0.4, soft=1, grad=s.hgrad(lx - 30, lx + 30, 0, 1))
    contour(s, shade, 1.4, INK, 1.2)
    bulb = s.ellipse(lx, ly + 6, 9, 8)
    s.lift(bulb, 1.0, soft=0.6, wob=0)
    s.wash(bulb, OCHRE, 0.2, wob=0.3, soft=0.5)
    halo = s.rgrad(lx, ly + 10, 12, 230, 1, 0)
    s.wash(halo, None, 0.4, wob=0, soft=0, edge=0, lift=True)
    s.wash(halo, AMBER, 0.2, wob=0, soft=0, edge=0)
    # ---- the bench under the window, a cushion and a cup left on it
    bx, by = BENCH
    seat = s.rect(bx - 300, by - 14, bx + 300, by + 10)
    put(s, seat, SIENNA, 0.75, lift=0.95, wob=0.6, edge=0.45, ew=3, flow=0.5, fs=40, gran=0.5)
    s.wash(seat, OCHRE, 0.35, wob=0.5, soft=1, grad=s.vgrad(by - 14, by + 10, 1, 0))
    top = Q(s, [(bx - 300, by - 14), (bx + 300, by - 14), (bx + 288, by - 34), (bx - 288, by - 34)])
    put(s, top, SIENNA, 0.5, lift=0.95, wob=0.5, edge=0.4, ew=2)
    s.wash(top, AMBER, 0.4, wob=0.5, soft=1)
    for lx_ in (bx - 270, bx + 270, bx - 240, bx + 240):
        near = lx_ in (bx - 270, bx + 270)
        lm = s.rect(lx_ - 10, by + 10, lx_ + 10, G + (12 if near else -6))
        put(s, lm, UMBER if near else SEPIA, 0.85, lift=0.9, wob=0.5, edge=0.45, ew=2)
        contour(s, lm, 1.4, SEPIA, 1.2)
    s.ink([(bx - 260, by + 80), (bx + 260, by + 82)], 3, SEPIA, 1.2)
    pl(s, r, [(bx - 300, by - 14), (bx - 288, by - 34), (bx + 288, by - 34), (bx + 300, by - 14)], 1.8, SEPIA, 1.4, over=3)
    box(s, r, bx - 300, by - 14, bx + 300, by + 10, 2.0, SEPIA, 1.5, over=4, dbl=0.4)
    cu = s.poly([(bx - 220, by - 30), (bx - 60, by - 32), (bx - 50, by - 56), (bx - 120, by - 70), (bx - 214, by - 58)])
    put(s, cu, OSMAN, 0.55, lift=0.95, wob=0.6, edge=0.45, ew=3)
    s.wash(cu, SIENNA, 0.3, wob=1, soft=2, grad=s.vgrad(by - 70, by - 30, 0, 1))
    contour(s, cu, 1.6, SEPIA, 1.3)
    s.ink([(bx - 200, by - 46), (bx - 70, by - 46)], 1.0, SEPIA, 0.6, skip=0.3)
    tc = s.poly([(bx + 140, by - 34), (bx + 176, by - 34), (bx + 172, by - 70), (bx + 144, by - 70)], smooth=False)
    put(s, tc, CREAM, 0.15, lift=0.95, wob=0.4, edge=0.4, ew=2)
    s.wash(s.ellipse(bx + 158, by - 70, 14, 4), SIENNA, 0.6, wob=0.3, soft=0.5)
    contour(s, tc, 1.4, SEPIA, 1.2)
    oval(s, bx + 158, by - 70, 14, 4, 1.2, SEPIA, 1.2)
    # ---- potted plants at the shop's foot
    for (px, pw, ph, c, kind) in ((838, 80, 90, TERRA, "fern"), (2268, 70, 80, CELADON, "tall"), (1930, 46, 50, TERRA, "small")):
        pot = Q(s, [(px - pw / 2, G - ph), (px + pw / 2, G - ph), (px + pw * 0.4, G + 4), (px - pw * 0.4, G + 4)])
        put(s, pot, c, 0.7, lift=0.9, wob=0.6, edge=0.45, ew=3)
        s.wash(pot, PAYNE, 0.4, wob=1, soft=2, grad=s.hgrad(px - pw / 2, px + pw / 2, 0, 1))
        put(s, s.rect(px - pw / 2 - 4, G - ph - 8, px + pw / 2 + 4, G - ph + 4), c, 0.75, lift=0.6, wob=0.4)
        path(s, [(px - pw / 2, G - ph), (px - pw * 0.4, G + 4), (px + pw * 0.4, G + 4), (px + pw / 2, G - ph)], 1.6, SEPIA, 1.3, over=2, r=r)
        nl = {"fern": 26, "tall": 14, "small": 10}[kind]
        for _ in range(nl):
            if kind == "fern":
                a = U(r, -2.9, -0.25)
                Lf = U(r, 70, 170)
            elif kind == "tall":
                a = U(r, -1.9, -1.25)
                Lf = U(r, 160, 300)
            else:
                a = U(r, -2.6, -0.5)
                Lf = U(r, 30, 70)
            b0 = (px + J(r, pw * 0.25), G - ph)
            tip = (b0[0] + math.cos(a) * Lf, b0[1] + math.sin(a) * Lf)
            mid = ((b0[0] + tip[0]) / 2 + J(r, 10), (b0[1] + tip[1]) / 2 - Lf * 0.1)
            s.stroke([b0, mid, tip], U(r, 8, 16) if kind != "tall" else U(r, 10, 18), (LEAF, MOSS, PRUSSIAN)[int(r.integers(0, 3))], U(r, 0.55, 0.9),
                     taper=(0.1, 0.8), wob=0.8, edge=0.5, body=0.95)
            if r.random() < 0.4:
                s.ink([b0, mid, tip], 1.0, INKBLUE, 0.8, skip=0.2)
    # the shop's own edges
    line(s, r, (SX0, EAVE), (SX0 + J(r, 1.5), G), 2.4, SEPIA, 1.5, dbl=0.5)
    line(s, r, (SX1, EAVE), (SX1 + J(r, 1.5), G), 2.4, SEPIA, 1.5, dbl=0.5)
    s.wash(s.rect(SX0, G - 40, SX1, G + 6), INK, 0.3, wob=2, bleed=8, edge=0.1, clip=fm)


def jar(s, r, cx, by, w, h):
    """A glass jar of osmanthus syrup: gold, the little flowers hanging in it, a red paper cap tied with string."""
    body = np.maximum(s.rect(cx - w / 2, by - h + 12, cx + w / 2, by - 6), s.ellipse(cx, by - 6, w / 2, 5))
    body = np.maximum(body, s.ellipse(cx, by - h + 12, w / 2, 8))
    s.lift(body, 0.8, soft=0.6, wob=0)
    lv = by - h * U(r, 0.62, 0.84)
    liq = body * s.vgrad(lv - 1, lv + 2, 0, 1)
    s.wash(liq, OSMAN, U(r, 0.6, 0.95), wob=0.5, soft=0.8, edge=0.5, ew=3, flow=0.4, fs=20, body=1)
    s.wash(liq, AMBER, 0.35, wob=0.5, soft=2, grad=s.hgrad(cx - w / 2, cx + w / 2, 0, 1))
    s.wash(liq, BURNT, 0.2, wob=0.5, soft=2, grad=s.vgrad(by - h * 0.6, by, 0, 1))
    g = rs(int(cx * 7 + by))
    for _ in range(int(w * h / 90)):            # the flowers suspended in it
        fx, fy = cx + J(g, w * 0.4), by - U(g, 8, h * 0.62)
        s.wash(s.ellipse(fx, fy, 1.6, 1.6), AMBER, 1.2, wob=0.2, soft=0.4, clip=liq)
    s.lift(s.rect(cx - w / 2 + 5, by - h + 18, cx - w / 2 + 11, by - 12), 0.75, soft=1, wob=0.4)
    s.wash(s.rect(cx + w / 2 - 8, by - h + 16, cx + w / 2 - 2, by - 10), SEPIA, 0.3, wob=0.4, soft=1)
    cap = s.poly([(cx - w / 2 - 6, by - h + 14), (cx - w / 2 + 2, by - h - 2), (cx, by - h - 8), (cx + w / 2 - 2, by - h - 2), (cx + w / 2 + 6, by - h + 14), (cx + w / 2 - 4, by - h + 22), (cx - w / 2 + 4, by - h + 22)])
    put(s, cap, PERSIMMON, U(r, 0.6, 0.85), lift=0.8, wob=0.5, edge=0.45, ew=2)
    s.wash(cap, BURNT, 0.25, wob=0.5, soft=1.5, grad=s.hgrad(cx - w / 2, cx + w / 2, 0, 1))
    s.lift(s.ribbon([(cx - w / 2 + 2, by - h + 12), (cx + w / 2 - 2, by - h + 13)], 2.0, smooth=False), 0.7, soft=0.5, wob=0)
    lab = s.rect(cx - w * 0.3, by - h * 0.5, cx + w * 0.3, by - h * 0.22)
    if r.random() < 0.6:
        put(s, lab, CREAM, 0.12, lift=0.8, wob=0.4)
        s.wash(letters(s, "桂", cx, by - h * 0.36, max(12, w * 0.32), KAI, 400), PERSIMMON, 1.2, wob=0.2, soft=0.4)
        box(s, r, cx - w * 0.3, by - h * 0.5, cx + w * 0.3, by - h * 0.22, 0.9, SEPIA, 0.9, over=0.5)
    contour(s, body, 1.4, SEPIA, 1.2)
    contour(s, cap, 1.2, SEPIA, 1.1)
    oval(s, cx, by - h + 12, w / 2, 7, 1.0, SEPIA, 0.8, a0=0.1, a1=math.pi - 0.1)


def cake_plate(s, r, cx, by, w):
    """A white plate of 桂花糕: pale translucent squares with osmanthus scattered on top."""
    pm = s.ellipse(cx, by - 6, w / 2, 10)
    put(s, pm, CREAM, 0.12, lift=1.0, wob=0.4, edge=0.4, ew=2)
    s.wash(pm, SLATE, 0.2, wob=0.4, soft=1.5, grad=s.vgrad(by - 14, by, 0, 1))
    oval(s, cx, by - 6, w / 2, 10, 1.3, SEPIA, 1.2)
    for row, (ox, oy) in enumerate(((-34, -14), (0, -14), (34, -14), (-17, -38), (17, -38))):
        x0, y0 = cx + ox - 15, by + oy - 22
        cm = np.maximum(s.rect(x0, y0 + 6, x0 + 30, y0 + 24), Q(s, [(x0, y0 + 6), (x0 + 30, y0 + 6), (x0 + 36, y0), (x0 + 6, y0)]))
        put(s, cm, CREAM, 0.3, lift=0.9, wob=0.4, edge=0.5, ew=2)
        s.wash(cm, OCHRE, 0.25, wob=0.4, soft=1)
        s.wash(s.rect(x0, y0 + 6, x0 + 30, y0 + 24), OCHRE, 0.18, wob=0.4, soft=1)
        for _ in range(5):
            s.wash(s.ellipse(x0 + U(r, 6, 32), y0 + U(r, 1, 6), 1.8, 1.4), OSMAN, 1.3, wob=0.2, soft=0.4)
        contour(s, cm, 1.0, SEPIA, 1.0)


def ground(S, r):
    """The lane: old concrete slabs with dark joints, the window's light lying warm on them, a few fallen flowers."""
    s = Sub(S, 0, G - 20, W, H)
    gm = s.rect(-10, G, W + 10, H + 10)
    s.wash(gm, SLATE, 0.5, wob=1, soft=0.9, edge=0.2, ew=6, flow=0.6, fs=260, gran=0.55, body=1)
    s.wash(gm, DUSK, 0.4, wob=3, bleed=14, edge=0.1, flow=0.8, fs=180, gran=0.5, clip=gm)
    s.wash(gm, PAYNE, 0.35, wob=3, bleed=14, edge=0.1, grad=s.vgrad(G, H, 0.2, 1), clip=gm)
    mottle(s, r, 0, G + 40, W, H, [INDIGO, UMBER, VIOLET], 8, 0.14, size=(100, 260), clip=gm)
    # slab joints in perspective
    VPX = 1540
    for z in (0.5, 1.3, 2.4):
        y = HZ + (G - HZ) * 8 / (8 - z)
        if y < H:
            seg(s, (-10, y + J(r, 2)), (W + 10, y + J(r, 2)), 1.4, INKBLUE, 0.9, r=r, skip=0.25, bow=0.001)
    for x in np.arange(-1200, 4400, 260):
        a = (x, G)
        b = (VPX + (x - VPX) * 8 / 5.4, HZ + (G - HZ) * 8 / 5.4)
        seg(s, a, b, 1.2, INKBLUE, 0.8, r=r, skip=0.3, bow=0.002)
    # a gutter along the houses
    s.wash(s.rect(-10, G, W + 10, G + 16), INK, 0.4, wob=0.6, soft=1, edge=0.4)
    line(s, r, (-10, G + 16), (W + 10, G + 17), 1.6, INKBLUE, 1.2, dbl=0.4)
    # the lights lying on the ground
    for (x0, x1, a, c) in ((WIN[0], WIN[1], 0.6, OCHRE), (DOOR[0], DOOR[2], 0.5, OCHRE)):
        pool = s.poly([(x0, G), (x1, G), (x1 + (x1 - VPX) * 0.4 + 80, H), (x0 + (x0 - VPX) * 0.4 - 80, H)], smooth=False)
        pool = cv2.GaussianBlur(pool, (0, 0), 26) * s.vgrad(G, H + 60, 1, 0.1)
        s.wash(pool, None, a * 1.35, wob=0, soft=0, edge=0, lift=True, gran=0.3, flow=0.4, fs=70)
        s.wash(pool, c, 0.3, wob=0, soft=0, edge=0, flow=0.5, fs=60)
        s.wash(pool, AMBER, 0.14, wob=0, soft=0, edge=0)
    s.wash(s.rect(BENCH[0] - 300, G + 4, BENCH[0] + 300, G + 34), INK, 0.2, wob=2, bleed=8, edge=0)     # the bench's shadow
    # fallen osmanthus
    for _ in range(90):
        x = U(r, 1700, W - 40) if r.random() < 0.7 else U(r, 900, 1700)
        y = U(r, G + 10, H - 10)
        s.wash(s.ellipse(x, y, U(r, 1.8, 3.2), U(r, 1.2, 2.0)), OSMAN, 1.0, wob=0.2, soft=0.4, body=0.9)


def dusk(S, r):
    """Blue hour over everything that is not lit; the lit things keep their warmth."""
    s = S
    lit = np.zeros((H, W), np.float32)
    for (x0, y0, x1, y1) in (WIN, DOOR, (300, 300, 560, 480), (1150, 548, 1690, 712), (2850, 430, 2990, 560)):
        lit = np.maximum(lit, s.rect(x0 + 4, y0 + 4, x1 - 4, y1 - 4))
    lit = cv2.GaussianBlur(lit, (0, 0), 5)
    sk = s.vgrad(700, 900, 1, 0)
    s.wash((1 - lit) * (1 - sk), DUSK, 0.32, wob=0, soft=0, edge=0, flow=0.6, fs=400, gran=0.2)
    s.wash((1 - lit) * (1 - sk), INDIGO, 0.14, wob=0, soft=0, edge=0, grad=s.vgrad(400, H, 1, 0.5), flow=0.6, fs=300, gran=0.2)
    # the window's warmth reaching the wall and awning around it
    wx0, wy0, wx1, wy1 = WIN
    g = s.rgrad((wx0 + wx1) / 2, (wy0 + wy1) / 2, 300, 900, 1, 0, sy=0.6) * (1 - lit)
    s.wash(g, None, 0.12, wob=0, soft=0, edge=0, lift=True)
    s.wash(g, AMBER, 0.12, wob=0, soft=0, edge=0)


# ================================================================================================ near things
def os_tree():
    """An osmanthus shrub in a big glazed pot, near: dense small dark leaves, clusters of tiny orange flowers, the
    window's warm light on its left side."""
    ox, oy = 2300, 760
    s = Off(W - ox + 40, H - oy, ox, oy, 402)
    r = rs(402)
    px, pb = 2740, H                          # the pot's foot
    # the pot: a round-bellied pot in a dark blue-green glaze
    rimy, belly = 1462, 1600
    pts = [(px - 168, rimy), (px + 168, rimy), (px + 200, rimy + 50), (px + 218, belly), (px + 196, H - 10), (px + 160, H + 20), (px - 160, H + 20), (px - 196, H - 10), (px - 218, belly), (px - 200, rimy + 50)]
    pm = s.poly(pts)
    put(s, pm, TEAL, 0.75, lift=1.0, wob=0.8, edge=0.45, ew=5, flow=0.5, fs=60, gran=0.45)
    s.wash(pm, INDIGO, 0.55, wob=1, soft=4, edge=0, grad=s.hgrad(px - 80, px + 218, 0, 1))
    s.wash(pm, PRUSSIAN, 0.3, wob=2, bleed=10, edge=0.1, flow=0.8, fs=40)
    s.wash(pm, OCHRE, 0.3, wob=1, soft=5, edge=0, grad=s.hgrad(px - 218, px - 80, 1, 0))
    hl(s, [(px - 150, rimy + 40), (px - 190, belly - 30), (px - 178, belly + 60)], 9, 0.5, dry=0.25)
    s.ink(ell(px, rimy + 54, 200, 20, 0.1, math.pi - 0.1, 30), 1.2, INKBLUE, 0.8, skip=0.2)
    for _ in range(12):                       # crackle in the glaze
        x0, y0 = px + J(r, 170), U(r, rimy + 70, H)
        s.ink([(x0, y0), (x0 + J(r, 20), y0 + U(r, 10, 30)), (x0 + J(r, 30), y0 + U(r, 20, 50))], 0.8, INKBLUE, 0.5, skip=0.3)
    rim = s.ellipse(px, rimy, 170, 22)
    put(s, rim, TEAL, 0.6, lift=0.9, wob=0.5, edge=0.45, ew=3)
    s.wash(s.ellipse(px, rimy + 3, 152, 15), SEPIA, 0.9, wob=0.5, soft=1)
    contour(s, pm, 2.2, INKBLUE, 1.4)
    oval(s, px, rimy, 170, 22, 2.0, INKBLUE, 1.4)
    # the trunk and branches going up into the leaves
    for pts_, w_ in (([(px - 10, 1462), (px - 20, 1300), (px - 60, 1160)], 26), ([(px - 20, 1300), (px + 60, 1180), (px + 120, 1060)], 16),
                     ([(px - 40, 1220), (px - 160, 1120), (px - 240, 1060)], 12)):
        s.stroke(pts_, w_, SEPIA, 0.9, taper=(0.0, 0.6), wob=1, edge=0.5, body=1)
        s.ink(pts_, 1.4, INK, 1.0, skip=0.2)
    # the leaves, in masses
    crown = s.zeros()
    masses = ((px - 160, 1180, 200, 150, 21), (px + 60, 1100, 230, 180, 22), (px + 230, 1260, 180, 160, 23), (px - 40, 1320, 260, 110, 24),
              (px - 300, 1300, 120, 110, 25), (px + 120, 950, 160, 110, 26), (px - 120, 1020, 130, 100, 27))
    for (cx, cy, rx, ry, sd) in masses:
        m0 = leafmass(s, r, cx, cy, rx, ry, int(rx * ry / 32), (11, 20), (LEAF, INDIGO, MOSS), (0.4, 0.6),
                      light=(2200, 1250, 520, OCHRE, 0.35), seed=sd)
        crown = np.maximum(crown, m0)
    s.wash(crown, DUSK, 0.3, wob=2, soft=6, edge=0, grad=s.hgrad(px - 50, W, 0, 1))
    s.wash(crown, OCHRE, 0.25, wob=2, soft=6, edge=0, grad=s.hgrad(px - 450, px - 60, 1, 0))
    # midribs on some of the nearest leaves
    g = rs(28)
    for _ in range(70):
        cx, cy, rx, ry, _sd = masses[int(g.integers(0, len(masses)))]
        a = g.random() * 2 * math.pi
        d = g.random() ** 0.7
        x, y = cx + math.cos(a) * rx * d * 0.8, cy + math.sin(a) * ry * d * 0.8
        ang = g.random() * math.pi
        L = U(g, 9, 16)
        s.ink([(x - math.cos(ang) * L, y - math.sin(ang) * L), (x + math.cos(ang) * L, y + math.sin(ang) * L)], 1.0, INKBLUE, 0.7, taper=(0.2, 0.5))
    # flowers in the axils
    fpts = []
    for _ in range(70):
        cx, cy, rx, ry, _sd = masses[int(g.integers(0, len(masses)))]
        a = g.random() * 2 * math.pi
        d = g.random() ** 0.6
        fpts.append((cx + math.cos(a) * rx * d * 0.85, cy + math.sin(a) * ry * d * 0.85))
    blossoms(s, r, fpts, (6, 16), 10, 1.15, seed=29)
    contour(s, crown, 1.6, INKBLUE, 1.0, skip=0.3)
    fl = sorted(fpts, key=lambda p: p[1])[:36]
    s.save("os_tree", crop=True, anchor=(px - ox, pb - oy), q=92,
           meta={"flowers": [[int(x - px), int(y - pb)] for x, y in fl], "lit": [-420, -480]})


def os_fg():
    """A near wall corner at the left edge: old plaster over brick, a bracket lamp with a warm bulb, a meter box."""
    s = Off(420, H, 0, 0, 403)
    r = rs(403)
    x1 = 236
    wm = Q(s, [(-10, -10), (x1, -10), (x1 + 6, H + 10), (-10, H + 10)])
    put(s, wm, DUSK, 0.65, lift=1.0, wob=1.0, edge=0.45, ew=6, flow=0.5, fs=120, gran=0.5)
    s.wash(wm, INDIGO, 0.45, wob=3, bleed=12, edge=0.1, flow=0.7, fs=100)
    s.wash(wm, PAYNE, 0.4, wob=0, soft=2, edge=0, grad=s.vgrad(800, H, 0, 1))
    mottle(s, r, 0, 0, x1, H, [UMBER, VIOLET, PAYNE], 6, 0.15, size=(40, 120), clip=wm)
    for _ in range(5):                        # plaster fallen away, brick showing
        cx, cy = U(r, 20, x1 - 40), U(r, 200, 1500)
        bm = s.blob(cx, cy, U(r, 40, 80), U(r, 24, 44), irr=0.3, n=9) * wm
        s.lift(bm, 0.3, soft=1, wob=1)
        s.wash(bm, BURNT, 0.22, wob=1.5, soft=1, edge=0.7, ew=2, gran=0.6)
        for j in range(5):
            y = cy - 32 + j * 14
            s.ink([(cx - 80, y), (cx + 80, y + J(r, 1))], 0.9, SEPIA, 0.6, skip=0.3, clip=cv2.GaussianBlur(bm, (0, 0), 1))
            for i in range(-3, 4):
                xx = cx + i * 28 + (14 if j % 2 else 0)
                s.ink([(xx, y), (xx, y + 14)], 0.8, SEPIA, 0.5, clip=cv2.GaussianBlur(bm, (0, 0), 1))
    streaks(s, r, 10, x1 - 10, 0, H, 10, PAYNE, 0.25, (6, 16), clip=wm)
    # a meter box and its pipe
    mb = s.rect(60, 980, 170, 1130)
    put(s, mb, SLATE, 0.6, lift=0.9, wob=0.6, edge=0.45, ew=3)
    s.wash(mb, PAYNE, 0.4, wob=1, soft=2, grad=s.hgrad(60, 170, 0, 1))
    oval(s, 115, 1040, 30, 30, 1.6, INK, 1.3)
    s.lift(s.ellipse(115, 1040, 26, 26), 0.4, soft=1)
    box(s, r, 60, 980, 170, 1130, 2.0, INK, 1.4, over=3, dbl=0.4)
    s.ink([(150, 1130), (152, 1400), (148, H)], 10, SLATE, 0.8)
    s.ink([(145, 1130), (147, 1400), (143, H)], 1.4, INK, 1.1)
    # the lamp: an iron bracket, an enamel shade, a bare warm bulb
    lx, ly = 300, 640
    s.ink([(x1, ly - 80), (x1 + 40, ly - 82), (lx, ly - 60), (lx, ly - 36)], 4.0, INK, 1.4)
    s.ink([(x1, ly - 40), (x1 + 30, ly - 66)], 3.0, INK, 1.3)
    put(s, s.rect(x1 - 4, ly - 100, x1 + 10, ly - 20), INK, 0.8, lift=0.5, wob=0.4)
    halo = s.rgrad(lx, ly + 8, 10, 260, 1, 0)
    s.wash(halo * wm, None, 0.55, wob=0, soft=0, edge=0, lift=True)
    s.wash(halo, AMBER, 0.3, wob=0, soft=0, edge=0)
    shade = s.poly([(lx - 14, ly - 38), (lx + 14, ly - 38), (lx + 52, ly + 2), (lx - 52, ly + 2)])
    put(s, shade, TEAL, 0.7, lift=1.0, wob=0.4, edge=0.45, ew=2)
    s.wash(shade, INK, 0.35, wob=0.4, soft=1, grad=s.hgrad(lx - 52, lx + 52, 0, 1))
    s.wash(s.rect(lx - 52, ly - 4, lx + 52, ly + 4), CREAM, 0.15, wob=0.3, soft=0.6, body=1)
    contour(s, shade, 1.8, INK, 1.4)
    bulb = s.ellipse(lx, ly + 14, 13, 14)
    s.lift(bulb, 1.0, soft=0.5, wob=0)
    s.wash(bulb, OCHRE, 0.25, wob=0.3, soft=0.6, body=1)
    oval(s, lx, ly + 14, 13, 14, 1.2, SEPIA, 1.0)
    # a strip of old poster paper, half torn
    pp = Q(s, [(30, 1240), (190, 1236), (194, 1380), (34, 1396)]) * wm
    put(s, pp, CREAM, 0.25, lift=0.85, wob=0.6, edge=0.4, ew=2)
    s.wash(pp, DUSK, 0.35, wob=2, bleed=6)
    paint_letters(s, letters(s, "尋人啟事", 112, 1270, 28, SERIF, 900), INK, 0.9, wob=0.3)
    for q in range(4):
        line(s, r, (48, 1300 + q * 18), (176, 1298 + q * 18), 1.0, SEPIA, 0.6, over=0, skip=0.3)
    s.lift(s.blob(150, 1380, 60, 30, irr=0.4) * pp, 0.7, soft=1)
    line(s, r, (x1, -10), (x1 + 6, H + 10), 2.8, INK, 1.6, dbl=0.5)
    hl(s, [(x1 - 6, 0), (x1, H)], 4, 0.25)
    s.save("os_fg", crop=True, anchor=(130, H), q=92, meta={"lamp": [lx - 130, ly + 14 - H]})


ALL = {"shop_bg": os_bg, "shop_tree": os_tree, "shop_fg": os_fg}
