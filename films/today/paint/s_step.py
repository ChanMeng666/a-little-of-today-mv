"""Night by the river (a Taipei riverside like 大稻埕): wide concrete steps going down to the water, seen from behind
and a little above the step where the two of us will sit (the figures are another layer: the step is left empty).
Sheet 2880x1620. The far bank's waterline is the horizon (y=600); the river fills 600..1100; the steps run from the
water's edge (y~1100) to the bottom of the sheet; the step we sit on is the broad band below its nosing at y=1330.
Behind it all: 觀音山's ridge, the low skyline across the water, a long lit bridge coming in from the right.
A lamp somewhere behind the camera (up and to the left) lays a faint warm light on the steps: shadows fall toward the
river, a little to the right.

  sp_bg  opaque. meta:
         seat_me [x,y]   the left seat: the point on the step's tread under her (her figure's base);
         seat_you [x,y]  the right seat, 360 px to the right;
         figure_h        the height a seated adult should be drawn there (seat to the top of the head), px;
         reflect [x0,y0,x1,y1]  the river band to shimmer at runtime (the reflections are painted into it);
         bridge [[x,y],...]     the bridge's lamps, near (right) to far (left);
         lights [[x,y],...]     brighter lights on the far bank (for runtime twinkle; their reflections are below them);
         moon [x,y,r]; nosing_y  the front edge of the sitting step; water_y the near waterline.
"""
import math

import cv2
import numpy as np

from kit import *
from s_stall import (KAI, SERIF, Off, Sub, J, U, box, contour, drags, hl, letters, line, mottle, pl, put, streaks)

W, H = 2880, 1620
HZ = 600                         # the far bank's waterline
VPX = 1440
WATER = 1100                     # the near waterline, at the foot of the steps
NOSE = 1330                      # the front edge of the step we sit on
SEAT_ME = (1250, 1452)
SEAT_YOU = (1610, 1452)
FIG_H = 540
BRIDGE = []
LIGHTS = []
MOON = (2330, 150, 26)


def sp_bg():
    S = Sheet(W, H, 501)
    r = rs(501)
    sky(S, r)
    mountain(S, r)
    skyline(S, r)
    bridge(S, r)
    river(S, r)
    steps(S, r)
    night(S, r)
    meta = {"seat_me": list(SEAT_ME), "seat_you": list(SEAT_YOU), "figure_h": FIG_H, "reflect": [0, HZ + 4, W, WATER - 6],
            "bridge": [[int(x), int(y)] for x, y in BRIDGE], "lights": [[int(x), int(y)] for x, y in LIGHTS[:40]],
            "moon": list(MOON), "nosing_y": NOSE, "water_y": WATER}
    S.save("sp_bg", opaque=True, q=90, meta=meta)


def sky(S, r):
    s = Sub(S, 0, 0, W, HZ + 2)
    F = s.full()
    s.wash(F, INDIGO, 0.95, wob=0, soft=0, edge=0, flow=0.45, fs=500, gran=0.2)
    s.wash(F, PAYNE, 0.7, wob=0, soft=0, edge=0, grad=s.vgrad(-100, 380, 1, 0), flow=0.5, fs=400, gran=0.2)
    s.wash(F, INKBLUE, 0.4, wob=0, soft=0, edge=0, grad=s.vgrad(-100, 200, 1, 0), flow=0.6, fs=300, gran=0.2)
    # the city's glow low in the sky: violet, then a warm haze right on the skyline
    s.wash(F, None, 0.45, wob=0, soft=0, edge=0, lift=True, grad=s.vgrad(250, 560, 0, 1))
    s.wash(F, VIOLET, 0.35, wob=0, soft=0, edge=0, grad=s.vgrad(240, 520, 0, 1), flow=0.6, fs=300, gran=0.15)
    s.wash(F, ROSE, 0.22, wob=0, soft=0, edge=0, grad=s.vgrad(380, 590, 0, 1), flow=0.7, fs=250, gran=0.15)
    s.wash(F, OCHRE, 0.16, wob=0, soft=0, edge=0, grad=s.vgrad(470, 600, 0, 1), flow=0.7, fs=200)
    # a few night clouds lit from below
    for cx, cy, rx, ry, k in ((700, 140, 520, 34, 0.3), (1700, 90, 700, 30, 0.26), (2500, 260, 420, 26, 0.24), (1200, 300, 600, 22, 0.2)):
        s.wash(s.blob(cx, cy, rx, ry, irr=0.45, n=17), PAYNE, k, wob=8, bleed=26, edge=0.25, ew=8, flow=0.5, fs=150, gran=0.25)
        s.wash(s.blob(cx + J(r, 60), cy + ry * 0.6, rx * 0.6, ry * 0.4, irr=0.4), VIOLET, k * 0.5, wob=6, bleed=14, edge=0.2)
    # the moon, a little haloed, and stars (few: the city is bright)
    mx, my, mr = MOON
    s.lift(s.rgrad(mx, my, mr, mr * 5, 1, 0), 0.3, soft=0, wob=0)
    mm = s.ellipse(mx, my, mr, mr)
    s.lift(mm, 1.0, soft=0.6, wob=0)
    s.wash(mm, CREAM, 0.3, wob=0.3, soft=0.6, edge=0.6, ew=2)
    s.wash(s.ellipse(mx - 9, my - 4, mr * 0.9, mr * 0.95), PAYNE, 0.5, wob=0.3, soft=1.5)          # a crescent: the rest in shadow
    s.wash(s.ellipse(mx + 6, my + 6, 5, 4), SLATE, 0.2, wob=0.3, soft=1)
    for _ in range(40):
        x, y = U(r, 0, W), U(r, 0, 330) ** 1.0
        rr = U(r, 0.8, 2.0)
        s.lift(s.ellipse(x, y, rr, rr), U(r, 0.5, 1.0), soft=0.4, wob=0)


def mountain(S, r):
    """觀音山: a long soft ridge across the water, darker than the sky, losing its foot in the haze."""
    s = Sub(S, 0, 200, W, HZ + 20)
    pts = [(-20, 540), (120, 480), (300, 430), (480, 380), (620, 332), (720, 312), (800, 318), (880, 300), (960, 322), (1100, 360), (1240, 380),
           (1380, 420), (1560, 440), (1760, 470), (1960, 500), (2200, 520), (2500, 540), (2900, 550), (2900, HZ - 4), (-20, HZ - 4)]
    m = s.poly(pts)
    s.wash(m, VIOLET, 0.5, wob=3, soft=1, edge=0.35, ew=8, flow=0.5, fs=200, gran=0.35)
    s.wash(m, INDIGO, 0.55, wob=3, bleed=16, edge=0.1, flow=0.6, fs=150, grad=s.vgrad(300, 600, 1, 0.3))
    s.wash(m, None, 0.35, wob=0, soft=0, edge=0, lift=True, grad=s.vgrad(440, 600, 0, 1))
    for _ in range(6):                     # its folds
        x = U(r, 250, 1600)
        y0 = 340 + abs(x - 800) * 0.14
        s.stroke([(x, y0 + 10), (x + U(r, 30, 90), y0 + U(r, 80, 140))], U(r, 20, 50), INDIGO, 0.15, dry=0.3, edge=0.2, wob=2, clip=m)
    s.ink([(p[0], p[1] + 1) for p in pts[:18]], 1.2, INKBLUE, 0.7, skip=0.45)


def skyline(S, r):
    """The far bank: a levee with a road of orange lamps along it, low blocks with many small lit windows, a few
    taller towers with red lamps, a park's dark trees."""
    s = Sub(S, 0, 330, W, HZ + 30)
    x = -30.0
    g = rs(77)
    while x < W + 30:
        w_ = U(g, 50, 160)
        tall = g.random() < 0.12
        top = U(g, 380, 420) if tall else U(g, 470, 560)
        m = s.rect(x, top, x + w_, HZ)
        c = (PAYNE, INDIGO, VIOLET)[int(g.integers(0, 3))]
        s.wash(m, c, U(g, 0.55, 0.85), wob=0.7, soft=0.9, edge=0.4, ew=4, flow=0.5, fs=60, gran=0.3, body=1)
        s.wash(m, INKBLUE, 0.3, wob=2, bleed=6, edge=0.1, grad=s.vgrad(top, HZ, 0.2, 1))
        line(s, r, (x - 1, top), (x + w_ + 1, top + J(r, 0.8)), 1.1, INKBLUE, 0.9, over=1)
        # windows: rows of small lights, some warm, some cold
        rows = int((HZ - 18 - top) / 14)
        for j in range(rows):
            for i in range(int(w_ / 12)):
                if g.random() < 0.22:
                    wx, wy = x + 5 + i * 12 + J(g, 1), top + 8 + j * 14
                    wm = s.rect(wx, wy, wx + U(g, 4, 7), wy + U(g, 5, 8))
                    s.lift(wm, U(g, 0.6, 0.95), soft=0.5, wob=0)
                    s.wash(wm, (OCHRE, OCHRE, AMBER, CREAM, CELADON)[int(g.integers(0, 5))], U(g, 0.3, 0.6), wob=0.3, soft=0.5)
                    if g.random() < 0.08:
                        LIGHTS.append((wx + 3, wy + 3))
        if tall:
            s.lift(s.ellipse(x + w_ / 2, top - 6, 3, 3), 1.0, soft=0.5, wob=0)
            s.wash(s.ellipse(x + w_ / 2, top - 6, 3, 3), PERSIMMON, 0.9, wob=0.2, soft=0.5)
            s.wash(s.rgrad(x + w_ / 2, top - 6, 2, 18, 1, 0), PERSIMMON, 0.3, wob=0, soft=0, edge=0)
            LIGHTS.append((x + w_ / 2, top - 6))
        x += w_ * U(g, 0.75, 1.05)
    # a neon sign on one roof
    nx, ny = 560, 452
    nm = letters(s, "大飯店", nx, ny, 34, SERIF, 900)
    s.lift(cv2.GaussianBlur(nm, (0, 0), 4), 0.5, soft=0, wob=0)
    s.lift(nm, 1.0, soft=0.4, wob=0.3)
    s.wash(nm, ROSE, 0.6, wob=0.3, soft=0.5)
    s.wash(cv2.GaussianBlur(nm, (0, 0), 6), ROSE, 0.3, wob=0, soft=0, edge=0)
    LIGHTS.append((nx, ny))
    # the park's trees along the bank, and the levee with its road lamps
    for _ in range(30):
        tx = U(g, 0, W)
        s.wash(s.blob(tx, HZ - 18, U(g, 30, 80), U(g, 14, 30), irr=0.35), INKBLUE, 0.7, wob=2, soft=1, edge=0.3, body=1)
    lv = s.rect(-10, HZ - 14, W + 10, HZ + 2)
    s.wash(lv, INKBLUE, 0.85, wob=0.6, soft=0.8, edge=0.3, body=1)
    for x in np.arange(10, W, 46):
        lx = x + J(g, 6)
        ly = HZ - 20 - J(g, 1)
        s.lift(s.ellipse(lx, ly, 2.6, 2.2), 1.0, soft=0.4, wob=0)
        s.wash(s.ellipse(lx, ly, 2.6, 2.2), AMBER, 0.6, wob=0.2, soft=0.4)
        s.wash(s.rgrad(lx, ly, 1, 14, 1, 0), AMBER, 0.2, wob=0, soft=0, edge=0)
        if g.random() < 0.25:
            LIGHTS.append((lx, ly))
    line(s, r, (-10, HZ - 14), (W + 10, HZ - 14), 1.2, INKBLUE, 0.9, skip=0.3)


def bridge(S, r):
    """A long low bridge coming in from the right: deck, railing, piers going down into the water, a row of lamps."""
    s = Sub(S, 1100, 300, W, HZ + 60)
    near, far = (W + 60, 430), (1240, 588)
    def P(t, dy=0.0):
        x = near[0] + (far[0] - near[0]) * t
        y = near[1] + (far[1] - near[1]) * t
        sc = 1.5 * (1 - 0.82 * t)
        return x, y + dy * sc, sc
    deck = s.poly([P(t, 0)[:2] for t in np.linspace(0, 1, 12)] + [P(t, 34)[:2] for t in np.linspace(1, 0, 12)])
    s.wash(deck, INKBLUE, 0.9, wob=0.6, soft=0.8, edge=0.4, ew=3, body=1)
    s.wash(deck, PAYNE, 0.4, wob=1, bleed=4, edge=0.1)
    # piers
    for t in np.linspace(0.04, 0.96, 11):
        x, y, sc = P(t, 34)
        pm = s.rect(x - 12 * sc, y, x + 12 * sc, HZ + 2)
        s.wash(pm, INKBLUE, 0.85, wob=0.4, soft=0.7, edge=0.4, body=1)
    # the underside of the deck catching the lamps' light
    s.wash(s.poly([P(t, 26)[:2] for t in np.linspace(0, 1, 12)] + [P(t, 36)[:2] for t in np.linspace(1, 0, 12)]), AMBER, 0.3, wob=0.5, soft=1)
    # railing and lamps
    s.ink([P(t, -10)[:2] for t in np.linspace(0, 1, 20)], 1.4, INKBLUE, 1.1, smooth=False)
    s.ink([P(t, 0)[:2] for t in np.linspace(0, 1, 20)], 2.0, INKBLUE, 1.2, smooth=False)
    for t in np.linspace(0.0, 1.0, 120):
        x, y, sc = P(t, 0)
        s.ink([(x, y), (x, y - 10 * sc)], 0.8, INKBLUE, 0.8, taper=(0, 0))
    t = 0.0
    while t < 1.0:
        x, y, sc = P(t, 0)
        top = y - 46 * sc
        s.ink([(x, y), (x, top)], max(1.0, 2.6 * sc), INKBLUE, 1.2, taper=(0, 0))
        s.ink([(x, top), (x - 10 * sc, top - 2 * sc)], max(0.9, 2.0 * sc), INKBLUE, 1.1, taper=(0, 0))
        lx, ly = x - 10 * sc, top + 2 * sc
        s.wash(s.rgrad(lx, ly, 1, 34 * sc + 8, 1, 0), None, 0.5, wob=0, soft=0, edge=0, lift=True)
        s.wash(s.rgrad(lx, ly, 1, 34 * sc + 8, 1, 0), AMBER, 0.3, wob=0, soft=0, edge=0)
        bm = s.ellipse(lx, ly, max(2.0, 5 * sc), max(2.0, 4 * sc))
        s.lift(bm, 1.0, soft=0.4, wob=0)
        s.wash(bm, OCHRE, 0.3, wob=0.2, soft=0.5)
        BRIDGE.append((lx, ly))
        t += 0.06 * (1 - 0.6 * t)
    # traffic: a smear of headlights and tail lights along the deck
    for _ in range(14):
        tt = U(r, 0.05, 0.95)
        x, y, sc = P(tt, 14)
        c = (CREAM, PERSIMMON)[int(r.integers(0, 2))]
        m = s.ribbon([(x - 30 * sc, y), (x + 30 * sc, y + 1)], 4 * sc + 1, smooth=False)
        s.lift(m, 0.9, soft=0.8, wob=0)
        s.wash(m, c, 0.5, wob=0.3, soft=1)


def river(S, r):
    """Dark water: the sky's glow on its far half, ripples, and every light on the far bank dropped into it as a long
    broken vertical streak."""
    s = Sub(S, 0, HZ - 4, W, WATER + 30)
    wm = s.rect(-10, HZ, W + 10, WATER + 30)
    s.wash(wm, PRUSSIAN, 0.95, wob=0, soft=0, edge=0, flow=0.5, fs=300, gran=0.3, body=1)
    s.wash(wm, INDIGO, 0.6, wob=0, soft=0, edge=0, grad=s.vgrad(HZ, WATER, 0.3, 1), flow=0.6, fs=200, gran=0.3)
    s.wash(wm, None, 0.4, wob=0, soft=0, edge=0, lift=True, grad=s.vgrad(HZ, HZ + 160, 1, 0))           # the sky's glow lying on it
    s.wash(wm, VIOLET, 0.25, wob=0, soft=0, edge=0, grad=s.vgrad(HZ, HZ + 200, 1, 0), flow=0.6, fs=200)
    # the mountain and the skyline, inverted and dark, just under the far bank
    s.wash(s.rect(-10, HZ, W + 10, HZ + 60), INKBLUE, 0.45, wob=0, soft=0, edge=0, grad=s.vgrad(HZ, HZ + 60, 1, 0), flow=0.5, fs=80)
    # ripples and reflections are drawn as many small marks into a few masks, then laid as one wash each
    def canvas():
        return np.zeros((s.h * 2, s.w * 2), np.uint8)

    def mark(m, x, y, L, w_, v):
        cv2.ellipse(m, (int((x - s.ox) * 16), int((y - s.oy) * 16)), (max(1, int(L / 2 * 16)), max(1, int(w_ / 2 * 16))), J(r, 1.5), 0, 360,
                    int(np.clip(v, 0, 1) * 255), -1, cv2.LINE_AA, 3)

    def done(m):
        return cv2.resize(m, (s.w, s.h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255

    dark, light = canvas(), canvas()
    for _ in range(520):
        y = HZ + (WATER - HZ) * U(r, 0, 1) ** 1.4
        f = (y - HZ) / (WATER - HZ)
        x = U(r, -100, W)
        L = U(r, 30, 220) * (0.4 + f * 1.4)
        w_ = (1 + 4 * f) * U(r, 0.6, 1.4)
        if r.random() < 0.55:
            mark(dark, x, y, L, w_, U(r, 0.4, 1.0))
        else:
            mark(light, x, y, L, w_ * 0.8, U(r, 0.3, 0.8) * (1.2 - f))
    s.wash(done(dark), INKBLUE, 0.3, wob=0.6, soft=0.6, edge=0.3, ew=1.5, gran=0.3, flow=0.4, fs=60)
    s.lift(done(light), 0.25, soft=0.6, wob=0.4)
    # the reflections: under each light a glow and a column of broken dashes, wider and longer toward us
    srcs = [(x, 1.0, AMBER) for x, y in BRIDGE] + [(x, 0.6, (OCHRE, AMBER, CREAM, ROSE)[i % 4]) for i, (x, y) in enumerate(LIGHTS[:60])]
    g = rs(88)
    cols = {}
    glow = {}
    for (x0, bright, c) in srcs:
        L = (WATER - HZ) * U(g, 0.55, 1.0) * (0.5 + 0.5 * bright)
        gm = glow.setdefault(c, s.zeros())
        cv2.rectangle(gm, (int(x0 - 5 - s.ox), HZ - s.oy), (int(x0 + 5 - s.ox), int(HZ + L - s.oy)), bright, -1)
        m = cols.setdefault(c, canvas())
        y = HZ + 4
        while y < HZ + L:
            f = (y - HZ) / (WATER - HZ)
            ww = U(g, 6, 22) * (0.4 + 1.6 * f) * bright
            xx = x0 + J(g, 3 + 14 * f)
            mark(m, xx, y, ww, max(1.2, U(g, 1.5, 4) * (0.6 + f)), U(g, 0.7, 1.0) * (1 - 0.5 * (y - HZ) / max(1, L)))
            y += U(g, 4, 10) * (0.5 + 1.6 * f)
    fade = s.vgrad(HZ, WATER - 100, 1, 0)
    for c, gm in glow.items():
        gm = cv2.GaussianBlur(gm, (0, 0), 8) * fade
        s.wash(gm, None, 0.4, wob=0, soft=0, edge=0, lift=True)
        s.wash(gm, c, 0.25, wob=0, soft=0, edge=0)
    for c, m in cols.items():
        mm = done(m)
        s.lift(mm, 0.95, soft=0.6, wob=0.4)
        s.wash(mm, c, 0.5, wob=0.3, soft=0.6, edge=0.3, ew=1.2)
    # a soft warm column under the bridge's lamps, all together
    for x0, y0 in BRIDGE:
        col = s.rect(x0 - 8, HZ, x0 + 8, HZ + 300) * s.vgrad(HZ, HZ + 300, 1, 0)
        s.wash(cv2.GaussianBlur(col, (0, 0), 8), AMBER, 0.2, wob=0, soft=0, edge=0)
    # the near water, just below the steps: the step's edge doubled, a lap of light along it
    s.wash(s.rect(-10, WATER - 30, W + 10, WATER + 30), INKBLUE, 0.25, wob=2, bleed=10, edge=0, grad=s.vgrad(WATER - 30, WATER, 0, 1))
    for _ in range(40):                        # a lap of light along the bottom step
        x = U(r, 0, W)
        hl(s, [(x, WATER - 8 + J(r, 3)), (x + U(r, 30, 140), WATER - 8 + J(r, 3))], U(r, 1.5, 3), U(r, 0.2, 0.45))


def steps(S, r):
    """Concrete steps going down to the water, seen from above and behind: treads, nosings, the shadow each step throws
    on the one below, joints running toward the water, wet and mossy at the bottom; a cup someone left."""
    s = Sub(S, 0, WATER - 50, W, H)
    # tread bands, from the water's edge up to the step we sit on; each is lit on its far part, shadowed by the
    # nosing above on its near part (the risers face the river and are not seen)
    bands = [(WATER - 12, WATER + 14), (WATER + 14, WATER + 46), (WATER + 50, WATER + 92), (WATER + 98, WATER + 152), (WATER + 160, NOSE)]
    for i, (y0, y1) in enumerate(bands):
        m = s.rect(-10, y0, W + 10, y1 + 2)
        f = i / (len(bands) - 1)
        s.wash(m, SLATE, 0.62, wob=0.6, soft=0.8, edge=0.35, ew=4, flow=0.5, fs=200, gran=0.6, body=1)
        s.wash(m, INDIGO, 0.45 - 0.2 * f, wob=2, bleed=8, edge=0.1, flow=0.6, fs=150, gran=0.5)
        s.wash(m, INKBLUE, 0.75, wob=0, soft=0, edge=0, grad=s.vgrad(y1 - (y1 - y0) * 0.55, y1, 0, 1), flow=0.5, fs=60)   # the nearer step's shadow on it
        hl(s, [(-10, y0 + 2), (W + 10, y0 + 2)], 2 + 3 * f, 0.45 + 0.25 * f)                                 # its own nosing catching the lamp
        line(s, r, (-10, y0), (W + 10, y0 + J(r, 1)), 1.2 + 0.6 * f, INKBLUE, 1.2, dbl=0.4)
        line(s, r, (-10, y0 + 4 + 2 * f), (W + 10, y0 + 4 + 2 * f + J(r, 1)), 0.9, INKBLUE, 0.6, skip=0.4)
        if i < 2:                              # wet and green at the water
            s.wash(m, LEAF, 0.35, wob=3, bleed=8, edge=0.1, flow=0.8, fs=40)
            s.wash(m, INKBLUE, 0.4, wob=2, bleed=6, edge=0.1)
    # the step we sit on: the broad near band
    m = s.rect(-10, NOSE, W + 10, H + 10)
    s.wash(m, SLATE, 0.5, wob=0.6, soft=0.8, edge=0.3, ew=6, flow=0.5, fs=300, gran=0.45, body=1)
    s.wash(m, INDIGO, 0.3, wob=2, bleed=10, edge=0.1, flow=0.6, fs=200, gran=0.5)
    s.wash(m, PAYNE, 0.3, wob=0, soft=0, edge=0, grad=s.vgrad(NOSE, H, 0, 1))
    mottle(s, r, 0, NOSE + 20, W, H, [UMBER, VIOLET, SLATE, OCHRE], 10, 0.12, size=(120, 300), clip=m)
    drags(s, r, 0, NOSE + 20, W, H, PAYNE, 10, clip=m)
    s.spatter(500, INKBLUE, 0.35, box=(0, NOSE + 14, W, H), r=(0.8, 2.4))                 # the aggregate in the concrete
    s.spatter(200, None, 0.5, box=(0, NOSE + 14, W, H), r=(0.8, 2.0), lift=True)
    for (x, y, rx, ry) in ((380, 1560, 220, 40), (2600, 1470, 180, 30), (1800, 1590, 260, 30)):   # damp patches
        s.wash(s.blob(x, y, rx, ry, irr=0.35), PAYNE, 0.25, wob=4, bleed=10, edge=0.5, ew=6)
    hl(s, [(-10, NOSE + 4), (W + 10, NOSE + 4)], 7, 0.7)
    line(s, r, (-10, NOSE), (W + 10, NOSE + J(r, 1)), 2.4, INKBLUE, 1.5, dbl=0.5)
    line(s, r, (-10, NOSE + 12), (W + 10, NOSE + 12 + J(r, 1)), 1.0, INKBLUE, 0.7, skip=0.3)
    # the joints between the slabs, running toward the river
    for xb in np.arange(-1600, 4600, 420):
        a = (xb, H + 20)
        b = (VPX + (xb - VPX) * 0.36, WATER - 10)
        for (y0, y1) in [(NOSE, H + 20)] + bands:
            ta, tb = (y0 - a[1]) / (b[1] - a[1]), (y1 - a[1]) / (b[1] - a[1])
            pa = (a[0] + (b[0] - a[0]) * ta, y0 + 2)
            pb = (a[0] + (b[0] - a[0]) * tb, y1 - 2)
            if -50 < pa[0] < W + 50 or -50 < pb[0] < W + 50:
                seg(s, pa, pb, 1.2 if y0 == NOSE else 0.9, INKBLUE, 0.9 if y0 == NOSE else 0.6, r=r, skip=0.25)
    # cracks, a chipped nosing, moss in the bottom joints
    for _ in range(6):
        x, y = U(r, 0, W), U(r, NOSE + 30, H - 40)
        a = U(r, -0.5, 0.5)
        pts = [(x, y)]
        for _ in range(5):
            a += J(r, 0.5)
            x, y = x + math.cos(a) * U(r, 20, 45), y + math.sin(a) * U(r, 10, 25) + 6
            pts.append((x, y))
        s.ink(pts, 0.9, INKBLUE, 0.6, skip=0.35, tremor=1.2)
    for x in (420, 1980, 2560):
        s.wash(s.blob(x, NOSE + 4, 26, 8, irr=0.4), PAYNE, 0.6, wob=1, soft=1, edge=0.5)
    # the lamp behind us: a broad warm light on the near step and the nosings, fading toward the water
    lamp = s.rgrad(600, H + 200, 200, 1900, 1, 0, sy=0.5)
    s.wash(lamp, None, 0.45, wob=0, soft=0, edge=0, lift=True, gran=0.3, flow=0.4, fs=100)
    s.wash(lamp, OCHRE, 0.3, wob=0, soft=0, edge=0, flow=0.5, fs=90)
    s.wash(lamp, AMBER, 0.1, wob=0, soft=0, edge=0)
    # things left on the steps: a paper cup on its side, a bottle cap, a leaf or two (none where we sit)
    cx, cy = 2230, 1530                        # a paper cup someone left standing, its straw bent
    cm = s.poly([(cx - 26, cy - 92), (cx + 26, cy - 92), (cx + 19, cy), (cx - 19, cy)], smooth=False)
    s.wash(s.ellipse(cx + 40, cy + 4, 60, 9), INKBLUE, 0.45, wob=1, soft=4, edge=0)
    put(s, cm, CREAM, 0.22, lift=0.95, wob=0.5, edge=0.4, ew=2)
    s.wash(cm, SLATE, 0.35, wob=1, soft=2, grad=s.hgrad(cx - 26, cx + 26, 0, 1))
    s.wash(s.rect(cx - 30, cy - 66, cx + 30, cy - 40), PERSIMMON, 0.45, wob=0.4, soft=0.8, clip=cm)
    s.wash(s.ellipse(cx, cy - 92, 26, 6), MIST, 0.4, wob=0.3, soft=0.6, body=1)
    oval(s, cx, cy - 92, 26, 6, 1.3, SEPIA, 1.2)
    contour(s, cm, 1.4, SEPIA, 1.2)
    put(s, s.ribbon([(cx + 4, cy - 94), (cx + 8, cy - 140), (cx + 26, cy - 154)], 5), CREAM, 0.15, lift=0.9, wob=0.3)
    s.ink([(cx + 1.5, cy - 94), (cx + 5.5, cy - 140), (cx + 24, cy - 156)], 0.9, SEPIA, 1.0)
    s.ink([(cx + 6.5, cy - 94), (cx + 10.5, cy - 139), (cx + 27, cy - 151)], 0.9, SEPIA, 1.0)
    for (lx, ly) in ((620, 1560), (1930, 1400), (2460, 1580), (310, 1440)):
        s.wash(s.ellipse(lx, ly, 14, 5, J(r, 30)), (OCHRE, SIENNA)[int(r.integers(0, 2))], 0.6, wob=0.5, soft=0.6, edge=0.5, ew=2)
        s.ink([(lx - 12, ly), (lx + 12, ly)], 0.8, SEPIA, 0.6)


def night(S, r):
    """Pull the whole thing together: a faint cool glaze, the edges of the sheet a little darker."""
    s = S
    s.wash(s.full(), INDIGO, 0.1, wob=0, soft=0, edge=0, flow=0.6, fs=500, gran=0.2)
    v = 1 - s.rgrad(W / 2, H * 0.5, W * 0.35, W * 0.75, 1, 0, sy=0.6)
    s.wash(v, PAYNE, 0.3, wob=0, soft=0, edge=0, flow=0.5, fs=400)


ALL = {"step_bg": sp_bg}
