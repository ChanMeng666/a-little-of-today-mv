"""The window sill in the flat, a rainy afternoon: the mint in its pot against the glass. Sheet 2880x1620.

One-point perspective, vanishing point VP = (1440, 640) (eye a little above the sill, looking straight at the window).
The window is set deep in the wall: the plaster opening (440,100)-(2440,1150) narrows to the aluminium frame
(640,208)-(2240,1048); the sill (small square cream tiles) runs from the frame (y 1048) to its front edge (y 1205).

Layers, back to front (the page puts its rain-on-glass pass over the panes, between sl_grille and sl_frame):
  sl_out      opaque  what is outside, out of focus: the facades across the lane, grilles, AC units, a pale sky.
  sl_grille   sprite  the flat's own iron grille just outside the glass (pale green, scrolls). Place at 0:0 (anchor 0,0).
  sl_frame    sprite  plaster wall, reveals, aluminium sliding window (two sashes), tiled sill, a glass of water.
                      Place at 0:0. meta: panes [[x0,y0,x1,y1]..] glass, sill_y (where things stand), sill_back /
                      sill_front (y of the sill's back and front edges), light [x, y] (the brightest sky, upper left
                      pane; light comes from there toward us), pot [x, y] (where sl_pot stands), hand [x, y] (where
                      sl_hand's wrist rests), water [x, y] (the glass of water's foot), vp.
  sl_pot      sprite  terracotta pot in its saucer; anchor bottom centre (stands on the sill). meta soil [dx, dy].
  sl_mint     sprite  the mint; anchor at the stem base = soil centre. meta tips [[dx, dy, deg]..]: stem tips, deg is
                      the stem's direction in degrees clockwise from straight up (rotate sl_leaf by it).
  sl_mint_big sprite  the same plant later in the film, taller and fuller, with rain drops on it. Same conventions.
  sl_leaf     sprite  one young leaf on a short stalk, pointing up; anchor at the stalk base. Scale 0->1 at a tip.
  sl_hand     sprite  my right hand resting on the sill, cream knit cuff from the lower right; anchor at the wrist
                      (centre of the cuff's edge). meta tip [dx, dy] = the middle fingertip.
"""
import math

import cv2
import numpy as np

from kit import *

SW, SH = 2880, 1620
VP = (1440.0, 640.0)
OX0, OY0, OX1, OY1 = 440, 100, 2440, 1150          # the opening at the wall face
FX0, FY0, FX1, FY1 = 640, 208, 2240, 1048          # the aluminium frame's outer edge
FW = 36                                            # fixed frame member width
IX0, IY0, IX1, IY1 = FX0 + FW, FY0 + FW, FX1 - FW, FY1 - FW
MEET = 1440                                        # where the two sashes overlap
PANES = [[IX0 + 32, IY0 + 32, MEET - 26, IY1 - 36], [MEET + 26, IY0 + 32, IX1 - 32, IY1 - 36]]
SILL_BACK, SILL_FRONT, SILL_Y = 1048, 1205, 1122
POT = (1080, 1124)
HAND = (1660, 1112)
WATER = (2010, 1112)
LIGHT = (900, 360)
GZ = 0.80                                          # the grille's front plane, as a scale toward VP from the frame


def _lift(s, m, amt, clip=None, soft=1.5, wob=1.2, bleed=0.0, dry=0.0, gran=0.15, flow=0.0, fs=80):
    return s.wash(m, None, amt, wob=wob, soft=soft, edge=0, gran=gran, flow=flow, fs=fs, bleed=bleed, dry=dry, lift=True, clip=clip)


def _toward(p, k):
    """Scale a point toward the vanishing point (k<1 = farther away)."""
    return (VP[0] + (p[0] - VP[0]) * k, VP[1] + (p[1] - VP[1]) * k)


# ============================================================================================== outside
def _window(s, r, x, y, w, h, kind):
    """One window of the block across the lane, out of focus. kind: 'sky' (the glass gives back the pale sky),
    'dark' (open, the room unlit), 'lit' (a lamp on), 'curtain'. Each has its box grille and its rain hood."""
    m = s.rect(x, y, x + w, y + h)
    if kind == "lit":
        s.lift(m, 0.6, soft=6, wob=3)
        s.wash(m, OCHRE, 0.5, wob=3, bleed=10, edge=0.15)
        s.wash(s.rect(x + w * 0.6, y, x + w, y + h), SIENNA, 0.22, wob=3, bleed=10)
    elif kind == "sky":
        s.wash(m, SLATE, 0.32, wob=3, bleed=8, edge=0.3, flow=0.4)
        s.lift(s.poly([(x + w * 0.1, y + h), (x + w * 0.5, y), (x + w * 0.8, y), (x + w * 0.4, y + h)], smooth=False), 0.25, soft=8, wob=3)
    elif kind == "curtain":
        s.wash(m, SLATE, 0.3, wob=3, bleed=8, edge=0.3)
        s.wash(s.rect(x + 4, y + 4, x + w * (0.4 + 0.3 * r.random()), y + h), [ROSE, CREAM, CELADON][int(r.random() * 3)], 0.3, wob=3, bleed=10)
    else:
        s.wash(m, PAYNE, 0.55 + 0.2 * r.random(), wob=3, bleed=8, edge=0.25, flow=0.4)
    # the frame divides it in two
    s.wash(s.rect(x + w * 0.5 - 4, y, x + w * 0.5 + 4, y + h), MIST, 0.3, wob=2, bleed=3)
    # box grille: a few soft pale bars, its floor sticking out, the shadow under it
    for bx in np.arange(x - 4, x + w + 8, 34):
        s.lift(s.rect(bx, y - 8, bx + 6, y + h + 14), 0.22, soft=2.5, wob=1.5)
    s.lift(s.rect(x - 16, y + h + 6, x + w + 16, y + h + 18), 0.35, soft=3, wob=2)
    s.wash(s.rect(x - 16, y + h + 18, x + w + 16, y + h + 44), PAYNE, 0.22, wob=3, bleed=10)
    # a rain hood of corrugated plastic
    hood = s.poly([(x - 30, y - 50), (x + w + 30, y - 50), (x + w + 38, y - 10), (x - 38, y - 10)], smooth=False)
    s.lift(hood, 0.45, soft=4, wob=2)
    s.wash(hood, [CELADON, MIST, TEAL][int(r.random() * 3)], 0.35, wob=2, bleed=6, edge=0.3)
    s.wash(s.rect(x - 38, y - 12, x + w + 38, y + 10), PAYNE, 0.22, wob=2, bleed=6)
    # rain streaks from the sill down the tiles
    for _ in range(3):
        sx = x + r.random() * w
        s.stroke([(sx, y + h + 40), (sx + 3, y + h + 110 + 120 * r.random())], 8 + 12 * r.random(), UMBER, 0.08 + 0.08 * r.random(), wob=3, bleed=6, taper=(0, 0.7), dry=0.2)


def _ac(s, r, x, y):
    am = s.rect(x, y, x + 96, y + 64)
    s.lift(am, 0.7, soft=3, wob=2)
    s.wash(am, CREAM, 0.3, wob=2, bleed=5, edge=0.4)
    s.wash(s.ellipse(x + 60, y + 32, 22, 22), SLATE, 0.45, wob=1.5, bleed=4)
    s.wash(s.rect(x, y + 52, x + 96, y + 64), PAYNE, 0.25, wob=2, bleed=4)
    s.stroke([(x + 74, y + 66), (x + 76, y + 140 + 80 * r.random())], 10 + 8 * r.random(), SIENNA, 0.2, wob=3, bleed=6, taper=(0, 0.8))


def _tiles(s, m, cell, k, color):
    """Small square facade tiles, seen through rain and an unfocused eye: only a faint, uneven grid."""
    gx = np.abs(((s.X / cell) % 1.0) - 0.5)
    gy = np.abs(((s.Y / cell) % 1.0) - 0.5)
    g = np.clip((np.maximum(gx, gy) - 0.42) * 10, 0, 1).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.6) * m * sstep(0.35, 0.75, s.noise(120, 3))
    s.wash(g, color, k, wob=2, soft=0, edge=0, gran=0.2, flow=0.4)
    for _ in range(5):                                       # whole patches of slightly different tiles (repairs)
        roi = s._roi(m, 0)
        y0, y1, x0, x1 = roi[0].start, roi[0].stop, roi[1].start, roi[1].stop
        px, py = x0 + s_r.random() * (x1 - x0), y0 + s_r.random() * (y1 - y0)
        pw, ph = 80 + 160 * s_r.random(), 60 + 120 * s_r.random()
        s.wash(s.rect(px, py, px + pw, py + ph) * m, [SLATE, CREAM, color][int(s_r.random() * 3)], 0.10, wob=3, bleed=6, edge=0.4)


s_r = rs(219)


def outside():
    """Seen through the panes (x 708..2172, y 276..976): pale sky upper left over block A's roof and its rooftop
    shed; block A's tiled face below; block B taller on the right. Everything soft."""
    s = Sheet(SW, SH, 201)
    r = rs(201)
    s.glaze(MIST, 0.42, flow=0.5, fs=500)
    s.glaze(SLATE, 0.14, grad=s.rgrad(LIGHT[0], LIGHT[1] - 120, 150, 1600, 0.0, 1), flow=0.5, fs=360)
    for _ in range(7):
        s.wash(s.blob(r.random() * SW, 40 + r.random() * 380, 380 + 300 * r.random(), 60 + 60 * r.random(), 0.3), SLATE, 0.10, wob=8, bleed=40, edge=0.15)
    # ---- block A (left): rooftop shed and a water tower against the sky, then the tiled face
    AR = 470                      # A's parapet
    BX = 1660                     # where B begins
    shed = s.poly([(380, 330), (1180, 300), (1210, AR), (360, AR)], smooth=False)
    s.wash(shed, SLATE, 0.45, wob=4, bleed=12, edge=0.3)
    s.wash(s.poly([(370, 318), (1190, 286), (1196, 312), (376, 344)], smooth=False), SIENNA, 0.35, wob=4, bleed=8, edge=0.3)  # rusty roof edge
    for x in range(390, 1190, 30):
        s.stroke([(x, 330), (x + 2, AR)], 5, PAYNE, 0.12, wob=2, bleed=4, taper=(0, 0))
    s.wash(s.rect(560, 370, 660, AR), PAYNE, 0.35, wob=3, bleed=6)                               # its door
    for x in (1290, 1390):                                                                          # the water tower on legs
        s.stroke([(x, 370), (x + 4, AR)], 7, PAYNE, 0.35, wob=2, bleed=3)
    tank = s.poly([(1260, 250), (1420, 250), (1420, 370), (1260, 370)])
    s.lift(tank, 0.4, soft=6)
    s.wash(tank, MIST, 0.35, wob=3, bleed=10, edge=0.35)
    s.wash(s.rect(1350, 250, 1420, 370), SLATE, 0.25, wob=3, bleed=8)
    for _ in range(4):                                                                              # potted things on the roof
        x = 220 + r.random() * 1100
        s.wash(s.blob(x, AR - 18, 30 + 30 * r.random(), 22, 0.4), MOSS, 0.35, wob=3, bleed=8)
    fa = s.rect(-40, AR, BX, SH + 10)
    s.wash(fa, TILE, 0.62, wob=4, bleed=14, edge=0.3, ew=12, flow=0.5, fs=220, gran=0.5)
    s.wash(fa, SLATE, 0.18, wob=0, soft=0, edge=0, grad=s.vgrad(AR, SH, 0.0, 0.8), flow=0.5)
    s.wash(s.rect(-40, AR - 14, BX, AR + 22), CREAM, 0.3, wob=3, bleed=6, edge=0.4)               # parapet coping
    s.wash(s.rect(-40, AR + 22, BX, AR + 52), SLATE, 0.25, wob=3, bleed=10)
    _tiles(s, fa, 24, 0.10, SLATE)
    kinds = ["sky", "curtain", "dark", "sky", "lit", "sky", "curtain", "dark", "sky"]
    i = 0
    for wy in (620, 1010):
        for wx in (120, 560, 1080):
            _window(s, r, wx, wy, 300, 210, kinds[i % len(kinds)])
            i += 1
    for (x, y) in ((440, 860), (990, 640), (1420, 870)):
        _ac(s, r, x, y)
    # ---- block B (right): taller, beige-pink tiles
    fb = s.rect(BX, 120, SW + 40, SH + 10)
    s.wash(fb, SOY, 0.75, wob=4, bleed=14, edge=0.3, ew=12, flow=0.5, fs=220, gran=0.5)
    s.wash(fb, TERRA, 0.14, wob=4, bleed=20, edge=0.1, flow=0.6, fs=300)
    s.wash(fb, SLATE, 0.2, wob=0, soft=0, edge=0, grad=s.vgrad(120, SH, 0.0, 0.8), flow=0.5)
    s.wash(s.rect(BX, 120, BX + 40, SH), SLATE, 0.3, wob=3, bleed=8)                               # its corner turning away
    _tiles(s, fb, 19, 0.10, UMBER)
    i = 0
    for wy in (330, 720):
        for wx in (1790, 2230, 2640):
            _window(s, r, wx, wy, 280, 220, ["dark", "sky", "lit", "curtain", "sky", "dark"][i])
            i += 1
    for (x, y) in ((2110, 470), (2540, 900), (1930, 1000)):
        _ac(s, r, x, y)
    # the slot between the two blocks: pipes, darkness
    s.wash(s.rect(BX - 30, AR - 30, BX + 4, SH), PAYNE, 0.6, wob=3, bleed=8, edge=0.3)
    s.stroke([(BX - 14, AR - 30), (BX - 15, SH)], 8, SLATE, 0.4, wob=2, bleed=3)
    # laundry on a pole under one window, faded
    for j in range(4):
        x = 1820 + j * 62
        s.wash(s.rect(x, 590, x + 44, 640 + 30 * (j % 2)), [ROSE, MIST, CELADON, CREAM][j], 0.35, wob=3, bleed=6, edge=0.3)
    s.stroke([(1790, 588), (2090, 592)], 4, PAYNE, 0.4, wob=1, bleed=2, taper=(0, 0))
    # wires sagging across the lane
    for (a, b, sag) in (((-40, 380), (SW + 40, 520), 120), ((-40, 410), (SW + 40, 600), 140)):
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + sag)
        s.stroke([a, mid, b], 3, PAYNE, 0.35, wob=0.8, soft=1.5, taper=(0, 0), edge=0.1)
    s.wash(s.rect(-10, 1400, SW + 10, SH + 10), PAYNE, 0.5, wob=4, bleed=20, edge=0.2)         # the lane, far below
    # defocus, then air and paper on top
    s.A = cv2.GaussianBlur(s.A, (0, 0), 2.6)
    s.glaze(MIST, 0.16, grad=s.vgrad(0, SH, 1.0, 0.5), flow=0.6, fs=420, gran=0.45)
    _lift(s, s.rgrad(LIGHT[0], LIGHT[1], 100, 1000, 1, 0), 0.22, soft=0, wob=0)
    s.save("sl_out", opaque=True, q=88)


# ============================================================================================== the grille
def _spiral(c, r0, a0, turns, sgn, decay=0.42, n=90):
    t = np.linspace(0, turns * 2 * math.pi, n)
    rr = r0 * np.exp(-decay * t / (2 * math.pi) * 2.2)
    a = a0 + sgn * t
    return np.stack([c[0] + np.cos(a) * rr, c[1] + np.sin(a) * rr], 1)


def _lyre(x0, x1, y0, y1):
    """Two C-scrolls back to back hanging from the top bar of a bay: the commonest 鐵窗花 figure."""
    w, h = x1 - x0, y1 - y0
    cx = (x0 + x1) / 2
    out = []
    for sgn in (-1, 1):
        c = (cx + sgn * w * 0.20, y0 + h * 0.62)
        start = (cx, y0)
        a0 = math.atan2(start[1] - c[1], start[0] - c[0])
        r0 = math.hypot(start[0] - c[0], start[1] - c[1])
        out.append(_spiral(c, r0, a0, 0.95, -sgn, 0.55))
    return out


def grille():
    s = Sheet(SW, SH, 203)
    r = rs(203)
    gx0, gy0 = _toward((FX0, FY0), GZ)
    gx1, gy1 = _toward((FX1, FY1), GZ)
    paths = []                                    # (points, width)
    BW = 12
    # front plane: frame, verticals, two bands
    paths += [([(gx0, gy0), (gx1, gy0)], BW + 2), ([(gx0, gy1), (gx1, gy1)], BW + 2), ([(gx0, gy0), (gx0, gy1)], BW + 2), ([(gx1, gy0), (gx1, gy1)], BW + 2)]
    n = 16
    xs = np.linspace(gx0, gx1, n + 1)
    b1, b2 = gy0 + 118, gy1 - 96
    for y in (b1, b2):
        paths.append(([(gx0, y), (gx1, y)], BW))
    for x in xs[1:-1]:
        paths.append(([(x, gy0), (x, gy1)], BW - 2))
    for i in range(n):
        xa, xb = xs[i] + 6, xs[i + 1] - 6
        for sp in _lyre(xa, xb, gy0 + 6, b1 - 6):
            paths.append((sp, 7))
        # along the lower band: a small square stood on its corner, a ring inside it
        cx, cy = (xa + xb) / 2, (b2 + gy1) / 2
        hd = min((xb - xa) * 0.42, (gy1 - b2) * 0.40)
        cs = [(cx, cy - hd), (cx + hd, cy), (cx, cy + hd), (cx - hd, cy), (cx, cy - hd)]
        dia = np.vstack([np.linspace(cs[j], cs[j + 1], 8, endpoint=False) for j in range(4)] + [np.asarray(cs[:1], float)])
        paths.append((dia.tolist(), 6))
        paths.append((ell(cx, cy, hd * 0.36, hd * 0.36, 0, 2 * math.pi + 0.2, 24).tolist(), 5))
    # side and top panels of the cage, going back to the wall
    for y in (gy0, b1, b2, gy1, (b1 + b2) / 2):
        for xg, xf in ((gx0, FX0), (gx1, FX1)):
            k = 1 / GZ
            p = (xg, y)
            q = _toward(p, k)
            paths.append(([p, q], BW - 2))
    for t in np.linspace(0, 1, 5)[1:-1]:
        for xg in (gx0, gx1):
            pf = (xg, gy0)
            pb = _toward(pf, 1 / GZ)
            p = (pf[0] + (pb[0] - pf[0]) * t, pf[1] + (pb[1] - pf[1]) * t)
            q = (p[0], gy1 + (_toward((xg, gy1), 1 / GZ)[1] - gy1) * t)
            paths.append(([p, q], BW - 3))
    for x in xs[::2]:
        p = (x, gy0)
        paths.append(([p, _toward(p, 1 / GZ)], BW - 3))
    m = s.zeros()
    for pts, w in paths:
        m = np.maximum(m, s.ribbon(pts, w, smooth=len(pts) > 2))
    # paint: chalky pale green, wet, a cooler shade on the underside of every bar, rust where water sits
    base(s, m, CELADON, 0.95, wob=0.8, soft=1.2, edge=0.55, ew=3, flow=0.5, fs=80, gran=0.5)
    s.wash(m, TILE, 0.35, wob=0, soft=0, edge=0, grad=s.noise(200, 3), gran=0.4)
    under = np.clip(m - cv2.warpAffine(m, np.float32([[1, 0, 0], [0, 1, -4]]), (SW, SH)), 0, 1)
    s.wash(under, SLATE, 0.8, wob=0.5, soft=1.0, edge=0, clip=m)
    left = np.clip(m - cv2.warpAffine(m, np.float32([[1, 0, -4], [0, 1, 0]]), (SW, SH)), 0, 1)
    s.wash(left, RAIN, 0.5, wob=0.5, soft=1.0, edge=0, clip=m)
    top = np.clip(m - cv2.warpAffine(m, np.float32([[1, 0, 0], [0, 1, 3]]), (SW, SH)), 0, 1)
    _lift(s, top, 0.5, clip=m, soft=0.8, wob=0.5)
    # the side panels are farther in the shade of the wall
    side = np.maximum(s.rect(0, 0, gx0 - 6, SH), s.rect(gx1 + 6, 0, SW, SH)) + s.rect(0, 0, SW, gy0 - 6)
    s.wash(m * np.clip(side, 0, 1), PAYNE, 0.35, wob=1, soft=2, edge=0)
    rust = s.zeros()
    for _ in range(70):
        x, y = gx0 + r.random() * (gx1 - gx0), gy0 + r.random() * (gy1 - gy0)
        rust = np.maximum(rust, s.blob(x, y, 3 + 9 * r.random(), 2 + 4 * r.random(), 0.4, 7))
    s.wash(rust, SIENNA, 0.7, wob=1, soft=1, edge=0.6, ew=1.5, clip=m)
    for y in (b1, b2, gy1):
        for _ in range(4):
            x = gx0 + r.random() * (gx1 - gx0)
            s.stroke([(x, y + 6), (x + 1, y + 30 + 40 * r.random())], 4, SIENNA, 0.35, wob=1, bleed=1.5, taper=(0, 0.8))
    # pen: soft grey-green, only some of the edges (it is just past the glass, a little out of focus)
    for pts, w in paths:
        P = np.asarray(pts, float)
        if len(P) == 2:
            d = P[1] - P[0]
            L = math.hypot(*d) + 1e-6
            nrm = np.array([-d[1], d[0]]) / L
            for sgn in (-1, 1):
                if r.random() < 0.6:
                    a, b = P[0] + nrm * sgn * w * 0.5, P[1] + nrm * sgn * w * 0.5
                    s.ink([a, (a + b) / 2, b], 1.3, SLATE, 0.8, skip=0.35, tremor=0.8)
        else:
            s.ink(P, 1.2, SLATE, 0.8, skip=0.3, tremor=0.6)
    # beads of rain hanging under the horizontal bars
    dm = s.zeros()
    for y in (gy0, b1, b2, gy1):
        for _ in range(26):
            x = gx0 + r.random() * (gx1 - gx0)
            rad = 3 + 3 * r.random()
            dm = np.maximum(dm, s.ellipse(x, y + BW * 0.5 + rad * 0.8, rad * 0.8, rad))
    base(s, dm, RAIN, 0.6, wob=0.3, soft=0.6, edge=0.9, ew=1.5)
    _lift(s, dm * cv2.warpAffine(dm, np.float32([[1, 0, 1.5], [0, 1, 2.5]]), (SW, SH)), 0.6, soft=0.5, wob=0.2)
    s.A = cv2.GaussianBlur(s.A, (0, 0), 0.9)
    s.body = cv2.GaussianBlur(s.body, (0, 0), 0.9)
    s.save("sl_grille", crop=True, anchor=(0, 0), meta={"front": [round(gx0), round(gy0), round(gx1), round(gy1)], "bands": [round(b1), round(b2)]})


# ============================================================================================== the room side
F_ = 1800.0          # focal length of the sill's perspective (for the tile grid); the sill front is at depth 1


def _sill_tiles(s, m, cell=54.0, rowd=0.03):
    """Tile index fields over the sill top (VP perspective): per-tile random, and the grout lines."""
    Z = 560.0 / np.maximum(s.Y - VP[1], 1.0)
    u = (s.X - VP[0]) * Z / cell + 0.5
    v = (Z - 1.0) / rowd
    iu, iv = np.floor(u), np.floor(v)
    h = np.sin(iu * 12.9898 + iv * 78.233) * 43758.5453
    rnd = (h - np.floor(h)).astype(np.float32)
    du = np.minimum(u - iu, 1 - (u - iu)) * cell / Z            # screen px to the nearest grout line
    dv = np.minimum(v - iv, 1 - (v - iv)) * 560.0 * rowd / (Z * Z)
    gro = np.clip(1.6 - np.minimum(du, dv), 0, 1).astype(np.float32) * m
    return rnd * m, gro


def frame():
    s = Sheet(SW, SH, 204)
    r = rs(204)
    glass = s.zeros()
    for p in PANES:
        glass = np.maximum(glass, s.rect(*p))
    solid = 1 - glass
    # ---- plaster wall: pale, cool, darker away from the window, a damp tide mark under the right corner
    wall = solid * (1 - Q(s, [(OX0, OY0), (OX1, OY0), (OX1, OY1), (OX0, OY1)]))
    s.wash(solid, CREAM, 0.40, wob=0, soft=0, edge=0, flow=0.35, fs=420, gran=0.4, body=1)
    s.wash(wall, RAIN, 0.16, wob=0, soft=0, edge=0, flow=0.3, fs=360, gran=0.4)
    s.wash(wall, PAYNE, 0.30, wob=0, soft=0, edge=0, grad=s.rgrad(1440, 700, 1000, 2100, 0.0, 1, sy=0.7), flow=0.3, fs=500)
    s.wash(wall, SIENNA, 0.06, wob=0, soft=0, edge=0, grad=s.noise(300, 3))
    for _ in range(9):
        s.wash(s.blob(r.random() * SW, r.random() * SH, 200 + 200 * r.random(), 90 + 90 * r.random(), 0.3) * wall, SLATE, 0.05, wob=6, bleed=30, edge=0.7, ew=14)
    tide = s.blob(2560, 1330, 210, 150, 0.35, 10)
    s.wash(tide * wall, OCHRE, 0.22, wob=6, bleed=10, edge=1.0, ew=10, flow=0.5)
    s.wash(s.blob(2550, 1310, 120, 80, 0.35, 9) * wall, SEPIA, 0.10, wob=5, bleed=8, edge=1.0, ew=6)
    s.ink([(OX1, OY0 + 6), (OX1 + 40, 40), (OX1 + 70, 10), (OX1 + 130, -20)], 1.2, SEPIA, 0.8, tremor=1.6, skip=0.2, smooth=False)  # a hairline crack
    s.ink([(OX0 - 4, OY1 + 140), (OX0 - 60, OY1 + 210), (OX0 - 70, OY1 + 300)], 1.1, SEPIA, 0.6, tremor=1.6, skip=0.3, smooth=False)
    # ---- the reveals: the right one faces the light, the left and the soffit are in shade
    revL = Q(s, [(OX0, OY0), (FX0, FY0), (FX0, FY1), (OX0, OY1)])
    revR = Q(s, [(OX1, OY0), (FX1, FY0), (FX1, FY1), (OX1, OY1)])
    soff = Q(s, [(OX0, OY0), (OX1, OY0), (FX1, FY0), (FX0, FY0)])
    s.wash(revL, SLATE, 0.30, wob=1, soft=1, edge=0.4, flow=0.4)
    s.wash(revL, RAIN, 0.18, wob=1, soft=1, edge=0, grad=s.hgrad(OX0, FX0, 1, 0.3))
    s.wash(soff, SLATE, 0.40, wob=1, soft=1, edge=0.4, flow=0.4)
    s.wash(soff, PAYNE, 0.15, wob=1, soft=1, edge=0, grad=s.vgrad(OY0, FY0, 0.2, 1))
    _lift(s, revR * s.hgrad(OX1, FX1, 0.3, 1), 0.35, soft=4, wob=1)
    s.wash(revR, RAIN, 0.06, wob=1, soft=1, edge=0.5)
    s.wash(revR * s.vgrad(FY1 - 300, OY1, 0, 1), SLATE, 0.12, wob=2, bleed=10)
    # ---- the aluminium window: fixed frame, two sashes (the right one on the inner track), rubber, the crescent lock
    def alu(m, k=1.0, vert=False, lit=0.0):
        base(s, m, MIST, 0.55 * k, wob=0.7, soft=0.8, edge=0.6, ew=3, flow=0.4, fs=100, gran=0.3)
        s.wash(m, SLATE, 0.16 * k, wob=0.5, soft=0.8, edge=0.2, flow=0.5, fs=60)
        if lit:
            _lift(s, m, lit, soft=1, wob=0.5)
    fr = Q(s, [(FX0, FY0), (FX1, FY0), (FX1, FY1), (FX0, FY1)]) * (1 - Q(s, [(IX0, IY0), (IX1, IY0), (IX1, IY1), (IX0, IY1)]))
    alu(fr, 1.0)
    s.wash(fr * s.rect(FX0, IY1, FX1, FY1), SLATE, 0.30, wob=0.5, soft=1)                 # the bottom track
    for yy in (IY1 + 10, IY1 + 22):
        seg(s, (FX0 + 20, yy), (FX1 - 20, yy), 1.4, INKBLUE, 0.8, r=r, skip=0.2)
    _lift(s, s.rect(FX0, FY1 - 6, FX1, FY1 - 1), 0.4, soft=1)
    path(s, [(FX0, FY0), (FX1, FY0), (FX1, FY1), (FX0, FY1)], 2.2, INK, 1.3, closed=True, over=6, r=r)
    path(s, [(IX0, IY0), (IX1, IY0), (IX1, IY1), (IX0, IY1)], 1.8, INK, 1.1, closed=True, over=4, r=r)
    for (sx0, sx1, front) in ((IX0, MEET + 20, False), (MEET - 20, IX1, True)):
        p = PANES[1] if front else PANES[0]
        sm = s.rect(sx0, IY0, sx1, IY1) * (1 - s.rect(*p))
        alu(sm, 1.0 if front else 1.15)
        if not front:
            s.wash(sm, SLATE, 0.18, wob=0.5, soft=1)
        # inner edges catch the light from the glass
        _lift(s, s.rect(p[0] - 4, p[1], p[0] + 1, p[3]), 0.45, soft=1)
        _lift(s, s.rect(p[0], p[1] - 4, p[2], p[1] + 1), 0.35, soft=1)
        s.wash(s.rect(p[0], p[3], p[2], p[3] + 5), PAYNE, 0.5, soft=1, wob=0.5)
        path(s, [(sx0, IY0), (sx1, IY0), (sx1, IY1), (sx0, IY1)], 1.9, INK, 1.2, closed=True, over=3, r=r)
        # the black rubber round the glass
        path(s, [(p[0], p[1]), (p[2], p[1]), (p[2], p[3]), (p[0], p[3])], 3.0, INK, 1.4, closed=True, over=2, r=r)
        if front:
            seg(s, (sx0 + 12, IY0 + 30), (sx0 + 12, IY1 - 30), 1.4, INKBLUE, 0.8, r=r, skip=0.2)
            s.wash(s.rect(sx0, IY0, sx0 + 6, IY1), PAYNE, 0.5, wob=0.5, soft=1.5)
    # the crescent lock on the meeting stiles
    lk = (MEET - 4, 640)
    lb = s.rect(lk[0] - 14, lk[1] - 30, lk[0] + 14, lk[1] + 30)
    alu(lb, 1.3)
    s.wash(s.ellipse(lk[0] - 22, lk[1] - 4, 14, 22), SLATE, 0.9, wob=0.5, soft=0.8, body=1)
    path(s, [(lk[0] - 14, lk[1] - 30), (lk[0] + 14, lk[1] - 30), (lk[0] + 14, lk[1] + 30), (lk[0] - 14, lk[1] + 30)], 1.6, INK, 1.2, closed=True, over=2, r=r)
    s.ink(ell(lk[0] - 18, lk[1] - 4, 16, 22, 1.6, 4.7, 20), 1.6, INK, 1.2)
    seg(s, (lk[0] + 6, lk[1] + 6), (lk[0] + 34, lk[1] + 32), 4.5, SLATE, 1.6, bow=0)
    # ---- the sill: small cream tiles in perspective, the window's light lying on them
    top = Q(s, [(FX0, FY1), (FX1, FY1), (2538, SILL_FRONT - 5), (2580, SILL_FRONT - 5), (2584, SILL_FRONT), (296, SILL_FRONT), (300, SILL_FRONT - 5), (342, SILL_FRONT - 5)])
    rnd, gro = _sill_tiles(s, top)
    base(s, top, CREAM, 0.45, wob=0.8, soft=0.8, edge=0.4, flow=0.4, fs=160, gran=0.4)
    s.wash(top, TILE, 0.30, wob=0, soft=0, edge=0, grad=np.clip((rnd - 0.55) * 3, 0, 1), gran=0.3)          # a few greener tiles
    s.wash(top, SOY, 0.35, wob=0, soft=0, edge=0, grad=np.clip((0.35 - rnd) * 3, 0, 1), gran=0.3)
    s.wash(top, SLATE, 0.22, wob=0, soft=0, edge=0, grad=s.vgrad(FY1, SILL_FRONT, 0.0, 1.0) * (0.6 + 0.4 * s.noise(200, 3)), flow=0.4)
    s.wash(gro, UMBER, 0.30, wob=0.5, soft=0.7, edge=0, gran=0.3, flow=0.6, fs=60)
    _lift(s, Q(s, [(PANES[0][0], FY1 + 2), (PANES[1][2], FY1 + 2), (PANES[1][2] + 60, FY1 + 50), (PANES[0][0] - 60, FY1 + 50)]), 0.4, soft=12, wob=3)
    _lift(s, s.rgrad(LIGHT[0] + 150, FY1 + 30, 30, 520, 1, 0, sy=0.15) * top, 0.35, soft=0, wob=0)
    for x in np.arange(300, 2580, 54 * 4):           # the pen finds a few of the grout lines
        q0 = (x + r.random() * 30, SILL_FRONT)
        q1 = _toward(q0, (FY1 - VP[1]) / (SILL_FRONT - VP[1]))
        if r.random() < 0.5 and FX0 - 40 < q1[0] < FX1 + 40:
            seg(s, q0, (q0[0] + (q1[0] - q0[0]) * (0.4 + 0.5 * r.random()), q0[1] + (q1[1] - q0[1]) * (0.4 + 0.5 * r.random())), 1.2, SEPIA, 0.6, r=r, skip=0.35)
    # front edge: rounded edge tiles, then the wall under the ledge in its shadow
    fe = s.rect(296, SILL_FRONT, 2584, SILL_FRONT + 38)
    base(s, fe, CREAM, 0.5, wob=0.8, soft=0.8, edge=0.45, flow=0.4, gran=0.4)
    s.wash(fe, SLATE, 0.45, wob=0.5, soft=1, grad=s.vgrad(SILL_FRONT, SILL_FRONT + 38, 0.1, 1))
    _lift(s, s.rect(296, SILL_FRONT - 2, 2584, SILL_FRONT + 6), 0.5, soft=2)
    for x in np.arange(318, 2584, 54):
        seg(s, (x + r.random() * 3, SILL_FRONT + 6), (x + r.random() * 3, SILL_FRONT + 37), 1.2, UMBER, 0.6, bow=0, skip=0.2)
    s.wash(s.rect(296, SILL_FRONT + 38, 2584, SILL_FRONT + 110) * s.vgrad(SILL_FRONT + 38, SILL_FRONT + 110, 1, 0), SLATE, 0.30, wob=0, soft=4, edge=0)
    path(s, [(296, SILL_FRONT), (2584, SILL_FRONT), (2584, SILL_FRONT + 38), (296, SILL_FRONT + 38)], 2.2, INK, 1.3, closed=True, over=6, r=r)
    seg(s, (FX0, FY1), (342, SILL_FRONT - 5), 1.8, INK, 1.1, r=r, over=4)
    seg(s, (FX1, FY1), (2538, SILL_FRONT - 5), 1.8, INK, 1.1, r=r, over=4)
    # pen on the reveals and the opening
    for a, b in (((OX0, OY0), (FX0, FY0)), ((OX1, OY0), (FX1, FY0)), ((OX0, OY0), (OX1, OY0)), ((OX0, OY0), (OX0, OY1 - 4)), ((OX1, OY0), (OX1, OY1 - 4))):
        seg(s, a, b, 2.2, INK, 1.3, r=r, over=5)
    # ---- a glass of water on the sill: what is behind it, lighter and shifted, its rims, the water line
    wx, wy = WATER
    gw0, gw1, gh = 52, 44, 150
    gl = Q(s, [(wx - gw0, wy - gh), (wx + gw0, wy - gh), (wx + gw1, wy), (wx - gw1, wy)])
    cast(s, s.ellipse(wx, wy + 4, gw1 + 6, 12), 0, 14, SLATE, 0.30, blur=9)
    _lift(s, gl, 0.25, soft=1, wob=0.6)
    s.wash(gl, MIST, 0.18, wob=0.6, soft=0.8, edge=0.9, ew=3)
    water = Q(s, [(wx - gw0 + 3, wy - gh + 42), (wx + gw0 - 3, wy - gh + 42), (wx + gw1 - 4, wy - 10), (wx - gw1 + 4, wy - 10)])
    s.wash(water, RAIN, 0.16, wob=0.6, soft=1, edge=0.7, ew=3)
    s.wash(water * s.hgrad(wx - gw0, wx + gw0, 0.0, 1), SLATE, 0.18, wob=0.6, soft=2)
    _lift(s, s.ellipse(wx, wy - gh + 42, gw0 - 4, 7), 0.35, soft=1, wob=0.4)
    s.wash(s.ellipse(wx, wy - 8, gw1 - 2, 8), SLATE, 0.35, wob=0.4, soft=1, edge=0.7)
    _lift(s, s.ribbon([(wx - gw0 + 12, wy - gh + 14), (wx - gw1 + 12, wy - 22)], 7, (0.2, 0.3)), 0.8, soft=1)
    _lift(s, s.ribbon([(wx + gw0 - 16, wy - gh + 30), (wx + gw1 - 14, wy - 50)], 3, (0.3, 0.3)), 0.6, soft=0.8)
    oval(s, wx, wy - gh, gw0, 9, 1.5, SLATE, 1.1, twice=True)
    seg(s, (wx - gw0, wy - gh), (wx - gw1, wy), 1.5, SLATE, 1.0, bow=0.01)
    seg(s, (wx + gw0, wy - gh), (wx + gw1, wy), 1.5, SLATE, 1.0, bow=0.01)
    s.ink(ell(wx, wy - 2, gw1, 9, 0.1, math.pi - 0.1, 24), 1.5, SLATE, 1.0)
    s.ink(ell(wx, wy - gh + 42, gw0 - 3, 7, 0.0, math.pi, 24), 1.1, SLATE, 0.8)
    s.save("sl_frame", crop=True, anchor=(0, 0), meta={"panes": PANES, "sill_y": SILL_Y, "sill_back": SILL_BACK, "sill_front": SILL_FRONT, "light": list(LIGHT),
                                                     "pot": list(POT), "hand": list(HAND), "water": list(WATER), "vp": list(VP)})


# ============================================================================================== the pot
def pot():
    s = Sheet(560, 460, 205)
    r = rs(205)
    cx, by = 280, 400
    RT, RR = by - 236, 152                     # rim top centre y, rim radius
    # contact shadow, toward us (the light is behind the pot)
    s.wash(s.ellipse(cx + 6, by + 6, 176, 26), PAYNE, 0.40, wob=2, bleed=10, edge=0.1)
    # saucer
    sau = s.poly([(cx - 158, by - 30), (cx + 158, by - 30), (cx + 146, by - 4), (cx + 100, by + 4), (cx - 100, by + 4), (cx - 146, by - 4)])
    base(s, sau, TERRA, 0.85, wob=1, edge=0.7, ew=4, flow=0.5, fs=60, gran=0.6)
    s.wash(sau, BURNT, 0.35, wob=1, soft=2, grad=s.vgrad(by - 26, by + 4, 0.2, 1))
    s.wash(s.ellipse(cx, by - 30, 158, 20), BURNT, 0.5, wob=1, soft=1.2, edge=0.5)
    s.ink(ell(cx, by - 30, 158, 20, -0.1, math.pi + 0.1, 30), 1.8, SEPIA, 1.2)
    s.ink([(cx - 158, by - 30), (cx - 146, by - 4), (cx - 100, by + 4), (cx + 100, by + 4), (cx + 146, by - 4), (cx + 158, by - 31)], 2.0, SEPIA, 1.3)
    # body: tapered, its own little shadow side, a pale mineral bloom low down
    body = s.poly([(cx - 140, RT + 50), (cx + 140, RT + 50), (cx + 117, by - 70), (cx + 98, by - 34), (cx + 50, by - 26), (cx - 50, by - 26), (cx - 98, by - 34), (cx - 117, by - 70)])
    base(s, body, TERRA, 1.0, wob=1.2, edge=0.8, ew=5, flow=0.55, fs=70, gran=0.65)
    s.wash(body, BURNT, 0.45, wob=1.5, bleed=12, grad=s.hgrad(cx - 60, cx + 140, 0.15, 1), edge=0.2)
    s.wash(body, UMBER, 0.25, wob=1, bleed=10, grad=s.vgrad(RT + 50, RT + 100, 1, 0))              # under the rim
    _lift(s, body * s.hgrad(cx - 140, cx - 90, 1, 0), 0.30, soft=4, wob=1)                          # the window's light round the left edge
    bloom = body * s.vgrad(by - 110, by - 40, 0, 1) * s.noise(30, 3)
    _lift(s, bloom, 0.35, soft=2, wob=2, dry=0.35)
    s.wash(bloom, CREAM, 0.25, wob=2, soft=2, dry=0.3)
    s.ink([(cx - 140, RT + 50), (cx - 117, by - 70), (cx - 98, by - 34), (cx - 50, by - 27)], 2.2, SEPIA, 1.3)
    s.ink([(cx + 140, RT + 50), (cx + 117, by - 70), (cx + 98, by - 34), (cx + 50, by - 27)], 2.4, SEPIA, 1.4)
    s.ink([(cx - 70, by - 27), (cx, by - 24), (cx + 70, by - 27)], 1.6, SEPIA, 1.0, skip=0.3)
    # rim band
    rim = s.poly([(cx - RR, RT), (cx + RR, RT), (cx + RR - 4, RT + 50), (cx - RR + 4, RT + 50)], smooth=False)
    rim = np.maximum(rim, s.ellipse(cx, RT + 50, RR - 4, 12))
    base(s, rim, TERRA, 0.95, wob=1, edge=0.8, ew=4, flow=0.5, fs=50, gran=0.6)
    s.wash(rim, BURNT, 0.4, wob=1.2, bleed=10, grad=s.hgrad(cx - 40, cx + RR, 0.1, 1))
    s.wash(s.ellipse(cx, RT + 54, RR - 4, 12) * (1 - s.ellipse(cx, RT + 48, RR, 12)), UMBER, 0.6, wob=1, soft=1.2)
    _lift(s, rim * s.hgrad(cx - RR, cx - RR + 40, 1, 0), 0.28, soft=3)
    # the rim's top and the inside: the back inner wall in shade, dark damp soil
    ring = s.ellipse(cx, RT, RR, 30)
    base(s, ring, TERRA, 0.75, wob=0.8, soft=0.8, edge=0.5, gran=0.5)
    _lift(s, ring * s.vgrad(RT - 30, RT + 10, 0.4, 1), 0.35, soft=1.5)
    inner = s.ellipse(cx, RT + 1, RR - 13, 23)
    s.wash(inner, UMBER, 0.85, wob=0.6, soft=0.8, edge=0.5, ew=3, body=1)
    soil = s.ellipse(cx, RT + 8, RR - 17, 17) * inner
    s.wash(soil, SEPIA, 1.25, wob=0.8, soft=0.8, edge=0.5, ew=3, gran=0.7, flow=0.5, fs=20)
    s.spatter(18, UMBER, 0.6, box=(cx - 110, RT - 4, cx + 110, RT + 20), r=(1.2, 3.2))
    _lift(s, s.rgrad(cx - 30, RT + 4, 2, 60, 1, 0, sy=0.2) * soil, 0.15, soft=0, wob=0)
    for _ in range(5):                                # bits of perlite in the soil
        x, y = cx + (r.random() - 0.5) * 200, RT + 4 + (r.random() - 0.5) * 18
        _lift(s, s.ellipse(x, y, 2.4, 1.8), 0.7, soft=0.5, wob=0.2)
    oval(s, cx, RT, RR, 30, 2.4, SEPIA, 1.4, twice=True)
    # the chip: a bite out of the front of the rim, raw paler clay in the break, the rim line broken there
    ca_ = 2.15
    ex, ey = cx + RR * math.cos(ca_), RT + 30 * math.sin(ca_)
    chip = s.poly([(ex - 26, ey - 5), (ex + 22, ey + 3), (ex + 8, ey + 20), (ex - 6, ey + 24), (ex - 20, ey + 12)])
    s.lift(chip, 1.0, soft=0.5, wob=0.4)
    s.wash(chip, TERRA, 0.42, wob=0.4, soft=0.5, edge=0.8, ew=1.5, gran=0.7, body=1)
    s.wash(s.poly([(ex - 20, ey + 12), (ex - 6, ey + 24), (ex + 8, ey + 20), (ex - 2, ey + 14)]), BURNT, 0.45, wob=0.4, soft=0.6)
    s.ink([(ex - 27, ey - 4), (ex - 20, ey + 12), (ex - 6, ey + 25), (ex + 8, ey + 21), (ex + 23, ey + 3)], 1.4, SEPIA, 1.1, smooth=False, tremor=0.5)
    s.ink(ell(cx, RT + 1, RR - 13, 23, math.pi + 0.15, 2 * math.pi - 0.15, 30), 1.5, SEPIA, 0.9)
    s.ink([(cx - RR, RT), (cx - RR + 4, RT + 50)], 2.2, SEPIA, 1.3)
    s.ink([(cx + RR, RT), (cx + RR - 4, RT + 50)], 2.2, SEPIA, 1.3)
    s.ink(ell(cx, RT + 50, RR - 4, 12, 0.15, math.pi - 0.15, 30), 1.8, SEPIA, 1.1, skip=0.15)
    s.save("sl_pot", crop=True, anchor=(cx, by), meta={"soil": [0, RT + 4 - by], "rim": [RR, RT - by]})


# ============================================================================================== the mint
def _sub(s, x0, y0, x1, y1):
    """A small working sheet over a region of a big one (same paper), so small things paint fast."""
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(s.w, int(x1)), min(s.h, int(y1))
    t = Sheet(x1 - x0, y1 - y0, s._s() & 0xFFFF)
    t.A = s.A[y0:y1, x0:x1].copy()
    t.body = s.body[y0:y1, x0:x1].copy()
    t.P, t.G = s.P[y0:y1, x0:x1], s.G[y0:y1, x0:x1]
    return t, x0, y0


def _put(s, t, x0, y0):
    s.A[y0:y0 + t.h, x0:x0 + t.w] = t.A
    s.body[y0:y0 + t.h, x0:x0 + t.w] = t.body


def _opaque(s, m, color, k, **kw):
    """An opaque thing over earlier paint: clear inside a slightly shrunk mask (no pale halo), lay the wash, mark body."""
    inner = cv2.erode(m, np.ones((3, 3), np.uint8))
    s.A *= (1 - cv2.GaussianBlur(inner, (0, 0), 0.6))[..., None]
    return s.wash(m, color, k, body=1, **kw)


def _leafpoly(x, y, ang, L, W, fold=1.0, asym=0.0, bend=0.0, teeth=7, seed=0):
    """A mint leaf: ovate, pointed, toothed toward the tip; `ang` is its direction (radians, screen), fold squashes it
    across (a leaf turned away), asym makes one half narrower, bend curls the midrib. Returns outline, midrib, the
    two half outlines and a mapper from leaf coords (t along, q across -1..1) to the sheet."""
    rr = rs(seed)
    t = np.linspace(0, 1, 70)
    u = t ** 0.78
    h = 0.5 * W * np.sin(np.pi * u) ** 0.85 * (1 - 0.18 * t)
    ph = (t * teeth + rr.random()) % 1.0
    tooth = W * 0.075 * ph ** 1.7 * sstep(0.08, 0.22, t) * (1 - sstep(0.86, 0.98, t))
    hh = h + tooth
    ca, sa = math.cos(ang), math.sin(ang)

    def mp(tt, q):
        tt = np.asarray(tt, float)
        q = np.asarray(q, float)
        hx = np.interp(tt, t, h)
        side = np.where(q < 0, fold * (1 + asym), fold * (1 - asym))
        lx = tt * L
        ly = bend * L * tt * tt + q * hx * side
        return np.stack([x + lx * ca - ly * sa, y + lx * sa + ly * ca], -1)

    def edge(sgn, hv):
        lx = t * L
        side = fold * (1 + asym) if sgn < 0 else fold * (1 - asym)
        ly = bend * L * t * t + sgn * hv * side
        return np.stack([x + lx * ca - ly * sa, y + lx * sa + ly * ca], 1)
    left, right = edge(-1, hh), edge(1, hh)
    mid = mp(t, np.zeros_like(t))
    return np.vstack([left, right[::-1]]), mid, left, right, mp


def _paint_leaf(S, r, x, y, ang, L, W, lit, fold=1.0, asym=0.0, bend=0.0, young=False, drops=0, catch=False):
    R = int(L * 1.25 + 30)
    s, ox, oy = _sub(S, x - R, y - R, x + R, y + R)
    x, y = x - ox, y - oy
    out, mid, left, right, mp = _leafpoly(x, y, ang, L, W, fold, asym, bend, 7 if L > 50 else 5, int(r.integers(1 << 30)))
    m = s.poly(out, smooth=False)
    # a soft shadow onto the leaves already painted behind it (never onto nothing)
    behind = np.clip((1 - np.exp(-s.A.max(2))) * 3, 0, 1)
    sh = cv2.GaussianBlur(cv2.warpAffine(m, np.float32([[1, 0, 3], [0, 1, 8]]), (s.w, s.h)), (0, 0), 5) * behind
    s.wash(sh, INKBLUE, 0.35, wob=0, soft=0, edge=0, gran=0.3)
    if young:
        _opaque(s, m, MINT, 0.85, wob=0.6, soft=0.7, edge=0.65, ew=2.5, flow=0.4, fs=30, gran=0.3)
        s.wash(m, LEAF, 0.22, wob=0.6, soft=1, grad=s.rgrad(x, y, 2, L, 1, 0))
    else:
        c0 = MINT if lit > 0.5 else LEAF
        _opaque(s, m, c0, (1.0 + 0.25 * r.random()) if lit > 0.5 else (0.8 + 0.2 * r.random()), wob=0.8, soft=0.7, edge=0.65, ew=3, flow=0.5, fs=40, gran=0.4)
        if lit <= 0.5:
            s.wash(m, MINT, 0.3, wob=0.6, soft=1, edge=0, grad=s.rgrad(*mp(0.7, 0), 3, L * 0.6, 1, 0))
        s.wash(m, LEAF if lit > 0.5 else INKBLUE, 0.22 + 0.2 * (1 - lit), wob=0.6, soft=1, edge=0, grad=s.rgrad(x, y, 3, L * 0.9, 1, 0.1))
    # the far half of the leaf turns from the light
    half = s.poly(np.vstack([mid, right[::-1]]), smooth=False) if (ang % (2 * math.pi)) < math.pi else s.poly(np.vstack([mid, left[::-1]]), smooth=False)
    s.wash(half, LEAF, 0.20 + 0.22 * (1 - lit), wob=0.5, soft=1.0, edge=0.3, ew=2, clip=m)
    # quilted: soft lifts between the veins
    nv = 4 if L > 60 else 3
    for j in range(nv):
        t0 = 0.18 + j * 0.62 / nv
        for q in (-0.55, 0.55):
            c = mp(t0 + 0.08, q)
            _lift(s, s.ellipse(c[0], c[1], max(2, L * 0.06), max(1.5, W * 0.10 * fold)), (0.16 if lit > 0.5 else 0.09), clip=m, soft=1.6, wob=0.6)
    if catch:          # the window's light along the upper half
        ln = mp(np.linspace(0.2, 0.7, 8), np.full(8, -0.38))
        _lift(s, s.ribbon(ln, max(4, W * 0.2 * fold), (0.4, 0.4)), 0.40, clip=cv2.erode(m, np.ones((5, 5), np.uint8)), soft=2.0, wob=0.6)
    # veins and pen
    mm = cv2.GaussianBlur(cv2.erode(m, np.ones((3, 3), np.uint8)), (0, 0), 1)
    vc = INKBLUE if lit < 0.5 else LEAF
    s.ink(mid[2:-6], 1.1 if L > 50 else 0.8, vc, 0.75, taper=(0.05, 0.6), tremor=0.3, clip=mm)
    if L > 34:
        for j in range(nv):
            t0 = 0.14 + j * 0.62 / nv
            for q in (-1, 1):
                pts = mp([t0, t0 + 0.10, t0 + 0.2], [0, q * 0.55, q * 0.86])
                s.ink(pts, 0.8, vc, 0.55, taper=(0.1, 0.5), tremor=0.3, skip=0.15, clip=mm)
    s.ink(out[::2], 1.3 if not young else 1.0, INKBLUE, 0.85 if not young else 0.65, closed=True, smooth=False, tremor=0.4, skip=0.12)
    if drops:
        for _ in range(drops):
            c = mp(0.3 + 0.5 * r.random(), (r.random() - 0.5) * 1.2)
            rad = 2.5 + 3.5 * r.random()
            s.wash(s.ellipse(c[0] + 1.5, c[1] + rad * 0.8, rad * 0.9, rad * 0.45), INKBLUE, 0.18, wob=0.2, soft=1.2)     # its little shadow
            dm = s.ellipse(c[0], c[1], rad, rad * 0.9)
            _lift(s, dm, 0.25, soft=0.6, wob=0.2)
            s.wash(dm, LEAF, 0.22, wob=0.2, soft=0.5, edge=1.0, ew=1.2)
            _lift(s, s.ellipse(c[0] - rad * 0.35, c[1] - rad * 0.35, rad * 0.32, rad * 0.28), 0.95, soft=0.4, wob=0.1)
    _put(S, s, ox, oy)


def _plant(name, seed, nst, hmin, hmax, spread, leaf_k=1.0, drops=0, W=1300, H=1100):
    S = Sheet(W, H, seed)
    r = rs(seed)
    ax, ay = W / 2, H - 70
    stems = []
    for i in range(nst):
        u = (i + 0.5) / nst
        a0 = (u - 0.5) * spread + (r.random() - 0.5) * 0.35
        bx, by = ax + (u - 0.5) * 140 + (r.random() - 0.5) * 40, ay + (r.random() - 0.65) * 26
        Lt = hmin + (hmax - hmin) * (r.random() ** 0.7) * (1 - 0.4 * abs(u - 0.5) * 2)
        stems.append((by + r.random() * 6, a0, bx, Lt, i))
    stems.sort(key=lambda z: z[0])                    # back ones first
    tips = []
    for by, a0, bx, Lt, i in stems:
        curl = (0.2 + 0.45 * r.random()) * (1 if a0 >= 0 else -1) * min(1, abs(a0) * 2.5 + 0.2)
        n = 60
        ss = np.linspace(0, 1, n)
        ang = a0 + curl * ss ** 1.5 - 0.12 * a0 * np.sin(ss * math.pi) + (r.random() - 0.5) * 0.15 * np.sin(ss * 5 + i)
        dx, dy = np.sin(ang), -np.cos(ang)
        P = np.vstack([[bx, by], np.stack([bx + np.cumsum(dx) * Lt / n, by + np.cumsum(dy) * Lt / n], 1)])
        # stem: square, green, purple toward the base
        S.stroke(P[::4], 8, LEAF, 0.95, taper=(0.0, 0.7), wob=0.6, edge=0.5, profile=lambda t: 1 - 0.45 * t)
        S.stroke(P[: n // 3: 3], 8, VIOLET, 0.35, taper=(0.0, 0.5), wob=0.6, edge=0.3)
        S.ink(P[::3] + np.array([-3.0, 0]), 1.2, INKBLUE, 0.8, taper=(0.05, 0.5), skip=0.2)
        # nodes: opposite pairs crossing at right angles, smaller and paler toward the tip
        nn = max(3, int(Lt / 64))
        for j in range(nn):
            f = 0.12 + 0.78 * (j / max(1, nn - 1)) ** 0.85
            k = min(n - 1, int(f * n))
            px, py = P[k + 1]
            th = ang[k]
            size = (1.05 - 0.6 * f) * (0.85 + 0.3 * r.random())
            Lf, Wf = 112 * size * leaf_k, 76 * size * leaf_k
            lit = float(np.clip(0.15 + 0.85 * f + (r.random() - 0.5) * 0.5 - 0.15 * (bx - ax) / 150, 0, 1))
            side = (j + i) % 2 == 0
            for sg in (-1, 1):
                if side:
                    dirn = th + sg * (1.15 + 0.35 * r.random() + 0.35 * (1 - f))       # lower ones droop
                    fold = 0.7 + 0.3 * r.random()
                    L2 = Lf
                else:
                    dirn = th + sg * (0.45 + 0.3 * r.random())
                    fold, L2 = 1.0, Lf * (0.55 + 0.2 * r.random())
                d = dirn - math.pi / 2
                pet = 6 + 10 * size
                lx0, ly0 = px + math.cos(d) * pet, py + math.sin(d) * pet
                S.ink([(px, py), (lx0, ly0)], 3.0, LEAF, 0.9, taper=(0, 0))
                droop = (0.10 + 0.15 * (1 - f)) * (1 if math.cos(d) > 0 else -1)
                _paint_leaf(S, r, lx0, ly0, d, L2, Wf, lit, fold, (r.random() - 0.5) * 0.4, droop + (r.random() - 0.5) * 0.1,
                            drops=drops if r.random() < 0.35 else 0, catch=lit > 0.7 and r.random() < 0.4)
        # the tip: two young leaves still pressed together, standing up, and the bud between them
        tx, ty = P[-1]
        th = ang[-1]
        for sg in (-1, 1):
            d = th + sg * 0.32 - math.pi / 2
            _paint_leaf(S, r, tx, ty, d, 40 * leaf_k, 24 * leaf_k, 0.9, 0.8, 0.2 * sg, -0.05 * sg, young=True)
        bud = S.ellipse(tx + math.sin(th) * 12, ty - math.cos(th) * 12, 5, 10, math.degrees(th))
        _opaque(S, bud, MINT, 0.75, wob=0.4, soft=0.6, edge=0.6)
        S.ink(ell(tx + math.sin(th) * 12, ty - math.cos(th) * 12, 5, 10, 0, 2 * math.pi, 20, th), 1.0, INKBLUE, 0.6)
        tips.append([round(float(tx - ax), 1), round(float(ty - ay), 1), round(math.degrees(th), 1)])
    S.save(name, crop=True, anchor=(ax, ay), meta={"tips": tips})


def mint():
    _plant("sl_mint", 206, 10, 280, 500, 1.6)


def mint_big():
    _plant("sl_mint_big", 207, 15, 420, 700, 1.55, leaf_k=1.1, drops=2, W=1600, H=1250)


def leaf():
    s = Sheet(200, 260, 208)
    r = rs(208)
    x, y = 100, 236
    s.stroke([(x, y), (x + 1, y - 24)], 5, LEAF, 0.8, taper=(0, 0.3), wob=0.3)
    # a young leaf: still half folded along the midrib, pale
    _paint_leaf(s, r, x + 1, y - 22, -math.pi / 2, 80, 52, 0.95, 0.72, 0.15, 0.06, young=True, catch=True)
    s.save("sl_leaf", crop=True, anchor=(x, y), meta={"len": 102})


# ============================================================================================== my hand
def hand():
    """Lying on the tiles, fingers toward the pot (left), seen at the sill's shallow angle: across the hand (which runs
    into the depth) everything is foreshortened. The forearm comes toward us from the lower right over the sill's edge
    and widens fast (it is close to the eye)."""
    s = Sheet(1500, 1000, 209)
    r = rs(209)
    wx, wy = 420, 300
    th = math.radians(-100)                   # local -y (the fingers) -> screen left, a little away
    c, sn = math.cos(th), math.sin(th)
    FS = 0.52                                 # foreshortening of the sill plane

    def T(p):
        x, y = p
        X, Y = x * c - y * sn, x * sn + y * c
        return (wx + X, wy + Y * FS)

    def TP(ps):
        return [T(p) for p in ps]

    def fing(ps, w, tip=0.74):
        return s.ribbon(TP(ps), w, profile=lambda t: 1.0 - (1 - tip) * t)
    back = s.poly(TP([(-82, -6), (-98, -80), (-90, -168), (-44, -184), (20, -186), (70, -172), (90, -100), (78, -6)]))
    # little finger (far) to index (near): relaxed, a little apart
    fing_pts = [([(62, -166), (78, -236), (88, -284)], 34), ([(24, -184), (30, -282), (32, -344)], 40),
                ([(-16, -188), (-22, -294), (-24, -368)], 42), ([(-58, -174), (-72, -264), (-80, -334)], 40)]
    fing_m = [fing(p, w) for p, w in fing_pts]
    thumb = fing([(-78, -40), (-112, -104), (-124, -160), (-126, -196)], 44, tip=0.72)
    hm = np.maximum(back, thumb)
    for f in fing_m:
        hm = np.maximum(hm, f)
    # cuff and sleeve
    c0, c1 = T((-120, 96)), T((122, 96))           # near (lower) and far (upper) corners of the cuff
    sleeve = s.poly([c1, (wx + 520, wy + 110), (wx + 860, wy + 380), (wx + 1080, 1010), (wx + 360, 1010), (wx + 200, wy + 300), c0])
    cuff = s.poly(TP([(-114, -6), (110, -6), (124, 100), (-122, 100)]))
    arm = np.maximum(np.maximum(hm, cuff), sleeve)
    sh = cv2.GaussianBlur(cv2.warpAffine(np.maximum(hm, cuff), np.float32([[1, 0, -4], [0, 1, 14]]), (s.w, s.h)), (0, 0), 9)
    s.wash(sh * (1 - arm), PAYNE, 0.45, wob=0, soft=0, edge=0, gran=0.4)
    base(s, sleeve, CREAM, 0.62, wob=1.5, edge=0.45, flow=0.5, fs=140, gran=0.5)
    s.wash(sleeve, OCHRE, 0.30, wob=2, bleed=18, grad=s.hgrad(wx + 640, wx + 260, 0.05, 1))           # the underside of the arm
    s.wash(sleeve, UMBER, 0.16, wob=2, bleed=18, grad=s.vgrad(wy + 300, 1000, 0, 1))
    _lift(s, sleeve * s.hgrad(wx + 500, wx + 900, 0, 1) * s.vgrad(wy + 500, wy + 100, 0, 1), 0.2, soft=10, wob=2)
    # the knit bunches up behind the cuff: a few soft folds, ribs along the arm, a dry texture of stitches
    for j in range(4):
        y0 = wy + 30 + j * 70 + r.random() * 20
        a = (c0[0] + 60 + j * 70, y0 + 120 + j * 30)
        b = (wx + 470 + j * 110, y0 + 40)
        s.stroke([a, ((a[0] + b[0]) / 2 + 10, (a[1] + b[1]) / 2 - 30), b], 26 + 10 * j, OCHRE, 0.20, wob=2, bleed=8, clip=sleeve, taper=(0.4, 0.4), edge=0.2)
        s.ink([(a[0] + 8, a[1] - 6), ((a[0] + b[0]) / 2 + 14, (a[1] + b[1]) / 2 - 36), (b[0] - 10, b[1] - 4)], 1.3, SEPIA, 0.55, skip=0.35, clip=sleeve, tremor=0.9)
    for i in range(8):
        q = (i + 0.5) / 8
        a = (c0[0] + (c1[0] - c0[0]) * q + 20, c0[1] + (c1[1] - c0[1]) * q)
        b = (a[0] + 420 + 160 * q, 1010)
        mid_ = ((a[0] + b[0]) / 2 + 40, (a[1] + b[1]) / 2 - 10)
        if i % 3 == 1:
            s.ink([a, mid_, b], 1.2, SEPIA, 0.25, skip=0.6, clip=sleeve, tremor=1.2)
    s.wash(sleeve, OCHRE, 0.12, wob=1, soft=1, edge=0.1, dry=0.7, gran=0.6)
    contour(s, sleeve, 2.2, SEPIA, 1.2)
    base(s, cuff, CREAM, 0.6, wob=1.2, edge=0.55, flow=0.4)
    s.wash(cuff, OCHRE, 0.36, wob=1.5, bleed=8, grad=s.vgrad(wy - 40, wy + 50, 0, 1))
    for i in range(15):
        q = -110 + i * 15.6
        s.ink(TP([(q, -2), (q * 1.06, 96)]), 1.3, SEPIA, 0.6, clip=cuff, tremor=0.5)
    contour(s, cuff, 2.0, SEPIA, 1.3)
    # the hand: fingers far to near (each lies over the one behind), the back of the hand, the thumb nearest
    for k, f in enumerate(fing_m):
        _opaque(s, f, SKIN, 0.62, wob=1.0, soft=0.8, edge=0.45, flow=0.3)
        s.wash(f, BURNT, 0.22 + 0.04 * k, wob=1.5, bleed=6, grad=s.vgrad(wy - 70, wy + 40, 0.1, 1))
        s.wash(f, ROSE, 0.14, wob=1.5, bleed=6)
        _lift(s, f * s.vgrad(wy - 80, wy - 20, 1, 0), 0.2, soft=3, wob=1)
        contour(s, f, 1.6, SEPIA, 1.1)
    _opaque(s, back, SKIN, 0.62, wob=1.2, soft=0.8, edge=0.45, flow=0.3)
    s.wash(back, BURNT, 0.28, wob=2, bleed=10, grad=s.vgrad(wy - 50, wy + 70, 0.05, 1), edge=0.15)
    _lift(s, back * s.vgrad(wy - 70, wy - 10, 1, 0), 0.25, soft=4, wob=2)
    _opaque(s, thumb, SKIN, 0.62, wob=1.0, soft=0.8, edge=0.45, flow=0.3)
    s.wash(thumb, BURNT, 0.34, wob=1.5, bleed=8, grad=s.vgrad(wy - 20, wy + 90, 0.1, 1))
    contour(s, np.maximum(back, thumb), 2.0, SEPIA, 1.3)
    cnt = np.clip(thumb - cv2.dilate(back, np.ones((3, 3), np.uint8)), 0, 1)
    contour(s, thumb * (1 - back * 0), 1.4, SEPIA, 0.8)
    cs_, _ = cv2.findContours((back > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for (x, y) in ((-58, -176), (-18, -190), (22, -186), (62, -168)):        # knuckles
        s.ink(TP([(x - 15, y + 8), (x, y - 6), (x + 15, y + 8)]), 1.3, BURNT, 0.8)
    for (x, y) in ((-72, -258), (-24, -290), (30, -276), (78, -232)):        # middle joints
        s.ink(TP([(x - 11, y + 4), (x, y - 3), (x + 11, y + 4)]), 1.1, BURNT, 0.65)
    for (x, y) in ((-79, -322), (-24, -354), (32, -332), (87, -274)):        # nails
        nm = s.ellipse(*T((x, y)), 13, 7, math.degrees(th) + 90)
        _lift(s, nm, 0.45, soft=1)
        s.wash(nm, ROSE, 0.15, wob=0.4, soft=1, edge=0.7)
    for (a, b) in (((-40, -50), (-56, -160)), ((0, -50), (-16, -172)), ((36, -50), (22, -168))):     # tendons, faint
        s.ink(TP([a, ((a[0] + b[0]) / 2 + 3, (a[1] + b[1]) / 2), b]), 1.0, BURNT, 0.35, skip=0.4)
    tip = T((-24, -368))
    s.save("sl_hand", crop=True, anchor=(wx, wy), meta={"tip": [round(tip[0] - wx, 1), round(tip[1] - wy, 1)]})


ALL = {"sl_out": outside, "sl_grille": grille, "sl_frame": frame, "sl_pot": pot, "sl_mint": mint, "sl_mint_big": mint_big, "sl_leaf": leaf, "sl_hand": hand}
