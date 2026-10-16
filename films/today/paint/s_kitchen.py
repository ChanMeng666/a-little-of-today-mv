"""The small old kitchen, wide, at eye level, in the morning: nobody in it. Sheet 3072x1728.

Everything is laid out in centimetres and drawn in one-point perspective (pj below):
  X  cm from the left edge of the back wall (the wall plane is 5 px/cm, x = 5 X)
  h  cm above the floor (the wall meets the floor at y = 1300)
  d  cm out from the back wall toward the camera, which stands 500 cm from the wall; vanishing point (1500, 525)
  screen: f = 500 / (500 - d);  x = 1500 + (5 X - 1500) f;  y = 525 + (1300 - 5 h - 525) f
Left to right along the back wall: the window over the sink (an outdoor iron grille, the neighbour's tiled wall and
the morning sky beyond), the counter with the sink, a dish rack, the two-burner gas stove and its kettle under the
hood, the rice cooker, open shelves; then the cream fridge with its tiny yellow note; a tear-off calendar; in the right
foreground a corner of the dining table (checked cloth) with two chairs. Terracotta floor tiles, 30 cm squares.
The floor and wall are evenly lit: the page draws the sun patch.

Layer kt_occ (RGBA, same sheet, placed by its ox/oy): the same painting cut out to the things that stand in the room
  (window, hood, shelves, calendar, fridge, counter, stove, kettle, cooker, table, chairs). Draw kt_bg, then the moving
  sun patch (on the floor and wall quads), then kt_occ on top, and the sun falls behind the furniture.
Layer kt_bg (opaque 3072x1728). meta:
  window       [x0, y0, x1, y1]  the glass opening
  kettle_spout [x, y]            where the steam leaves the kettle
  fridge_note  [x, y]            centre of the tiny yellow note on the fridge door (about 48 px square)
  floor        four corners [[x, y] x4] of the floor quad: the wall line (d = 0) from X = 0 to 614, and the same
               X at d = 178 (the bottom of the frame): a true floor rectangle in perspective
  floor_open   the part of that floor in front of the counter and fridge (d = 62..178): a sun patch kept in here
               never runs under the cabinets
  wall         four corners of the back-wall quad (frontal, so a rectangle) from the floor to the ceiling line
  persp        {"vp": [1500, 525], "px_per_cm": 5, "dist_cm": 500, "floor_y": 1300} (the formula above)
  burner       [x, y] the left burner under the kettle (a flame could flicker there)
  table        [x, y] the near-left corner of the table top
"""
import math

import cv2
import numpy as np

from kit import *
from s_fridge import tile_wall, dense, rr, dome

SW, SH = 3072, 1728
VPX, VPY = 1500.0, 525.0
PX, DIST, FLOOR = 5.0, 500.0, 1300.0
EYE = (FLOOR - VPY) / PX            # 155 cm
FONT_P = "D:/github_repository/a-little-of-today-mv/assets/fonts/NotoSerifSC-VF.ttf"
FONT_H = "D:/github_repository/a-little-of-today-mv/assets/fonts/LXGWWenKai-Regular.ttf"

WIN = (40, 196, 112, 205)           # window opening X0, X1, h0, h1 (cm)
CTR = (-10, 392, 85, 60)            # counter X0, X1, height, depth
FRG = (426, 491, 160, 64)           # fridge X0, X1, height, depth
STOVE = (246, 318, 8, 54)           # stove X0, X1, d0, d1
KETTLE = (266, 31)                  # kettle centre X, d
TABLE = (494, 680, 78, 176, 75)     # table X0, X1, d0, d1, height


def pj(X, h, d=0.0):
    f = DIST / (DIST - d)
    return (VPX + (X * PX - VPX) * f, VPY + (FLOOR - h * PX - VPY) * f)


def quad(s, pts):
    return Q(s, [pj(*p) for p in pts])


def boxf(X0, X1, h0, h1, d0, d1):
    """The faces of a box that face the camera: front, top (if below the eye), the side toward the centre."""
    F = {"front": [(X0, h0, d1), (X1, h0, d1), (X1, h1, d1), (X0, h1, d1)]}
    if h1 < EYE:
        F["top"] = [(X0, h1, d0), (X1, h1, d0), (X1, h1, d1), (X0, h1, d1)]
    if X0 * PX > VPX:
        F["side"] = [(X0, h0, d0), (X0, h0, d1), (X0, h1, d1), (X0, h1, d0)]
    elif X1 * PX < VPX:
        F["side"] = [(X1, h0, d0), (X1, h0, d1), (X1, h1, d1), (X1, h1, d0)]
    return F


def edges(s, pts, w=2.0, k=1.3, r=None, over=3, col=INK, closed=True, **kw):
    path(s, [pj(*p) for p in pts], w, col, k, closed=closed, over=over, r=r, **kw)


def kitchen():
    s = Sheet(SW, SH, 301)
    r = rs(301)
    wall = s.rect(-10, -10, SW + 10, FLOOR)
    # ---- the back wall: plaster above the tiles, the tiles, the ceiling line
    TOPT = pj(0, 190)[1]
    plaster = s.rect(-10, -10, SW + 10, TOPT)
    s.wash(plaster, CREAM, 0.50, wob=0, edge=0, flow=0.45, fs=420, gran=0.4)
    s.wash(plaster, OCHRE, 0.16, wob=0, edge=0, grad=s.noise(300, 3), flow=0.4)
    for _ in range(6):
        s.wash(s.blob(r.random() * SW, 60 + r.random() * 260, 200 + 220 * r.random(), 60 + 50 * r.random(), 0.3) * plaster, [OCHRE, SIENNA, MIST][int(r.random() * 3)], 0.10, wob=6, bleed=26, edge=0.6, ew=12)
    ceil = s.rect(-10, -10, SW + 10, pj(0, 250)[1])
    s.wash(ceil, MIST, 0.55, wob=1.5, edge=0.3, flow=0.4)
    s.wash(ceil, UMBER, 0.15, wob=0, edge=0, grad=s.vgrad(0, 50, 1, 0))
    seg(s, (-10, pj(0, 250)[1]), (SW + 10, pj(0, 250)[1] + 2), 2.2, INK, 1.2, r=r)
    # a damp stain in the corner of the plaster, an old one
    s.wash(s.blob(2840, 170, 160, 70, 0.4), SIENNA, 0.14, wob=5, bleed=20, edge=0.9, ew=6)
    tile_wall(s, r, -10, TOPT, SW + 10, FLOOR, 60, ox=0, oy=TOPT, k=0.85, grout=0.42)
    trim = s.rect(-10, TOPT - 16, SW + 10, TOPT + 4)                # a row of edging tiles along the top
    base(s, trim, CELADON, 0.85, wob=0.8, edge=0.5)
    s.wash(trim, TEAL, 0.25, wob=0.8, soft=2, grad=s.vgrad(TOPT - 16, TOPT + 4))
    seg(s, (-10, TOPT - 16), (SW + 10, TOPT - 15), 1.8, INK, 1.1, r=r)
    seg(s, (-10, TOPT + 4), (SW + 10, TOPT + 5), 1.6, SLATE, 0.9, r=r)
    # a band of darker celadon tiles at counter-splash height, as old kitchens have
    by = pj(0, 140)[1]
    band = s.rect(-10, by - 30, SW + 10, by + 30)
    s.wash(band, TEAL, 0.32, wob=1, edge=0.6, gran=0.5, flow=0.4)
    # a fluorescent tube in its tin holder on the wall above the window, and its cable
    tx0, ty = pj(52, 222)
    tx1, _ = pj(182, 222)
    hol = s.rect(tx0, ty - 10, tx1, ty + 14)
    cast(s, hol, 6, 14, UMBER, 0.35, blur=8)
    base(s, hol, CREAM, 0.4, wob=0.5, edge=0.4)
    s.wash(hol, SLATE, 0.35, wob=0.4, soft=2, grad=s.vgrad(ty - 10, ty + 14))
    tube = s.rect(tx0 + 14, ty + 14, tx1 - 14, ty + 34)
    base(s, tube, CREAM, 0.08, wob=0.4, edge=0.3)
    s.wash(tube, MIST, 0.25, wob=0.3, soft=1.5, grad=s.vgrad(ty + 14, ty + 34))
    path(s, [(tx0, ty - 10), (tx1, ty - 10), (tx1, ty + 14), (tx0, ty + 14)], 1.6, INK, 1.2, closed=True, over=3, r=r)
    path(s, [(tx0 + 14, ty + 14), (tx1 - 14, ty + 14), (tx1 - 14, ty + 34), (tx0 + 14, ty + 34)], 1.3, INK, 1.0, closed=True, over=2, r=r)
    s.ink([(tx1, ty), (tx1 + 60, ty + 4), (tx1 + 80, ty - 60), (tx1 + 84, -10)], 1.8, INK, 1.0)
    # ---- the floor: terracotta squares
    floor(s, r)
    s.body[:] = 0              # from here on, everything painted with a body is a thing in the room (see kt_occ)
    window(s, r)
    shelves(s, r)
    hood(s, r)
    calendar(s, r)
    fridge(s, r)
    counter(s, r)
    stove(s, r)
    cooker(s, r)
    table(s, r)
    # the whole room: a breath of warm light from the window, a little darker toward the right and the floor
    s.wash(s.full(), OCHRE, 0.10, wob=0, soft=0, edge=0, gran=0.2, flow=0.3, grad=s.rgrad(600, 520, 200, 1600, 1, 0))
    s.wash(s.full(), UMBER, 0.16, wob=0, soft=0, edge=0, gran=0.3, flow=0.4, grad=s.rgrad(1100, 600, 1000, 2300, 0, 1, sy=0.7))
    note = FNOTE
    ksp = KSPOUT
    fl = [pj(0, 0, 0), pj(614.4, 0, 0), pj(614.4, 0, 178), pj(0, 0, 178)]
    s.save("kt_bg", opaque=True, q=90, meta={
        "window": [round(v) for v in WGLASS], "kettle_spout": [round(ksp[0]), round(ksp[1])], "fridge_note": [round(note[0]), round(note[1])],
        "floor": [[round(x), round(y)] for x, y in fl], "wall": [[0, round(pj(0, 250)[1])], [SW, round(pj(0, 250)[1])], [SW, FLOOR], [0, FLOOR]],
        "persp": {"vp": [VPX, VPY], "px_per_cm": PX, "dist_cm": DIST, "floor_y": FLOOR},
        "floor_open": [[round(x), round(y)] for x, y in (pj(0, 0, 62), pj(614.4, 0, 62), pj(614.4, 0, 178), pj(0, 0, 178))],
        "burner": [round(v) for v in pj(KETTLE[0], 97, KETTLE[1])], "table": [round(v) for v in pj(TABLE[0], TABLE[4], TABLE[3])]})
    # the occluder: the same paint, only where there is a thing standing in the room (window, shelves, hood, calendar,
    # fridge, counter, stove, kettle, cooker, table, chairs). The page draws kt_bg, then the sun, then this.
    o = Sheet(SW, SH, 309)
    body = np.clip(cv2.GaussianBlur(s.body, (0, 0), 0.8) * 1.05, 0, 1)
    o.A = s.A * body[..., None]
    o.body = body
    o.save("kt_occ", crop=True, q=90)


def floor(s, r):
    fl = s.rect(-10, FLOOR, SW + 10, SH + 10)
    s.lift(fl, 1.0, soft=0.6, wob=0)
    s.wash(fl, SIENNA, 0.18, wob=0, edge=0, gran=0.5)                 # grout: warm, pale
    groups = [np.zeros((s.h, s.w), np.uint8) for _ in range(4)]
    T = 30
    for j in range(0, 7):
        for i in range(-30, 50):
            d0, d1 = j * T, (j + 1) * T
            X0, X1 = i * T, (i + 1) * T
            pts = [pj(X0 + 1.0, 0, d0 + 0.9), pj(X1 - 1.0, 0, d0 + 0.9), pj(X1 - 1.0, 0, d1 - 0.9), pj(X0 + 1.0, 0, d1 - 0.9)]
            if max(p[0] for p in pts) < -20 or min(p[0] for p in pts) > SW + 20 or min(p[1] for p in pts) > SH:
                continue
            P = np.round(np.array(pts) * 16).astype(np.int32)
            g = min(3, int(r.random() * 4.4))
            cv2.fillPoly(groups[g], [P], 255, cv2.LINE_AA, 4)
    for g, (col, kk) in zip(groups, ((TERRA, 0.55), (TERRA, 0.44), (BURNT, 0.32), (TERRA, 0.64))):
        m = g.astype(np.float32) / 255 * fl
        s.wash(m, col, kk, wob=1.0, soft=1.0, edge=0.6, ew=4, gran=0.55, flow=0.4, fs=60)
    s.wash(fl, UMBER, 0.12, wob=0, edge=0, grad=s.noise(200, 3))
    mm = cv2.GaussianBlur(fl, (0, 0), 1.0)
    for i in range(-30, 50):
        a, b = pj(i * T, 0, 0), pj(i * T, 0, 185)
        if -400 < b[0] < SW + 400:
            seg(s, a, b, 1.3, SEPIA, 0.55, skip=0.35, r=r, clip=mm)
    for j in range(0, 7):
        a, b = pj(-300, 0, j * T), pj(900, 0, j * T)
        seg(s, a, b, 1.3, SEPIA, 0.55, skip=0.35, r=r, clip=mm)
    # the skirting where the tiles meet the floor
    sk = s.rect(-10, FLOOR - 22, SW + 10, FLOOR + 1)
    base(s, sk, CELADON, 0.9, wob=0.6, edge=0.4)
    s.wash(sk, TEAL, 0.35, wob=0.6, soft=2)
    seg(s, (-10, FLOOR - 22), (SW + 10, FLOOR - 21), 1.6, SLATE, 0.9, r=r)
    seg(s, (-10, FLOOR + 1), (SW + 10, FLOOR + 1), 2.0, INK, 1.2, r=r)


def window(s, r):
    global WGLASS
    X0, X1, h0, h1 = WIN
    o = [pj(X0, h1), pj(X1, h1), pj(X1, h0), pj(X0, h0)]            # opening in the wall plane
    g = [pj(X0, h1, -16), pj(X1, h1, -16), pj(X1, h0, -16), pj(X0, h0, -16)]   # glass plane, set back into the wall
    WGLASS = (g[0][0], g[0][1], g[2][0], g[2][1])
    op = Q(s, o)
    gl = Q(s, g)
    # outside: the morning sky, the neighbour's tiled wall across the lane, its own window and grille, an air conditioner
    s.lift(op, 1.0, soft=0.6, wob=0)
    s.wash(gl, MIST, 0.40, wob=0, edge=0, grad=s.vgrad(g[0][1], g[2][1], 1, 0.2), flow=0.4, fs=200, body=1)
    s.wash(gl, RAIN, 0.18, wob=0, edge=0, grad=s.vgrad(g[0][1], g[0][1] + 200, 1, 0))
    nb = Q(s, [(g[0][0] - 10, 492), (g[1][0] + 10, 476), (g[1][0] + 10, g[2][1] + 10), (g[0][0] - 10, g[2][1] + 10)]) * gl
    s.wash(nb, CREAM, 0.45, wob=4, bleed=10, edge=0.4, flow=0.5, fs=120, gran=0.5)
    s.wash(nb, ROSE, 0.07, wob=3, edge=0, clip=nb)
    s.wash(nb, RAIN, 0.30, wob=3, bleed=20, grad=s.vgrad(560, g[2][1], 0, 1), edge=0, clip=nb)
    s.lift(nb * s.vgrad(480, 600, 1, 0), 0.3, soft=6, wob=3)                                                    # the sun on its top floor
    for yy in np.arange(500, g[2][1], 26):          # its tiles, soft
        s.ink([(g[0][0], yy + (g[0][0] - 200) * -0.02), (g[1][0], yy - 18 + (g[1][0] - 200) * -0.0)], 1.0, SLATE, 0.35, skip=0.4, clip=nb, tremor=1.2)
    nw = Q(s, [(560, 540), (800, 534), (800, 700), (560, 704)]) * gl          # the neighbour's window
    s.wash(nw, PAYNE, 0.55, wob=3, bleed=6, edge=0.3)
    s.wash(nw, INDIGO, 0.3, wob=2, bleed=10, grad=s.vgrad(540, 700, 0, 1))
    for x in np.arange(572, 800, 30):
        s.ink([(x, 528), (x, 712)], 2.2, INK, 0.9, clip=gl, tremor=0.8)
    s.ink([(552, 528), (808, 524)], 2.6, INK, 1.0, clip=gl)
    s.ink([(552, 712), (808, 708)], 2.6, INK, 1.0, clip=gl)
    for x in (620, 690, 760):
        s.ink(ell(x, 620, 14, 18, 0.3, 6.0, 20), 1.6, INK, 0.8, clip=gl)
    ac = Q(s, [(300, 590), (450, 586), (452, 676), (302, 680)]) * gl           # an air conditioner on a bracket
    base(s, ac, CREAM, 0.35, wob=1.5, edge=0.4)
    s.wash(ac, SLATE, 0.3, wob=1, soft=3, grad=s.vgrad(590, 680))
    for yy in range(606, 668, 9):
        s.ink([(312, yy), (440, yy - 1)], 1.2, SLATE, 0.7, clip=gl)
    contour(s, ac, 1.6, SEPIA, 1.0)
    s.ink([(310, 680), (330, 700), (440, 700), (448, 676)], 1.6, SEPIA, 0.9, clip=gl)
    pt = Q(s, [(838, 680), (900, 680), (892, 728), (846, 728)]) * gl            # a potted plant on the neighbour's sill
    s.wash(pt, TERRA, 0.8, wob=1.5, edge=0.4)
    for i in range(9):
        a = -math.pi / 2 + (i - 4) * 0.3
        s.stroke([(869, 680), (869 + math.cos(a) * 40, 680 + math.sin(a) * 46)], 12, MOSS, 0.7, wob=1.5, clip=gl, taper=(0.2, 0.6))
    for (a, b, c_) in (((200, 360), (600, 392), (1000, 340)), ((200, 400), (560, 430), (1000, 410))):   # electric wires
        s.ink([a, b, c_], 1.5, PAYNE, 0.75, clip=gl, tremor=1.0)
    # the glass: a faint cool sheen and a diagonal reflection
    s.wash(gl, RAIN, 0.10, wob=0, edge=0, grad=s.hgrad(g[0][0], g[1][0], 0, 1))
    s.lift(Q(s, [(700, g[0][1]), (790, g[0][1]), (560, g[2][1]), (470, g[2][1])]) * gl, 0.18, soft=14, wob=4)
    # the iron grille outside the glass: a frame, bars, and the scrolls (鐵窗花)
    grille(s, r, g, gl)
    # the reveal: the wall's thickness round the opening (left and top faces visible)
    rv = Q(s, [o[0], g[0], g[3], o[3]])
    base(s, rv, CREAM, 0.45, wob=0.8, edge=0.4)
    s.wash(rv, UMBER, 0.30, wob=0.8, soft=3)
    rt = Q(s, [o[0], o[1], g[1], g[0]])
    base(s, rt, CREAM, 0.45, wob=0.8, edge=0.4)
    s.wash(rt, UMBER, 0.45, wob=0.8, soft=3)
    sill = Q(s, [o[3], o[2], pj(X1 + 4, h0 - 1, 10), pj(X0 - 4, h0 - 1, 10)])          # a tiled sill sticking out a little
    base(s, sill, CELADON, 0.6, wob=0.8, edge=0.5)
    s.lift(sill, 0.3, soft=4)
    sf = Q(s, [pj(X0 - 4, h0 - 1, 10), pj(X1 + 4, h0 - 1, 10), pj(X1 + 4, h0 - 5, 10), pj(X0 - 4, h0 - 5, 10)])
    base(s, sf, CELADON, 0.85, wob=0.6, edge=0.5)
    s.wash(sf, TEAL, 0.3, wob=0.6, soft=2)
    cast(s, sf, 0, 10, UMBER, 0.35, blur=7)
    edges(s, [(X0 - 4, h0 - 1, 10), (X1 + 4, h0 - 1, 10), (X1 + 4, h0 - 5, 10), (X0 - 4, h0 - 5, 10)], 1.8, 1.2, r)
    # the aluminium window frame: two sliding sashes
    fw = 26
    xm = (g[0][0] + g[1][0]) / 2
    def bar(x0, y0, x1, y1):
        m = s.rect(x0, y0, x1, y1)
        base(s, m, MIST, 0.55, wob=0.5, edge=0.4)
        s.wash(m, SLATE, 0.35, wob=0.5, soft=2, grad=s.hgrad(x0, x1) if (x1 - x0) < (y1 - y0) else s.vgrad(y0, y1))
        path(s, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], 1.6, INK, 1.1, closed=True, over=2, r=r)
    bar(g[0][0], g[0][1], g[1][0], g[0][1] + fw)
    bar(g[0][0], g[2][1] - fw, g[1][0], g[2][1])
    bar(g[0][0], g[0][1], g[0][0] + fw, g[2][1])
    bar(g[1][0] - fw, g[0][1], g[1][0], g[2][1])
    bar(xm - 24, g[0][1] + fw, xm, g[2][1] - fw)
    bar(xm + 2, g[0][1] + fw, xm + 26, g[2][1] - fw)
    s.lift(s.rect(g[0][0] + fw, g[0][1] + fw, xm - 24, g[0][1] + fw + 4), 0.3, soft=2)
    # on the sill: dish soap, a sponge, a little steel cup of chopsticks
    bx, by = pj(70, WIN[2], 6)
    bt = s.poly([(bx - 16, by), (bx + 16, by), (bx + 18, by - 70), (bx + 8, by - 84), (bx + 6, by - 100), (bx - 6, by - 100), (bx - 8, by - 84), (bx - 18, by - 70)])
    base(s, bt, LEAF, 0.55, wob=1, edge=0.5)
    s.wash(bt, MINT, 0.4, wob=1, soft=3, grad=s.hgrad(bx - 16, bx + 16, 1, 0))
    s.lift(s.rect(bx - 10, by - 64, bx - 4, by - 10), 0.5, soft=2)
    lbl = s.rect(bx - 15, by - 54, bx + 15, by - 26)
    base(s, lbl, CREAM, 0.2, wob=0.5)
    contour(s, bt, 1.6, INK, 1.2)
    sx, sy = pj(96, WIN[2], 6)
    sp = s.rect(sx - 26, sy - 22, sx + 26, sy)
    base(s, sp, NOTE, 0.75, wob=1.2, edge=0.5, gran=0.7)
    s.wash(s.rect(sx - 26, sy - 22, sx + 26, sy - 14), LEAF, 0.8, wob=1, edge=0.4)
    contour(s, sp, 1.4, INK, 1.0)
    cx, cy = pj(176, WIN[2], 6)
    cup = s.poly([(cx - 20, cy), (cx + 20, cy), (cx + 22, cy - 54), (cx - 22, cy - 54)])
    for i in range(7):
        x = cx - 14 + i * 4.5
        s.ink([(x, cy - 50), (x + (i - 3) * 3, cy - 120 - (i % 3) * 8)], 3.0, [SIENNA, BURNT, UMBER][i % 3], 1.0, taper=(0, 0.1))
    base(s, cup, MIST, 0.6, wob=0.8, edge=0.4)
    s.wash(cup, SLATE, 0.4, wob=0.6, soft=2, grad=s.hgrad(cx - 20, cx + 22))
    s.lift(s.rect(cx - 12, cy - 50, cx - 6, cy - 4), 0.6, soft=1.5)
    contour(s, cup, 1.5, INK, 1.1)


def volute(x0, y0, cx, cy, R0, sg, up=False, turns=1.55):
    """A scroll of flat bar: it leaves (x0, y0) and runs round the centre (cx, cy), tightening into a curl.
    sg = +1 curls clockwise on the right half, -1 mirrors it; up turns it upside down."""
    pts = [(x0, y0)]
    th = np.linspace(-0.9, 2 * math.pi * turns - 0.9, 70)
    R = R0 * (1 - 0.72 * (th - th[0]) / (th[-1] - th[0]))
    for a, rr_ in zip(th, R):
        dx, dy = sg * rr_ * math.sin(a), -rr_ * math.cos(a)
        pts.append((cx + dx, cy - dy if up else cy + dy))
    return pts


def grille(s, r, g, gl):
    """The outdoor iron grille: a rectangular frame of flat bar, uprights, a rail across, and in every bay a pair of
    C-scrolls back to back (a heart) with a ring over it. Painted dark green-black, rust where water sits."""
    x0, y0, x1, y1 = g[0][0] + 8, g[0][1] + 8, g[1][0] - 8, g[2][1] - 6
    gm = np.zeros((s.h, s.w), np.float32)
    W_ = 7.0

    def line(pts, w=W_):
        nonlocal gm
        gm = np.maximum(gm, s.ribbon(pts, w, smooth=len(pts) > 2))
    line([(x0, y0), (x1, y0)])
    line([(x0, y1), (x1, y1)])
    line([(x0, y0), (x0, y1)])
    line([(x1, y0), (x1, y1)])
    nb = 6
    bw = (x1 - x0) / nb
    yr = y0 + (y1 - y0) * 0.30
    line([(x0, yr), (x1, yr)])
    for i in range(1, nb):
        line([(x0 + i * bw, y0), (x0 + i * bw, y1)], 6)
    for i in range(nb):
        cx = x0 + (i + 0.5) * bw
        # ring in the upper band
        ring = s.ellipse(cx, (y0 + yr) / 2, bw * 0.26, bw * 0.26) * (1 - s.ellipse(cx, (y0 + yr) / 2, bw * 0.26 - 5, bw * 0.26 - 5))
        gm = np.maximum(gm, ring)
        line([(cx - bw * 0.5, (y0 + yr) / 2), (cx - bw * 0.26, (y0 + yr) / 2)], 5)
        line([(cx + bw * 0.26, (y0 + yr) / 2), (cx + bw * 0.5, (y0 + yr) / 2)], 5)
        # in the lower bay: a heart of two volutes hanging from the rail, and under it two more curling up
        hh = y1 - yr
        for sg in (-1, 1):
            line(volute(cx, yr + 4, cx + sg * bw * 0.19, yr + hh * 0.33, bw * 0.29, sg, up=False, turns=1.3), 5.5)
            line(volute(cx, y1 - 4, cx + sg * bw * 0.19, yr + hh * 0.72, bw * 0.26, sg, up=True, turns=1.3), 5.5)
        ring2 = s.ellipse(cx, yr + hh * 0.53, bw * 0.09, bw * 0.09) * (1 - s.ellipse(cx, yr + hh * 0.53, bw * 0.09 - 4.5, bw * 0.09 - 4.5))
        gm = np.maximum(gm, ring2)
    gm = gm * gl
    # its shadow falls outward, onto nothing: so only a soft dark halo in the glass, and the grille itself
    s.wash(cv2.GaussianBlur(gm, (0, 0), 3), PAYNE, 0.25, wob=0, soft=0, edge=0)
    s.lift(gm, 1.0, soft=0.4, wob=0)
    s.wash(gm, PAYNE, 1.55, wob=0.6, soft=0.5, edge=0.4, gran=0.3)
    s.wash(gm, LEAF, 0.35, wob=0.6, soft=0.5, edge=0)
    s.wash(gm * s.vgrad(y1 - 60, y1, 0, 1), BURNT, 0.6, wob=1, soft=1, edge=0)         # rust along the bottom rail
    for _ in range(26):
        bx, by = x0 + r.random() * (x1 - x0), y0 + r.random() * (y1 - y0)
        s.wash(s.blob(bx, by, 6, 4, 0.4) * gm, BURNT, 0.8, wob=0.5, soft=0.5)
    s.lift(gm * s.hgrad(x0, x1, 0.5, 0) * (s.noise(18, 2) > 0.6), 0.25, soft=0.6)       # daylight catching the near edges


def shelves(s, r):
    """Two open shelves on the wall right of the hood: bowls, jars, a tea tin."""
    X0, X1 = 340, 416
    for h in (128, 162):
        top = quad(s, [(X0, h, 0), (X1, h, 0), (X1, h, 24), (X0, h, 24)])
        front = quad(s, [(X0, h, 24), (X1, h, 24), (X1, h - 3, 24), (X0, h - 3, 24)])
        under = Q(s, [pj(X0, h - 3, 0), pj(X1, h - 3, 0), pj(X1, h - 3, 24), pj(X0, h - 3, 24)])
        cast(s, front, 0, 26, UMBER, 0.45, blur=14)
        for bx in (X0 + 8, X1 - 8):                                    # brackets
            br = Q(s, [pj(bx - 1, h - 3, 1), pj(bx + 1, h - 3, 1), pj(bx + 1, h - 18, 1), pj(bx - 1, h - 18, 1)])
            base(s, br, PAYNE, 0.8, wob=0.4)
            s.ink([pj(bx, h - 18, 0), pj(bx, h - 3, 20)], 2.0, INK, 1.1)
        wood(s, np.maximum(front, under), SIENNA, BURNT, 0.85, n=6, seed=int(h), wob=0.6, lines=False)
        edges(s, [(X0, h, 24), (X1, h, 24), (X1, h - 3, 24), (X0, h - 3, 24)], 1.8, 1.3, r)
    # lower shelf: a stack of rice bowls (blue rims), a stack of small plates, two jars
    h = 128
    for k_, (X, n_) in enumerate(((352, 4), (368, 3))):
        bx, by = pj(X, h, 12)
        for i in range(n_):
            yy = by - i * 12
            m = s.poly([(bx - 34, yy - 16), (bx + 34, yy - 16), (bx + 24, yy), (bx - 24, yy)])
            base(s, m, CREAM, 0.15, wob=0.6, edge=0.4)
            s.wash(m, MIST, 0.3, wob=0.5, soft=2, grad=s.hgrad(bx - 30, bx + 34))
            s.wash(s.rect(bx - 34, yy - 16, bx + 34, yy - 12) * m, PRUSSIAN, 1.0, wob=0.5, soft=0.6)
            s.ink([(bx - 34, yy - 16), (bx - 24, yy), (bx + 24, yy), (bx + 34, yy - 16)], 1.4, INK, 1.0)
        s.ink(ell(bx, by - (n_ - 1) * 12 - 16, 34, 6, 0, 2 * math.pi, 30), 1.4, INK, 1.0)
    for X, col, hh in ((390, OCHRE, 70), (404, SEPIA, 60)):
        bx, by = pj(X, h, 10)
        jar = s.rect(bx - 22, by - hh, bx + 22, by)
        base(s, jar, MIST, 0.2, wob=0.6, edge=0.4)
        s.wash(s.rect(bx - 20, by - hh * 0.65, bx + 20, by - 2), col, 0.9, wob=0.8, edge=0.4, gran=0.7)
        s.spatter(10, UMBER, 0.6, box=(bx - 18, by - hh * 0.6, bx + 18, by - 4), r=(1.5, 3.5))
        lid = s.rect(bx - 24, by - hh - 12, bx + 24, by - hh)
        base(s, lid, TERRA, 0.9, wob=0.5, edge=0.4)
        s.lift(s.rect(bx - 14, by - hh + 6, bx - 9, by - 8), 0.6, soft=1.5)
        contour(s, np.maximum(jar, lid), 1.5, INK, 1.1)
    # upper shelf: a red tea tin, a teapot, a stack of plates on edge
    h = 162
    bx, by = pj(350, h, 12)
    tin = s.rect(bx - 26, by - 62, bx + 26, by)
    base(s, tin, TERRA, 1.0, wob=0.6, edge=0.5)
    s.wash(tin, BURNT, 0.4, wob=0.5, soft=2, grad=s.hgrad(bx - 20, bx + 26))
    s.lift(text_mask(s, "茶", bx, by - 30, 30, FONT_H), 0.75, soft=0.5, wob=0.3)
    contour(s, tin, 1.5, INK, 1.1)
    tx, ty = pj(372, h, 12)
    pot = s.ellipse(tx, ty - 32, 40, 32)
    sp_ = s.ribbon([(tx + 34, ty - 30), (tx + 58, ty - 52), (tx + 66, ty - 64)], 10, profile=lambda t: 1 - 0.4 * t)
    hd = s.ribbon([(tx - 36, ty - 44), (tx - 58, ty - 38), (tx - 54, ty - 18), (tx - 36, ty - 16)], 7)
    pm = np.maximum(np.maximum(pot, sp_), hd)
    base(s, pm, CELADON, 0.9, wob=0.6, edge=0.5)
    s.wash(pm, TEAL, 0.35, wob=0.5, soft=3, grad=s.hgrad(tx - 30, tx + 50))
    s.lift(s.ellipse(tx - 16, ty - 44, 10, 6), 0.7, soft=1.5)
    lid = s.ellipse(tx, ty - 62, 18, 6)
    base(s, lid, CELADON, 1.0, wob=0.4)
    contour(s, np.maximum(pm, lid), 1.5, INK, 1.1)
    for i in range(5):
        px_, py_ = pj(392 + i * 4.5, h, 6)
        pl = s.ellipse(px_, py_ - 34, 7, 34)
        base(s, pl, CREAM, 0.15, wob=0.4, edge=0.4)
        s.wash(pl, MIST, 0.3, wob=0.4, soft=1.5)
        s.ink(ell(px_, py_ - 34, 7, 34, -1.6, 1.6, 20), 1.3, INK, 1.0)
    # under the lower shelf: three hooks with mugs
    for i, col in enumerate((NAVY, CREAM, MINT)):
        hx_, hy_ = pj(350 + i * 22, 118, 4)
        s.ink([(hx_, hy_ - 14), (hx_, hy_), (hx_ + 6, hy_ + 6)], 2.0, INK, 1.2)
        m = s.rect(hx_ - 14, hy_ + 4, hx_ + 18, hy_ + 46)
        hd = s.ribbon([(hx_ + 16, hy_ + 12), (hx_ + 30, hy_ + 18), (hx_ + 30, hy_ + 32), (hx_ + 16, hy_ + 38)], 6)
        mm = np.maximum(m, hd)
        base(s, mm, col, 0.9 if col is NAVY else 0.5, wob=0.6, edge=0.5)
        s.wash(mm, SLATE, 0.2, wob=0.5, soft=2, grad=s.hgrad(hx_ - 10, hx_ + 20))
        contour(s, mm, 1.4, INK, 1.0)


def hood(s, r):
    """The range hood: an old enamel box over the stove, a grease-browned underside, two switches."""
    X0, X1 = 238, 326
    F = boxf(X0, X1, 166, 194, 0, 48)
    front = quad(s, F["front"])
    side = quad(s, F["side"]) if "side" in F else s.zeros()
    under = quad(s, [(X0, 166, 0), (X1, 166, 0), (X1, 166, 48), (X0, 166, 48)])
    cast(s, np.maximum(front, side), 14, 30, UMBER, 0.45, blur=18)
    base(s, under, UMBER, 0.6, wob=0.8, edge=0.4)
    s.wash(under, SEPIA, 0.4, wob=1, bleed=6)
    base(s, side, CREAM, 0.55, wob=0.8, edge=0.4)
    s.wash(side, SLATE, 0.35, wob=0.6, soft=3)
    base(s, front, CREAM, 0.45, wob=0.8, edge=0.4, flow=0.3)
    s.wash(front, SIENNA, 0.25, wob=1.5, bleed=10, grad=s.vgrad(pj(0, 180, 48)[1], pj(0, 166, 48)[1], 0, 1), clip=front)     # grease
    s.lift(front * s.vgrad(pj(0, 194, 48)[1], pj(0, 188, 48)[1], 1, 0), 0.25, soft=2)
    # a duct going up into the ceiling
    d0, d1 = pj(270, 194, 20), pj(296, 260, 20)
    duct = s.rect(d0[0], d1[1], d1[0], d0[1])
    base(s, duct, MIST, 0.6, wob=0.6, edge=0.4)
    s.wash(duct, SLATE, 0.4, wob=0.5, soft=3, grad=s.hgrad(d0[0], d1[0]))
    for yy in np.arange(d1[1] + 20, d0[1], 34):
        s.ink([(d0[0], yy), (d1[0], yy + 2)], 1.3, SLATE, 0.8)
    contour(s, duct, 1.6, INK, 1.1)
    for i, X in enumerate((304, 314)):
        sx, sy = pj(X, 176, 48)
        sm = s.rect(sx - 10, sy - 12, sx + 10, sy + 12)
        base(s, sm, CREAM, 0.3, wob=0.4)
        s.wash(s.rect(sx - 5, sy - (8 if i else 0), sx + 5, sy + (0 if i else 8)), INK, 1.1, wob=0.3, soft=0.5)
        path(s, [(sx - 10, sy - 12), (sx + 10, sy - 12), (sx + 10, sy + 12), (sx - 10, sy + 12)], 1.3, INK, 1.0, closed=True, over=1, r=r)
    edges(s, F["front"], 2.2, 1.4, r)
    if "side" in F:
        edges(s, F["side"], 2.0, 1.3, r)


def calendar(s, r):
    """A tear-off day calendar on the wall right of the fridge: a red board, a thick pad, today's big number."""
    cx, cy = pj(552, 150)
    bd = s.rect(cx - 70, cy - 120, cx + 70, cy + 110)
    cast(s, bd, 10, 14, UMBER, 0.4, blur=8)
    base(s, bd, TERRA, 1.0, wob=0.8, edge=0.5)
    s.wash(bd, BURNT, 0.3, wob=0.6, soft=3, grad=s.vgrad(cy - 120, cy + 110))
    s.wash(text_mask(s, "福", cx, cy - 92, 30, FONT_H), OCHRE, 1.3, wob=0.4, soft=0.5)
    pad = s.rect(cx - 56, cy - 64, cx + 56, cy + 92)
    base(s, pad, CREAM, 0.12, wob=0.6, edge=0.4)
    s.wash(s.rect(cx - 56, cy + 84, cx + 56, cy + 92), SLATE, 0.4, wob=0.4)
    s.wash(text_mask(s, "十月", cx, cy - 46, 18, FONT_P), INK, 1.0, wob=0.2, soft=0.4)
    s.wash(text_mask(s, "16", cx, cy + 10, 72, FONT_P), TERRA, 1.3, wob=0.3, soft=0.5, edge=0.3)
    s.wash(text_mask(s, "星期五", cx, cy + 62, 16, FONT_P), INK, 0.9, wob=0.2, soft=0.4)
    contour(s, bd, 1.6, INK, 1.2)
    path(s, [(cx - 56, cy - 64), (cx + 56, cy - 64), (cx + 56, cy + 92), (cx - 56, cy + 92)], 1.3, INK, 1.0, closed=True, over=2, r=r)
    s.ink([(cx, cy - 120), (cx, cy - 150)], 1.6, INK, 1.1)
    s.wash(s.ellipse(cx, cy - 152, 4, 4), INK, 1.2, wob=0.2)
    # a light switch below it
    lx, ly = pj(530, 118)
    sw = s.rect(lx - 26, ly - 36, lx + 26, ly + 36)
    base(s, sw, CREAM, 0.25, wob=0.5, edge=0.4)
    s.wash(s.rect(lx - 8, ly - 18, lx + 8, ly + 18), MIST, 0.6, wob=0.3, soft=1)
    contour(s, sw, 1.4, INK, 1.1)


def fridge(s, r):
    global FNOTE
    X0, X1, H, D = FRG
    F = boxf(X0, X1, 0, H, 0, D)
    front = quad(s, F["front"])
    side = quad(s, F["side"])
    allm = np.maximum(front, side)
    cast(s, allm, 30, 6, UMBER, 0.40, blur=26)
    s.wash(quad(s, [(X0 - 4, 0, 0), (X1 + 10, 0, 0), (X1 + 10, 0, D + 12), (X0 - 4, 0, D + 12)]), SEPIA, 0.55, wob=2, bleed=14, edge=0.1)
    base(s, side, CREAM, 0.55, wob=0.8, edge=0.4, flow=0.3)
    s.wash(side, RAIN, 0.40, wob=0.8, soft=3, grad=s.vgrad(pj(0, H, 0)[1], pj(0, 0, 0)[1], 0.4, 1))
    s.wash(side, UMBER, 0.12, wob=0.8, soft=3)
    # the front: freezer door above, main door below, rounded corners, the chrome pulls at the left
    fx0, fy0 = pj(X0, H, D)
    fx1, fy1 = pj(X1, 0, D)
    ys = pj(0, 124, D)[1]
    door = Q(s, rr(fx0 + 4, ys + 4, fx1 - 4, fy1 - 18, 12, step=10))
    frz = Q(s, rr(fx0 + 4, fy0 + 2, fx1 - 4, ys - 3, 12, step=10))
    base(s, front, CREAM, 0.35, wob=0.6, edge=0.3)
    s.wash(front, SEPIA, 0.7, wob=0.5, soft=1, clip=1 - np.maximum(door, frz))          # the gaps round the doors
    face = np.maximum(door, frz)
    base(s, face, CREAM, 0.50, wob=0.8, edge=0.3, flow=0.35, fs=120)
    s.wash(face, OCHRE, 0.16, wob=0, edge=0, grad=s.hgrad(fx0, fx1, 1, 0))
    s.wash(face, RAIN, 0.26, wob=0, edge=0, grad=s.hgrad(fx0 + (fx1 - fx0) * 0.6, fx1, 0, 1) ** 1.5)
    s.lift(s.rect(fx0 + 60, ys, fx0 + 84, fy1 - 30) * door, 0.25, soft=8)
    s.wash(s.rect(fx0 - 4, fy1 - 18, fx1 + 4, fy1 + 2), SEPIA, 0.9, wob=0.5, soft=1)                # the kick plate
    hx = fx0 + 26
    for y0_, y1_ in ((ys + 50, ys + 300), (fy0 + 30, ys - 20)):
        hm = s.rect(hx - 5, y0_, hx + 5, y1_)
        cast(s, hm, 8, 5, UMBER, 0.45, blur=4)
        base(s, hm, MIST, 0.6, wob=0.3, edge=0.3)
        s.wash(s.rect(hx - 5, y0_, hx - 1, y1_), PAYNE, 0.8, wob=0.3, soft=0.8)
        s.ink([(hx - 5, y0_), (hx - 5, y1_)], 1.4, INK, 1.1)
        s.ink([(hx + 5, y0_), (hx + 5, y1_)], 1.4, INK, 1.1)
    # the things on the door, small: the menu, the strawberry, three magnets and the receipt, the calendar, the gas card
    sc = (fx1 - fx0) / 2200.0                                             # the close-up's door is 2200 px wide
    def cl(x, y):          # close-up sheet coords -> here (the close-up's seam at y 150 is our freezer line)
        return (fx0 + (x - 70) * sc, ys + (y - 150) * sc)
    def slip(x0, y0, x1, y1, col=CREAM, k=0.08):
        a, b = cl(x0, y0), cl(x1, y1)
        m = s.rect(a[0], a[1], b[0], b[1])
        cast(s, m, 2, 3, UMBER, 0.35, blur=2)
        base(s, m, col, k, wob=0.3, edge=0.3)
        path(s, [a, (b[0], a[1]), b, (a[0], b[1])], 1.2, INK, 0.9, closed=True, over=1, r=r)
        return a, b
    a, b = slip(320, 620, 730, 1300)
    s.wash(s.rect(a[0] + 4, a[1] + 4, b[0] - 4, a[1] + 18), TERRA, 0.9, wob=0.3, soft=0.5)
    for i in range(7):
        s.ink([(a[0] + 6, a[1] + 28 + i * 8), (b[0] - 8 - (i % 3) * 4, a[1] + 28 + i * 8)], 1.0, SLATE, 0.7, skip=0.3, tremor=0.2)
    a, b = slip(1846, 760, 2170, 1150)
    s.wash(s.rect(a[0] + 4, a[1] + 4, a[0] + 22, a[1] + 14), TERRA, 1.0, wob=0.3, soft=0.5)
    for i in range(5):
        s.ink([(a[0] + 5, a[1] + 22 + i * 8), (b[0] - 5, a[1] + 22 + i * 8)], 0.9, SLATE, 0.6, skip=0.2, tremor=0.2)
    a, b = slip(1915, 280, 2060, 600, CREAM, 0.05)
    a, b = slip(1870, 1236, 2150, 1384, TERRA, 0.35)
    for (x, y, col) in ((520, 960 - 345 + 24, TERRA), (1893, 268, RAIN), (1985, 290, OCHRE), (2081, 258, TEAL), (560, 300, ROSE)):
        px_, py_ = cl(x, y)
        m = s.ellipse(px_, py_, 6, 6)
        cast(s, m, 2, 3, UMBER, 0.4, blur=2)
        base(s, m, col, 1.0, wob=0.2, edge=0.4)
        oval(s, px_, py_, 6, 6, 1.0, INK, 1.0)
    # the yellow note
    nx, ny = cl(1300, 470 + 215)
    FNOTE = (nx, ny)
    nh = 24
    nm = Q(s, [(nx - nh, ny - nh), (nx + nh, ny - nh - 1), (nx + nh + 1, ny + nh), (nx - nh, ny + nh + 1)])
    cast(s, nm, 3, 4, UMBER, 0.4, blur=3)
    base(s, nm, NOTE, 0.9, wob=0.4, edge=0.5, gran=0.4)
    s.wash(nm * s.vgrad(ny - nh, ny - nh + 8, 1, 0), NOTE, 0.35, wob=0.3, soft=1)
    path(s, [(nx - nh, ny - nh), (nx + nh, ny - nh - 1), (nx + nh + 1, ny + nh), (nx - nh, ny + nh + 1)], 1.4, INK, 1.1, closed=True, over=2, r=r)
    # pen
    edges(s, F["side"], 2.2, 1.3, r)
    s.ink(dense(rr(fx0 + 4, ys + 4, fx1 - 4, fy1 - 18, 12, step=10), 6), 1.8, INK, 1.2, closed=True)
    s.ink(dense(rr(fx0 + 4, fy0 + 2, fx1 - 4, ys - 3, 12, step=10), 6), 1.8, INK, 1.2, closed=True)
    edges(s, F["front"], 2.4, 1.4, r, over=2)
    # on top: a woven basket and a folded cloth
    bx0, by0 = pj(X0 + 8, H, D - 14)
    bx1, by1 = pj(X0 + 46, H + 18, D - 14)
    bk = s.poly([(bx0, by0), (bx1, by0), (bx1 + 6, by1), (bx0 - 6, by1)], smooth=False)
    base(s, bk, SIENNA, 0.75, wob=0.8, edge=0.5, gran=0.6)
    s.hatch(bk, 0.7, 9, 1.2, BURNT, 0.8, skip=0.2)
    s.hatch(bk, -0.7, 9, 1.2, BURNT, 0.6, skip=0.3)
    contour(s, bk, 1.6, INK, 1.2)


def counter(s, r):
    X0, X1, H, D = CTR
    F = boxf(X0, X1, 0, H, 0, D)
    front = quad(s, [(X0, 10, D - 6), (X1, 10, D - 6), (X1, H - 4, D - 6), (X0, H - 4, D - 6)])
    side = quad(s, [(X1, 0, 0), (X1, 0, D), (X1, H, D), (X1, H, 0)])
    top = quad(s, F["top"])
    lip = quad(s, [(X0, H, D), (X1, H, D), (X1, H - 4, D), (X0, H - 4, D)])
    kick = quad(s, [(X0, 0, D - 10), (X1, 0, D - 10), (X1, 10, D - 10), (X0, 10, D - 10)])
    cast(s, quad(s, [(X0, 0, 0), (X1 + 4, 0, 0), (X1 + 4, 0, D + 8), (X0, 0, D + 8)]), 0, 0, UMBER, 0.45, blur=10)
    base(s, kick, SEPIA, 0.9, wob=0.5, edge=0.3)
    # cabinet doors: pale wood, two by two, little round knobs
    wood(s, front, SIENNA, BURNT, 0.55, ang=math.pi / 2, n=60, seed=9, lw=1.1, streak=0.0)
    s.wash(front, OCHRE, 0.15, wob=0, edge=0, grad=s.hgrad(0, 1800, 1, 0.2))
    s.wash(front, UMBER, 0.25, wob=0, edge=0, grad=s.vgrad(pj(0, H, D)[1], pj(0, 10, D)[1], 0.6, 0))         # under the lip
    xs = list(range(0, X1 + 1, 49))
    for i in range(len(xs) - 1):
        a, b = xs[i], xs[i + 1]
        edges(s, [(a + 1, 14, D - 6), (b - 1, 14, D - 6), (b - 1, H - 8, D - 6), (a + 1, H - 8, D - 6)], 1.5, 1.0, r, over=2)
        kx, ky = pj(b - 5 if i % 2 == 0 else a + 5, H - 18, D - 4)
        kn = s.ellipse(kx, ky, 7, 7)
        base(s, kn, MIST, 0.6, wob=0.2, edge=0.5)
        oval(s, kx, ky, 7, 7, 1.2, INK, 1.1)
    # a towel over the first door
    tx, ty = pj(26, H - 6, D - 4)
    tw = s.poly([(tx - 50, ty - 6), (tx + 50, ty - 6), (tx + 56, ty + 170), (tx + 20, ty + 182), (tx - 44, ty + 172)])
    cast(s, tw, 6, 6, UMBER, 0.35, blur=5)
    base(s, tw, CREAM, 0.25, wob=1.2, edge=0.4)
    for yy in (ty + 120, ty + 136):
        s.wash(s.rect(tx - 50, yy, tx + 56, yy + 8) * tw, TERRA, 0.7, wob=0.8, soft=0.8)
    s.wash(tw, RAIN, 0.25, wob=1, bleed=8, grad=s.hgrad(tx - 40, tx + 56))
    contour(s, tw, 1.5, INK, 1.1)
    base(s, side, SIENNA, 0.65, wob=0.8, edge=0.4)
    s.wash(side, UMBER, 0.35, wob=0.8, soft=3)
    # the top: stainless steel, cool, with the window lying on it
    base(s, top, MIST, 0.55, wob=0.6, edge=0.3, flow=0.4, fs=80)
    s.wash(top, SLATE, 0.30, wob=0, edge=0, grad=s.hgrad(1100, 2000, 0, 1))
    s.lift(top * s.hgrad(200, 1000, 1, 0) * s.hgrad(0, 200, 0, 1), 0.35, soft=6)
    base(s, lip, MIST, 0.65, wob=0.4, edge=0.4)
    s.wash(lip, SLATE, 0.45, wob=0.4, soft=1)
    # the sink: a dark slot in the top, the tap from the wall
    sk = quad(s, [(84, H, 12), (154, H, 12), (154, H, 46), (84, H, 46)])
    s.wash(sk, SLATE, 1.0, wob=0.6, soft=1, edge=0.6)
    s.wash(sk * s.vgrad(pj(0, H, 12)[1], pj(0, H, 46)[1], 1, 0.3), PAYNE, 0.6, wob=0.4, soft=2)
    edges(s, [(84, H, 12), (154, H, 12), (154, H, 46), (84, H, 46)], 1.6, 1.2, r, over=1)
    t0, t1, t2 = pj(119, 108, 0), pj(119, 122, 10), pj(119, 112, 22)
    tap = s.ribbon([t0, (t0[0], t0[1] - 30), t1, (t2[0], t2[1] - 4), t2], 9, smooth=True)
    base(s, tap, MIST, 0.7, wob=0.3, edge=0.4)
    s.lift(s.ribbon([(t0[0] - 2, t0[1]), (t1[0] - 2, t1[1] + 4)], 3), 0.5, soft=1)
    contour(s, tap, 1.5, INK, 1.2)
    vx, vy = pj(119, 104, 0)
    for dx in (-22, 22):
        v = s.ellipse(vx + dx, vy, 9, 7)
        base(s, v, MIST, 0.7, wob=0.3)
        oval(s, vx + dx, vy, 9, 7, 1.2, INK, 1.0)
    # the dish rack beside the sink: a steel basket with bowls and a plate standing in it
    rx0, ry0 = pj(166, H, 40)
    rx1, ry1 = pj(226, H + 14, 40)
    for i in range(4):
        px_ = rx0 + 18 + i * 22
        pl = s.ellipse(px_, ry1 + 4, 9, 40)
        base(s, pl, CREAM, 0.15, wob=0.4, edge=0.4)
        s.wash(pl, MIST, 0.25, wob=0.3, soft=1.5)
        s.ink(ell(px_, ry1 + 4, 9, 40, -1.5, 1.6, 16), 1.2, INK, 0.9)
    for i, (dx, col) in enumerate(((54, PRUSSIAN), (96, TERRA))):
        bwx, bwy = rx0 + dx + 40, ry1 + 2
        bw_ = s.poly([(bwx - 34, bwy - 26), (bwx + 34, bwy - 26), (bwx + 24, bwy + 14), (bwx - 24, bwy + 14)])
        base(s, bw_, CREAM, 0.15, wob=0.5, edge=0.4)
        s.wash(s.rect(bwx - 34, bwy - 26, bwx + 34, bwy - 21), col, 0.9, wob=0.4, soft=0.6)
        s.wash(bw_, MIST, 0.25, wob=0.4, soft=2, grad=s.hgrad(bwx - 30, bwx + 34))
        contour(s, bw_, 1.3, INK, 1.0)
    rk = Q(s, [(rx0, ry0), (rx1 + 40, ry0), (rx1 + 40, ry1 + 18), (rx0, ry1 + 18)])
    for x in np.arange(rx0, rx1 + 40, 12):
        s.ink([(x, ry1 + 18), (x, ry0)], 1.2, SLATE, 0.9, tremor=0.3)
    path(s, [(rx0, ry1 + 18), (rx1 + 40, ry1 + 18), (rx1 + 40, ry0), (rx0, ry0)], 1.8, INK, 1.2, closed=True, over=2, r=r)
    edges(s, F["top"], 2.0, 1.2, r, over=2)
    edges(s, [(X0, H, D), (X1, H, D), (X1, H - 4, D), (X0, H - 4, D)], 2.2, 1.4, r, over=2)
    edges(s, [(X1, 0, 0), (X1, 0, D), (X1, H, D), (X1, H, 0)], 2.2, 1.4, r, over=2)
    seg(s, pj(X0, 10, D - 6), pj(X1, 10, D - 6), 1.6, INK, 1.1, r=r)


def stove(s, r):
    """The two-burner table-top gas stove, a kettle on the left ring, its orange hose."""
    global KSPOUT
    X0, X1, d0, d1 = STOVE
    H = CTR[2]
    top = quad(s, [(X0, H + 11, d0), (X1, H + 11, d0), (X1, H + 11, d1), (X0, H + 11, d1)])
    front = quad(s, [(X0, H, d1), (X1, H, d1), (X1, H + 11, d1), (X0, H + 11, d1)])
    cast(s, quad(s, [(X0, H, d0), (X1, H, d0), (X1, H, d1 + 4), (X0, H, d1 + 4)]), 8, 6, UMBER, 0.5, blur=6)
    base(s, front, MIST, 0.55, wob=0.5, edge=0.4)
    s.wash(front, SLATE, 0.30, wob=0.4, soft=2, grad=s.vgrad(pj(0, H + 11, d1)[1], pj(0, H, d1)[1]))
    base(s, top, CREAM, 0.30, wob=0.5, edge=0.4)
    s.wash(top, SLATE, 0.25, wob=0.4, soft=4)
    for X in (X0 + 20, X1 - 20):
        bx, by = pj(X, H + 11, (d0 + d1) / 2)
        rx, ry = 72, 9
        s.wash(s.ellipse(bx, by, rx, ry), PAYNE, 0.8, wob=0.4, soft=1, edge=0.6)
        s.wash(s.ellipse(bx, by, rx * 0.5, ry * 0.5), INK, 1.0, wob=0.3, soft=0.8)
        oval(s, bx, by, rx, ry, 1.4, INK, 1.1)
        for a in (0.3, 1.9, 3.5, 5.0):                 # pan supports
            s.ink([(bx + math.cos(a) * rx * 0.55, by + math.sin(a) * ry * 0.55 - 6), (bx + math.cos(a) * rx * 0.95, by + math.sin(a) * ry * 0.95 - 4)], 4.0, INK, 1.3)
    for X in (X0 + 20, X1 - 20):
        kx, ky = pj(X, H + 5.5, d1)
        kn = s.ellipse(kx, ky, 14, 14)
        base(s, kn, INK, 1.2, wob=0.3, edge=0.4)
        s.lift(s.ellipse(kx - 4, ky - 5, 4, 3), 0.6, soft=1)
    edges(s, [(X0, H, d1), (X1, H, d1), (X1, H + 11, d1), (X0, H + 11, d1)], 2.0, 1.3, r, over=2)
    edges(s, [(X0, H + 11, d0), (X1, H + 11, d0), (X1, H + 11, d1), (X0, H + 11, d1)], 1.6, 1.0, r, over=2, skip=0.2)
    hz = [pj(X1, H + 4, 26), pj(X1 + 7, H + 1.5, 22), pj(X1 + 10, H + 1.5, 8), pj(X1 + 9, H + 8, 0.5), pj(X1 + 8, H + 16, 0)]
    vx, vy = pj(X1 + 8, H + 18, 0)
    s.ink([(vx, vy), (vx, vy - 60), (vx + 300, vy - 60)], 7.0, SLATE, 1.0, taper=(0, 0))                    # the gas pipe on the wall
    vm = s.rect(vx - 10, vy - 14, vx + 10, vy + 6)
    base(s, vm, OCHRE, 0.9, wob=0.3)
    contour(s, vm, 1.2, INK, 1.0)
    hs = s.ribbon(hz, 10)
    base(s, hs, PERSIMMON, 0.8, wob=0.6, edge=0.4)
    contour(s, hs, 1.3, INK, 1.0)
    # the kettle: aluminium, a black handle arching over, the spout to the right
    kx, ky = pj(KETTLE[0], H + 14, KETTLE[1])
    w2, hh = 70, 92
    body = s.poly([(kx - w2, ky), (kx - w2 - 4, ky - 30), (kx - w2 + 14, ky - 62), (kx - 30, ky - hh), (kx + 30, ky - hh), (kx + w2 - 14, ky - 62), (kx + w2 + 4, ky - 30), (kx + w2, ky)])
    spout = s.ribbon([(kx + w2 - 10, ky - 30), (kx + w2 + 30, ky - 56), (kx + w2 + 50, ky - 84)], 22, profile=lambda t: 1 - 0.45 * t)
    KSPOUT = (kx + w2 + 54, ky - 90)
    cast(s, s.ellipse(kx, ky, w2, 14), 10, 2, UMBER, 0.6, blur=5)
    km = np.maximum(body, spout)
    base(s, km, MIST, 0.65, wob=0.6, edge=0.5)
    s.wash(km, SLATE, 0.55, wob=0.6, soft=4, grad=s.hgrad(kx - 20, kx + w2 + 30))
    s.wash(km, PAYNE, 0.35, wob=0.6, soft=3, grad=s.vgrad(ky - 20, ky))
    s.lift(s.ellipse(kx - 34, ky - 50, 10, 26, 20), 0.85, soft=3)                      # the window, small, in the metal
    s.wash(s.ellipse(kx + 10, ky - 40, 16, 14) * body, OCHRE, 0.25, wob=1, soft=4)
    lid = s.ellipse(kx, ky - hh + 2, 34, 8)
    base(s, lid, MIST, 0.6, wob=0.4, edge=0.4)
    knob = s.ellipse(kx, ky - hh - 6, 10, 7)
    base(s, knob, INK, 1.2, wob=0.3)
    hd = s.ribbon([(kx - w2 + 20, ky - 64), (kx - 50, ky - 132), (kx, ky - 150), (kx + 50, ky - 132), (kx + w2 - 20, ky - 64)], 10)
    base(s, hd, INK, 1.25, wob=0.4, edge=0.4)
    s.lift(s.ribbon([(kx - 40, ky - 136), (kx, ky - 148), (kx + 30, ky - 142)], 3), 0.45, soft=1)
    contour(s, km, 2.0, INK, 1.3)
    oval(s, kx, ky - hh + 2, 34, 8, 1.4, INK, 1.1)


def cooker(s, r):
    """An old electric rice cooker (the kind every flat has), cream with a flower band and a terracotta knob."""
    cx, cy = pj(364, CTR[2], 30)
    w2, hh = 82, 118
    body = s.poly([(cx - w2, cy), (cx - w2 - 2, cy - hh + 14), (cx - w2 + 10, cy - hh), (cx + w2 - 10, cy - hh), (cx + w2 + 2, cy - hh + 14), (cx + w2, cy)])
    cast(s, s.ellipse(cx, cy, w2, 12), 18, 2, UMBER, 0.6, blur=6)
    base(s, body, CREAM, 0.45, wob=0.6, edge=0.4)
    s.wash(body, SLATE, 0.30, wob=0.5, soft=4, grad=s.hgrad(cx - 10, cx + w2))
    s.lift(s.rect(cx - w2 + 16, cy - hh + 20, cx - w2 + 30, cy - 12), 0.4, soft=3)
    bd = s.rect(cx - w2, cy - 52, cx + w2, cy - 34)
    s.wash(bd, CELADON, 0.9, wob=0.5, soft=0.6, edge=0.4)
    for i in range(6):
        fx = cx - w2 + 14 + i * 27
        s.wash(s.ellipse(fx, cy - 43, 6, 5), TERRA, 1.0, wob=0.4)
    lid = s.ellipse(cx, cy - hh, w2 - 4, 14)
    base(s, lid, MIST, 0.55, wob=0.4, edge=0.4)
    s.wash(lid, SLATE, 0.3, wob=0.3, soft=2, grad=s.hgrad(cx - w2, cx + w2))
    kn = s.ellipse(cx, cy - hh - 8, 14, 8)
    base(s, kn, TERRA, 1.0, wob=0.3)
    oval(s, cx, cy - hh - 8, 14, 8, 1.2, INK, 1.0)
    sw = s.rect(cx - 10, cy - 26, cx + 10, cy - 10)
    base(s, sw, NOTE, 0.9, wob=0.3)
    contour(s, body, 2.0, INK, 1.3)
    oval(s, cx, cy - hh, w2 - 4, 14, 1.5, INK, 1.1)
    s.ink([(cx + w2, cy - 70), (cx + w2 + 30, cy - 40), (cx + w2 + 36, cy + 6), (cx + w2 + 70, cy + 4)], 2.0, INK, 1.1)    # the cord


def table(s, r):
    """The dining table's near corner with the checked cloth, and its two chairs."""
    X0, X1, d0, d1, H = TABLE
    # the far chair: between the table and the wall, its back toward the wall, facing us
    chair(s, r, 556, 600, 26, 70, facing="front")
    # the near-left chair, side on, facing the table
    chair(s, r, 446, 488, 104, 150, facing="right")
    # legs under the cloth
    for (X, d) in ((X0 + 4, d1 - 4), (X0 + 4, d0 + 4)):
        a, b = pj(X, 0, d), pj(X, 56, d)
        lw = 26 * DIST / (DIST - d) * 0.5
        lg = s.rect(a[0] - lw / 2, b[1], a[0] + lw / 2, a[1])
        base(s, lg, BURNT, 0.9, wob=0.6, edge=0.4)
        s.wash(lg, SEPIA, 0.4, wob=0.5, soft=2, grad=s.hgrad(a[0] - lw / 2, a[0] + lw / 2))
        contour(s, lg, 1.6, INK, 1.2)
    cast(s, quad(s, [(X0, 0, d0), (X1, 0, d0), (X1, 0, d1), (X0, 0, d1)]), 0, 0, UMBER, 0.40, blur=30)
    # the cloth: top, and the drop on the two near sides (front and left), checked
    top = quad(s, [(X0, H, d0), (X1, H, d0), (X1, H, d1), (X0, H, d1)])
    dl = quad(s, [(X0, H, d0), (X0, H, d1), (X0, H - 20, d1 + 2), (X0, H - 20, d0)])
    df = quad(s, [(X0, H, d1), (X1, H, d1), (X1, H - 20, d1 + 2), (X0, H - 20, d1 + 2)])
    cl = np.maximum(np.maximum(top, dl), df)
    base(s, cl, CREAM, 0.35, wob=1, edge=0.4)
    def checks(m, pts_fn, n_u, n_v):
        """Gingham: bands along u and along v, crossing darker; pts_fn(u, v) -> world point."""
        for i in range(n_u):
            if i % 2:
                continue
            P = [pj(*pts_fn(i, 0)), pj(*pts_fn(i + 1, 0)), pj(*pts_fn(i + 1, n_v)), pj(*pts_fn(i, n_v))]
            s.wash(Q(s, P) * m, CELADON, 0.75, wob=0.6, soft=0.8, edge=0.5, gran=0.4)
        for j in range(n_v):
            if j % 2:
                continue
            P = [pj(*pts_fn(0, j)), pj(*pts_fn(n_u, j)), pj(*pts_fn(n_u, j + 1)), pj(*pts_fn(0, j + 1))]
            s.wash(Q(s, P) * m, TILE, 0.60, wob=0.6, soft=0.8, edge=0.5, gran=0.4)
    cs = 5.0
    checks(top, lambda u, v: (X0 + u * cs, H, d0 + v * cs), int((X1 - X0) / cs) + 1, int((d1 - d0) / cs) + 1)
    checks(dl, lambda u, v: (X0, H - v * cs * 0.6, d0 + u * cs), int((d1 - d0) / cs) + 1, 4)
    checks(df, lambda u, v: (X0 + u * cs, H - v * cs * 0.6, d1 + 1), int((X1 - X0) / cs) + 1, 4)
    s.wash(dl, UMBER, 0.35, wob=1, soft=3)
    s.wash(df, UMBER, 0.15, wob=1, soft=3)
    s.lift(top * s.hgrad(2400, 3072, 0.5, 0), 0.15, soft=10)
    edges(s, [(X0, H, d0), (X1, H, d0)], 1.8, 1.2, r, closed=False)
    edges(s, [(X0, H, d1), (X0, H, d0), (X0, H - 20, d0), (X0, H - 20, d1 + 2), (X1, H - 20, d1 + 2)], 2.2, 1.4, r, closed=False)
    edges(s, [(X0, H, d1), (X1, H, d1)], 1.6, 1.0, r, closed=False, skip=0.3)
    # on the table: the bowl of oranges? no: a teapot-less morning, only a small sugar jar and a folded newspaper
    nx0, ny0 = pj(X0 + 40, H, d0 + 40)
    nx1, ny1 = pj(X0 + 90, H, d0 + 70)
    paper = Q(s, [(nx0, ny0), (nx1, ny0 - 6), (nx1 + 30, ny1), (nx0 + 20, ny1 + 6)])
    cast(s, paper, 6, 6, UMBER, 0.35, blur=5)
    base(s, paper, CREAM, 0.12, wob=0.6, edge=0.4)
    for i in range(6):
        y = ny0 + 10 + i * ((ny1 - ny0 - 10) / 6)
        s.ink([(nx0 + 10 + i * 3, y), (nx1 + i * 4, y - 3)], 1.0, SLATE, 0.7, skip=0.4, tremor=0.3)
    contour(s, paper, 1.4, INK, 1.0)


def chair(s, r, X0, X1, d0, d1, facing="front"):
    """A plain wooden chair: four legs, a seat at 45 cm, a back with two slats. 'front' has its back toward the wall;
    'right' stands side on with its back at X0."""
    SEATH = 45
    parts = []
    if facing == "front":
        back = [(X0, SEATH, d0), (X1, SEATH, d0), (X1, 92, d0), (X0, 92, d0)]
        legs = [(X0 + 2, d0 + 2), (X1 - 2, d0 + 2), (X0 + 2, d1 - 2), (X1 - 2, d1 - 2)]
    else:
        back = [(X0, SEATH, d0), (X0, SEATH, d1), (X0, 92, d1), (X0, 92, d0)]
        legs = [(X0 + 2, d0 + 2), (X1 - 2, d0 + 2), (X0 + 2, d1 - 2), (X1 - 2, d1 - 2)]
    # far legs first, then the back, the seat, the near legs
    for (X, d) in sorted(legs, key=lambda t: t[1]):
        a, b = pj(X, 0, d), pj(X, SEATH, d)
        lw = 4.0 * PX * DIST / (DIST - d) * 0.9
        lg = s.rect(a[0] - lw / 2, b[1], a[0] + lw / 2, a[1])
        base(s, lg, BURNT, 0.85, wob=0.5, edge=0.4)
        s.wash(lg, SEPIA, 0.35, wob=0.4, soft=1.5, grad=s.hgrad(a[0] - lw / 2, a[0] + lw / 2))
        contour(s, lg, 1.5, INK, 1.2)
    # rails between the legs
    for (p, q) in ((legs[0], legs[1]), (legs[0], legs[2]), (legs[1], legs[3])):
        s.ink([pj(p[0], 14, p[1]), pj(q[0], 14, q[1])], 4.0, SEPIA, 1.1)
    seat = quad(s, [(X0, SEATH, d0), (X1, SEATH, d0), (X1, SEATH, d1), (X0, SEATH, d1)])
    seatf = quad(s, [(X0, SEATH, d1), (X1, SEATH, d1), (X1, SEATH - 3, d1), (X0, SEATH - 3, d1)])
    cast(s, seat, 0, 30, UMBER, 0.3, blur=20)
    wood(s, seat, SIENNA, BURNT, 0.8, ang=0.1, n=10, seed=int(X0), wob=0.6)
    base(s, seatf, BURNT, 0.9, wob=0.4)
    edges(s, [(X0, SEATH, d0), (X1, SEATH, d0), (X1, SEATH, d1), (X0, SEATH, d1)], 1.8, 1.3, r, over=2)
    # the back: two posts, a top rail, a middle slat
    if facing == "front":
        posts = [(X0 + 2, d0 + 1), (X1 - 2, d0 + 1)]
        rail = [(X0, 92, d0 + 1), (X1, 92, d0 + 1), (X1, 82, d0 + 1), (X0, 82, d0 + 1)]
        slat = [(X0 + 3, 70, d0 + 1), (X1 - 3, 70, d0 + 1), (X1 - 3, 64, d0 + 1), (X0 + 3, 64, d0 + 1)]
    else:
        posts = [(X0 + 2, d0 + 2), (X0 + 2, d1 - 2)]
        rail = [(X0 + 1, 92, d0), (X0 + 1, 92, d1), (X0 + 1, 82, d1), (X0 + 1, 82, d0)]
        slat = [(X0 + 1, 70, d0 + 3), (X0 + 1, 70, d1 - 3), (X0 + 1, 64, d1 - 3), (X0 + 1, 64, d0 + 3)]
    for (X, d) in posts:
        a, b = pj(X, SEATH, d), pj(X, 92, d)
        lw = 4.0 * PX * DIST / (DIST - d) * 0.9
        pm = s.rect(a[0] - lw / 2, b[1], a[0] + lw / 2, a[1])
        base(s, pm, BURNT, 0.85, wob=0.5, edge=0.4)
        contour(s, pm, 1.5, INK, 1.2)
    for part in (rail, slat):
        m = quad(s, part)
        wood(s, m, SIENNA, BURNT, 0.85, n=5, seed=int(X0) + 3, wob=0.5, lines=False)
        s.wash(m, UMBER, 0.2, wob=0.4, soft=2)
        edges(s, part, 1.6, 1.2, r, over=2)


ALL = {"kt_bg": kitchen}
