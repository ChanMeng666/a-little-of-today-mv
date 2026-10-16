"""The fridge door, close: an old cream enamel fridge in the morning. The yellow notes, the hands that put them there.
Sheet 2560x1440. Light comes from the kitchen window, off to the left.

Layers
  fr_bg        opaque 2560x1440. The door fills x 70..2270 (main door from y 150 down; the freezer door's foot above it),
               the chrome handle at x ~140..200, tiles past x 2300. Things already on the door keep to the edges; the
               region x 850..1750, y 300..1150 is clean for the notes.
               meta note_main  [x, y, rot]  anchor (top centre) of the first note, rot in radians (+ = clockwise on
                               screen, as canvas rotate)
                    note_spots [[x, y, rot], ...] 14 more notes for "记了好多天", in the order to stick them, laid out
                               for notes drawn at note_spot_scale: they overlap each other and the menu/calendar a
                               little, never the main note
                    note_spot_scale  0.62 (a 267 px note)
                    handle     [x0, y0, x1, y1] the chrome handle
                    door       [x0, y0, x1, y1] the main door's face
  fr_note      sprite, a blank yellow sticky note 430 px square, anchor = top centre of the note.
               meta write [x0, y0, x1, y1] the area for handwriting, relative to the anchor
                    size  430
  fr_note_pink / fr_note_mint / fr_note_blue   the same note in other colours (same anchor, same meta)
  fr_note_sh   the note's soft cast shadow on the door (same anchor): draw it under the note
  fr_hand_you  your right hand (navy jacket) from the lower right, fingers flat pressing a note to the door, back of
               the hand to us. Anchor = centre of the fingertips' contact. meta cuff [dx, dy] centre of the cuff,
               arm [dx, dy] unit vector pointing from the fingers back along the arm (slide the hand along it)
  fr_hand_me   my right hand (cream knit cardigan) from the lower right, writing with a black felt-tip pen.
               Anchor = the pen tip. meta cuff [dx, dy], arm [dx, dy] as above
The hand painters (dorsal_hand, writing_hand) and tile_wall are shared with s_table and s_kitchen.
"""
import math

import cv2
import numpy as np

from kit import *

SW, SH = 2560, 1440
DL, DR = 70, 2270            # the door's left and right edges
SEAM = 150                   # top of the main door (the freezer door ends just above)
HX, HY0, HY1 = 172, 300, 1270   # chrome handle: centre x, top, bottom
TX = 2300                    # where the wall tiles start
NOTE = (0.98, 0.88, 0.42)
NOTE_MAIN = [1300, 470, -0.035]
NOTE_SPOTS = [[560, 200, -0.06], [1840, 250, 0.07], [850, 520, 0.05], [1700, 560, -0.08], [1300, 960, 0.03], [440, 640, 0.09],
              [2080, 470, -0.05], [1000, 190, -0.03], [1640, 980, 0.10], [620, 1010, -0.08], [1950, 820, 0.04], [960, 980, -0.11],
              [1320, 175, 0.06], [2050, 1080, -0.04]]
SPOT_SCALE = 0.62            # the spots are laid out for notes at this scale (they overlap a little, never the main note)
FONT_P = "D:/github_repository/a-little-of-today-mv/assets/fonts/NotoSerifSC-VF.ttf"
FONT_H = "D:/github_repository/a-little-of-today-mv/assets/fonts/LXGWWenKai-Regular.ttf"
NS = 430                     # note size


def rr(x0, y0, x1, y1, rad, n=7, step=30.0):
    """Points round a rounded rectangle, evenly spaced (a spline through them must not overshoot the long sides).
    Starts at the top edge's right end and goes clockwise, `n` points per corner."""
    P = []
    for cx, cy, a0 in ((x1 - rad, y0 + rad, -math.pi / 2), (x1 - rad, y1 - rad, 0), (x0 + rad, y1 - rad, math.pi / 2), (x0 + rad, y0 + rad, math.pi)):
        if P:
            a, b = np.array(P[-1]), np.array((cx + math.cos(a0) * rad, cy + math.sin(a0) * rad))
            k = int(np.hypot(*(b - a)) / step)
            P += [tuple(a + (b - a) * (i / (k + 1))) for i in range(1, k + 1)]
        for t in np.linspace(0, math.pi / 2, n):
            P.append((cx + math.cos(a0 + t) * rad, cy + math.sin(a0 + t) * rad))
    return P


def dense(pts, step=30.0):
    """Even out the spacing of a polyline (Catmull-Rom overshoots where long and short spans meet)."""
    P = np.asarray(pts, np.float64)
    return resample(P, step) if len(P) > 1 else P


def arc(cx, cy, rad, a0, a1, n=8):
    return [(cx + math.cos(a) * rad, cy + math.sin(a) * rad) for a in np.linspace(a0, a1, n)]


def rotp(cx, cy, deg, x, y):
    """Rotate (x, y) about (cx, cy) the way cv2.getRotationMatrix2D (and so kit.text_mask) does."""
    a, b = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    dx, dy = x - cx, y - cy
    return (cx + a * dx + b * dy, cy - b * dx + a * dy)


def rot(pts, ang, ox=0.0, oy=0.0):
    """Rotate local points by `ang` radians (clockwise on screen for ang > 0) and move them to (ox, oy)."""
    c, s_ = math.cos(ang), math.sin(ang)
    return [(ox + x * c - y * s_, oy + x * s_ + y * c) for x, y in pts]


def dome(s, cx, cy, rad, col, k=1.0, light=(-0.4, -0.5)):
    """A round glossy fridge magnet seen from the front."""
    m = s.ellipse(cx, cy, rad, rad)
    cast(s, m, rad * 0.22, rad * 0.30, UMBER, 0.55, blur=rad * 0.22)
    base(s, m, col, k, wob=0.8, edge=0.6, ew=3, gran=0.3)
    s.wash(m, SEPIA, 0.25, wob=0.5, soft=2, grad=s.rgrad(cx - light[0] * rad * 0.6, cy - light[1] * rad * 0.6, rad * 0.3, rad * 1.1, 0, 1))
    s.lift(s.ellipse(cx + light[0] * rad * 0.45, cy + light[1] * rad * 0.45, rad * 0.28, rad * 0.16, -35), 0.85, soft=2.5, wob=0.6)
    oval(s, cx, cy, rad, rad, 2.0, INK, 1.3)
    return m


def paper_slip(s, pts, k=0.10, col=CREAM, clip=None):
    """A sheet of white paper laid on the door: clear to paper, a whisper of colour, a soft cool fall-off."""
    m = Q(s, pts)
    base(s, m, col, k, wob=0.8, edge=0.35, flow=0.3, gran=0.3)
    return m


def door_bg():
    s = Sheet(SW, SH, 201)
    r = rs(201)
    # what is beside the fridge, in shade: the cabinet's own frame, then the gap
    s.glaze(UMBER, 0.55, flow=0.4, fs=200)
    s.wash(s.rect(-10, -10, 40, SH + 10), SEPIA, 0.9, wob=2, bleed=10, edge=0.2)
    s.wash(s.rect(40, -10, DL + 4, SH + 10), CREAM, 0.5, wob=1, edge=0.3)
    s.wash(s.rect(40, -10, DL + 4, SH + 10), SLATE, 0.30, wob=1, soft=4, grad=s.hgrad(40, DL + 4, 0.2, 1))
    # the wall tiles at the right
    tiles(s, r)
    # the fridge's own shadow on the tiles, and the dark gap between the door and the wall
    gap = s.rect(DR - 10, -10, TX + 4, SH + 10)
    s.wash(gap, PAYNE, 0.85, wob=1.5, bleed=4, edge=0.4)
    s.wash(gap, SEPIA, 0.35, wob=1, soft=3, grad=s.hgrad(DR, TX, 0, 1))
    s.wash(s.rect(TX, -10, SW + 10, SH + 10), PAYNE, 0.42, wob=0, edge=0, grad=s.hgrad(TX, TX + 110, 1, 0) * (0.75 + 0.25 * s.noise(90, 2)), gran=0.4)
    # the door and the freezer door's foot
    door = Q(s, rr(DL, SEAM, DR, SH + 80, 52))
    frz = Q(s, rr(DL, -80, DR, SEAM - 18, 52))
    face = np.maximum(door, frz)
    base(s, face, CREAM, 0.50, wob=1.0, edge=0.25, flow=0.35, fs=380, gran=0.3)
    s.wash(face, OCHRE, 0.20, wob=0, edge=0, grad=s.rgrad(520, 260, 150, 1700, 1, 0, sy=0.8), flow=0.4, fs=300)       # the window's warmth
    s.wash(face, RAIN, 0.26, wob=0, edge=0, grad=s.hgrad(1750, DR - 10, 0, 1) ** 1.6, flow=0.35, gran=0.4)            # the face bulges away at the right
    s.wash(face, MIST, 0.32, wob=0, edge=0, grad=s.hgrad(DL + 150, DL, 0, 1), gran=0.4)                               # and at the left
    s.wash(face, SLATE, 0.12, wob=0, edge=0, grad=s.vgrad(800, SH, 0, 1), flow=0.4)
    s.wash(frz, RAIN, 0.18, wob=0, edge=0)
    for _ in range(9):                    # slow blooms where the wash dried unevenly
        bx, by = 200 + r.random() * 2000, 200 + r.random() * 1200
        s.wash(s.blob(bx, by, 160 + 220 * r.random(), 90 + 120 * r.random(), 0.3) * face, [OCHRE, MIST, CREAM][int(r.random() * 3)], 0.10, wob=6, bleed=30, edge=0.6, ew=12)
    # enamel sheen: a soft vertical gleam, a second thinner one, and the bevel along the edges
    s.lift(s.rect(420, SEAM + 40, 560, SH + 10) * door, 0.30, soft=38, wob=10)
    s.lift(s.rect(640, SEAM + 80, 668, SH + 10) * door, 0.18, soft=10, wob=6)
    s.wash(door * (1 - Q(s, rr(DL + 16, SEAM + 14, DR - 22, SH + 80, 40))), SLATE, 0.45, wob=1, soft=5, edge=0.3)
    s.wash(frz * (1 - Q(s, rr(DL + 16, -80, DR - 22, SEAM - 34, 40))), SLATE, 0.45, wob=1, soft=5, edge=0.3)
    # the seam: a dark gap, the grey gasket lip just under it
    s.wash(s.rect(DL + 20, SEAM - 20, DR - 20, SEAM + 2) * (1 - door), SEPIA, 1.1, wob=1, soft=1.5, edge=0.3)
    s.wash(s.rect(DL + 40, SEAM + 2, DR - 40, SEAM + 14) * door, SLATE, 0.55, wob=1, soft=2)
    cast(s, s.rect(DL + 40, SEAM - 20, DR - 40, SEAM + 2), 0, 14, UMBER, 0.35, blur=9, clip=door)
    # the gasket along the right edge
    s.wash(s.rect(DR - 20, SEAM + 40, DR - 2, SH + 10), PAYNE, 0.35, wob=1, soft=2)
    # wear: a few small chips with a rust ring, and the grime of hands round the handle
    for (cx, cy, rad) in ((300, 1340, 7), (2190, 1180, 5), (960, 1390, 4), (2120, 210, 6), (1630, 1360, 4), (110, 840, 5)):
        s.wash(s.blob(cx, cy, rad * 2.2, rad * 1.8, 0.4), BURNT, 0.4, wob=1, soft=1, edge=0.6)
        s.wash(s.blob(cx, cy, rad, rad * 0.8, 0.3), SEPIA, 1.2, wob=0.6, soft=0.6, edge=0.4)
    s.wash(s.ellipse(HX + 80, (HY0 + HY1) / 2, 150, 520) * door, SIENNA, 0.10, wob=6, bleed=40, edge=0.2)
    # pen: the door's outline (each edge its own stroke), the freezer door, the seam
    q = 52
    s.ink(dense([(DL, SEAM + 260)] + arc(DL + q, SEAM + q, q, math.pi, 1.5 * math.pi) + [(DL + 700, SEAM - 1)], 20), 3.0, INK, 1.5, taper=(0.04, 0.0))
    s.ink(dense([(DL + 620, SEAM + 1), (DR - q, SEAM)] + arc(DR - q, SEAM + q, q, -0.5 * math.pi, 0) + [(DR + 1, SEAM + 300)], 20), 3.0, INK, 1.5, taper=(0.0, 0.08))
    s.ink([(DR, SEAM + 200), (DR + 1, 800), (DR - 1, SH + 20)], 3.0, INK, 1.5)
    s.ink([(DL, SH + 20), (DL - 1, 800), (DL, SEAM + 160)], 3.0, INK, 1.5)
    yb = SEAM - 18
    s.ink(dense([(DR + 1, -20)] + arc(DR - q, yb - q, q, 0, 0.5 * math.pi) + [(DR - 900, yb + 1)], 20), 2.8, INK, 1.5, taper=(0.0, 0.05))
    s.ink(dense([(DR - 980, yb)] + arc(DL + q, yb - q, q, 0.5 * math.pi, math.pi) + [(DL - 1, -20)], 20), 2.8, INK, 1.5, taper=(0.05, 0.0))
    seg(s, (DR - 20, SEAM + 50), (DR - 18, SH + 10), 1.4, SEPIA, 0.9, skip=0.25, r=r)
    seg(s, (DL + 40, SEAM + 10), (DR - 40, SEAM + 12), 1.4, SEPIA, 0.8, skip=0.3, r=r)
    seg(s, (DL + 18, SEAM + 60), (DL + 19, SH + 10), 1.3, SEPIA, 0.7, skip=0.35, r=r)
    handle(s, r, door, frz)
    things(s, r, door)
    # a last cool glaze toward the bottom of the frame, warm at the top left: the morning
    s.wash(s.full(), UMBER, 0.08, wob=0, soft=0, edge=0, gran=0.3, flow=0.4, grad=s.rgrad(1100, 600, 900, 1800, 0, 1, sy=0.7))
    s.save("fr_bg", opaque=True, q=90, meta={"note_main": NOTE_MAIN, "note_spots": NOTE_SPOTS, "note_spot_scale": SPOT_SCALE, "handle": [HX - 40, HY0 - 40, HX + 40, HY1 + 40],
                                              "door": [DL, SEAM, DR, SH]})


def tiles(s, r):
    """Small square wall tiles, pale celadon, hand-ruled grout."""
    tile_wall(s, r, TX, -10, SW + 10, SH + 10, 74, ox=TX + 6, oy=18)


def tile_wall(s, r, X0, Y0, X1, Y1, T, ox=0, oy=0, clip=None, k=1.0, grout=0.75):
    """Small square glazed tiles, pale celadon and white, each its own little wash; grout left as paper, ruled by hand."""
    wall = s.rect(X0, Y0, X1, Y1) if clip is None else clip
    s.lift(wall, 1.0, soft=0.6, wob=0)
    s.wash(wall, MIST, 0.22 * k, wob=0, edge=0, flow=0.3, gran=0.5)                         # the grout
    groups = [np.zeros((s.h, s.w), np.uint8) for _ in range(4)]
    shine = np.zeros((s.h, s.w), np.uint8)
    i0, i1 = int((X0 - ox) // T) - 1, int((X1 - ox) // T) + 1
    j0, j1 = int((Y0 - oy) // T) - 1, int((Y1 - oy) // T) + 1
    for j in range(j0, j1 + 1):
        for i in range(i0, i1 + 1):
            x0, y0 = ox + i * T, oy + j * T
            g = min(3, int(r.random() * 4.6))
            cv2.rectangle(groups[g], (int(x0 + 4 + r.random() * 2), int(y0 + 4 + r.random() * 2)), (int(x0 + T - 4 - r.random() * 2), int(y0 + T - 4 - r.random() * 2)), 255, -1)
            cv2.ellipse(shine, (int(x0 + T * 0.32), int(y0 + T * 0.28)), (int(T * 0.2), int(T * 0.09)), -32, 0, 360, 255, -1)
    for g, (col, kk) in zip(groups, ((CELADON, 0.55), (TILE, 0.42), (CELADON, 0.32), (CREAM, 0.22))):
        s.wash(g.astype(np.float32) / 255 * wall, col, kk * k, wob=0.8, soft=1.2, edge=0.75, ew=3, gran=0.45, flow=0.3, fs=40)
    s.lift(cv2.GaussianBlur(shine.astype(np.float32) / 255, (0, 0), T * 0.05) * wall, 0.45, soft=0, wob=1)
    mm = cv2.GaussianBlur(wall, (0, 0), 1.0)
    for i in range(i0, i1 + 1):
        x = ox + i * T
        if X0 - 4 < x < X1 + 4:
            seg(s, (x, Y0), (x + r.random() * 4 - 2, Y1), 1.4, SLATE, grout, bow=0.002, skip=0.3, r=r, clip=mm)
    for j in range(j0, j1 + 1):
        y = oy + j * T
        if Y0 - 4 < y < Y1 + 4:
            seg(s, (X0, y), (X1, y + r.random() * 3 - 1.5), 1.4, SLATE, grout, skip=0.3, r=r, clip=mm)


def handle(s, r, door, frz):
    """The long chrome pull: a bar on two feet, standing off the door."""
    x = HX
    feet = [s.poly(rr(x - 38, y - 34, x + 38, y + 34, 14), smooth=False) for y in (HY0, HY1)]
    bar = Q(s, [(x - 22, HY0 - 10), (x + 22, HY0 - 10), (x + 22, HY1 + 10), (x - 22, HY1 + 10)])
    allm = np.maximum(bar, np.maximum(*feet))
    cast(s, allm, 30, 22, UMBER, 0.55, blur=16, clip=door)
    for m in feet:
        base(s, m, MIST, 0.6, wob=0.6, edge=0.4)
        s.wash(m, SLATE, 0.6, wob=0.5, soft=2, grad=s.hgrad(x - 30, x + 40))
    base(s, bar, MIST, 0.55, wob=0.5, edge=0.3)
    s.wash(s.rect(x - 22, HY0, x - 8, HY1) * bar, PAYNE, 0.9, wob=0.8, soft=3)              # the room, dark, reflected down the left side
    s.wash(s.rect(x + 6, HY0, x + 22, HY1) * bar, SLATE, 0.7, wob=0.8, soft=3)
    s.wash(s.rect(x - 8, HY0, x + 4, HY1) * bar, OCHRE, 0.35, wob=0.6, soft=2)               # the window, warm, a thin line
    s.lift(s.rect(x - 4, HY0 + 30, x + 2, HY1 - 40), 0.9, soft=1.2, wob=1)
    for y in (HY0, HY1):
        path(s, rr(x - 38, y - 34, x + 38, y + 34, 14)[::3], 2.0, INK, 1.4, closed=True, over=1.5, r=r)
    s.ink([(x - 22, HY0 + 30), (x - 23, (HY0 + HY1) / 2), (x - 22, HY1 - 30)], 2.4, INK, 1.5)
    s.ink([(x + 22, HY0 + 30), (x + 23, (HY0 + HY1) / 2), (x + 22, HY1 - 30)], 2.4, INK, 1.5)
    # the freezer's handle, just its foot, above the seam
    fm = s.poly(rr(x - 38, -40, x + 38, 96, 14), smooth=False)
    cast(s, fm, 30, 18, UMBER, 0.5, blur=14, clip=frz)
    base(s, fm, MIST, 0.55, wob=0.6, edge=0.4)
    s.wash(fm, SLATE, 0.6, wob=0.5, soft=2, grad=s.hgrad(x - 30, x + 40))
    s.lift(s.rect(x - 6, -10, x, 80), 0.8, soft=1.5)
    path(s, rr(x - 38, -40, x + 38, 96, 14)[2::3], 2.0, INK, 1.4, closed=False, over=1.5, r=r)


def things(s, r, door):
    """What is already on the door: a takeout menu, a strawberry magnet, three round magnets and a receipt, a little
    calendar under a strip of tape, the gas man's magnet."""
    # --- the takeout menu, held by a red magnet
    cx, cy, deg = 520, 960, 2.4
    w2, h2 = 205, 345
    corners = [rotp(cx, cy, deg, cx + dx, cy + dy) for dx, dy in ((-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2))]
    sh = Q(s, corners)
    cast(s, sh, 12, 16, UMBER, 0.45, blur=10, clip=door)
    menu = paper_slip(s, corners, 0.10)
    s.wash(menu, RAIN, 0.16, wob=1, bleed=20, grad=s.hgrad(cx - 60, cx + w2, 0, 1))
    head = Q(s, [rotp(cx, cy, deg, cx + dx, cy + dy) for dx, dy in ((-w2 + 18, -h2 + 22), (w2 - 18, -h2 + 22), (w2 - 18, -h2 + 128), (-w2 + 18, -h2 + 128))])
    s.wash(head, TERRA, 0.85, wob=1.2, edge=0.5, gran=0.5, flow=0.3)
    s.wash(head, ROSE, 0.35, wob=1, soft=3)
    hx, hy = rotp(cx, cy, deg, cx, cy - h2 + 72)
    s.lift(text_mask(s, "巷口麵店", hx, hy, 60, FONT_P, rot=deg), 0.92, soft=0.6, wob=0.4)
    sx, sy = rotp(cx, cy, deg, cx, cy - h2 + 160)
    s.wash(text_mask(s, "營業時間 6:00 - 14:00", sx, sy, 22, FONT_P, rot=deg), INK, 0.9, wob=0.3, soft=0.5, edge=0.2, gran=0.3)
    items = [("牛肉麵", "130"), ("陽春麵", "45"), ("餛飩麵", "70"), ("炸醬麵", "60"), ("酸辣湯餃", "75"), ("滷肉飯", "35"), ("燙青菜", "40"), ("貢丸湯", "35")]
    txt = s.zeros()
    for i, (a, b) in enumerate(items):
        y = cy - h2 + 215 + i * 50
        ax_, ay_ = rotp(cx, cy, deg, cx - w2 + 34, y)
        bx_, by_ = rotp(cx, cy, deg, cx + w2 - 34, y)
        txt = np.maximum(txt, text_mask(s, a, ax_, ay_, 30, FONT_P, anchor="lm", rot=deg))
        txt = np.maximum(txt, text_mask(s, b, bx_, by_, 28, FONT_P, anchor="rm", rot=deg))
        for q in np.arange(cx - w2 + 34 + 30 * len(a) + 14, cx + w2 - 34 - 17 * len(b) - 10, 12):      # dotted leaders
            px_, py_ = rotp(cx, cy, deg, q, y + 8)
            s.wash(s.ellipse(px_, py_, 1.6, 1.6), INK, 0.7, wob=0, soft=0.4, edge=0, gran=0)
    s.wash(txt, INK, 1.0, wob=0.3, soft=0.5, edge=0.2, gran=0.3)
    fx, fy = rotp(cx, cy, deg, cx, cy + h2 - 46)
    s.wash(text_mask(s, "外送電話 2368-0516", fx, fy, 26, FONT_P, rot=deg), TERRA, 1.1, wob=0.3, soft=0.5, edge=0.2)
    f0, f1 = rotp(cx, cy, deg, cx + 30, cy - h2 + 4), rotp(cx, cy, deg, cx + 22, cy + h2 - 4)      # a fold down the middle
    s.ink([f0, ((f0[0] + f1[0]) / 2 + 3, (f0[1] + f1[1]) / 2), f1], 1.2, SLATE, 0.35, skip=0.45)
    s.wash(Q(s, [f0, f1, (f1[0] + 40, f1[1]), (f0[0] + 40, f0[1])]) * menu, RAIN, 0.12, wob=2, bleed=10)
    ring = s.ellipse(cx + 80, cy + 200, 62, 58) * (1 - s.ellipse(cx + 82, cy + 201, 55, 52))
    s.wash(ring * menu, SIENNA, 0.35, wob=2.5, edge=0.8, ew=2, gran=0.6, flow=0.6, fs=30)                          # where a bowl of soup stood
    path(s, corners, 2.2, INK, 1.3, closed=True, over=4, r=r)
    mx, my = rotp(cx, cy, deg, cx - 10, cy - h2 + 24)
    dome(s, mx, my, 34, TERRA, 1.0)
    # --- a ceramic strawberry
    sx, sy = 560, 300
    berry = s.poly([(sx - 46, sy - 30), (sx, sy - 44), (sx + 46, sy - 30), (sx + 40, sy + 20), (sx + 12, sy + 62), (sx - 6, sy + 64), (sx - 38, sy + 24)])
    cast(s, berry, 12, 16, UMBER, 0.5, blur=8, clip=door)
    base(s, berry, ROSE, 1.2, wob=1, edge=0.6, gran=0.4)
    s.wash(berry, PERSIMMON, 0.4, wob=1, soft=2, grad=s.rgrad(sx - 16, sy - 6, 6, 50, 1, 0))
    s.wash(berry, BURNT, 0.45, wob=1, soft=3, grad=s.rgrad(sx + 30, sy + 40, 10, 70, 1, 0))
    for i in range(14):
        a, d = r.random() * 6.28, r.random() * 0.8
        px, py = sx + math.cos(a) * 34 * d, sy + 10 + math.sin(a) * 40 * d
        s.wash(s.ellipse(px, py, 3.2, 4.4, 20), OCHRE, 1.3, wob=0.3, soft=0.4, edge=0.3)
    s.lift(s.ellipse(sx - 20, sy - 18, 12, 6, -30), 0.8, soft=1.5)
    leaf = s.poly([(sx - 50, sy - 30), (sx - 22, sy - 54), (sx, sy - 70), (sx + 20, sy - 54), (sx + 52, sy - 32), (sx + 10, sy - 38), (sx - 8, sy - 36)])
    base(s, leaf, LEAF, 1.1, wob=1, edge=0.5)
    contour(s, berry, 2.0, INK, 1.3)
    contour(s, leaf, 1.7, INK, 1.3)
    # --- three round magnets, the middle one holding a receipt
    rcx, rcy = 1985, 300
    rc = [(rcx - 72, rcy - 20), (rcx + 74, rcy - 26), (rcx + 86, rcy + 300), (rcx - 62, rcy + 306)]
    cast(s, Q(s, rc), 10, 14, UMBER, 0.4, blur=8, clip=door)
    rm = paper_slip(s, rc, 0.06)
    s.wash(rm, RAIN, 0.14, wob=1, bleed=12, grad=s.vgrad(rcy, rcy + 300))
    s.wash(text_mask(s, "電子發票證明聯", rcx + 4, rcy + 38, 18, FONT_P, rot=-2.5), INK, 0.9, wob=0.2, soft=0.4)
    s.wash(text_mask(s, "115年09-10月", rcx + 5, rcy + 66, 22, FONT_P, rot=-2.5), INK, 1.0, wob=0.2, soft=0.4)
    s.wash(text_mask(s, "QK-2741 6083", rcx + 6, rcy + 94, 20, FONT_P, rot=-2.5), INK, 0.9, wob=0.2, soft=0.4)
    for i in range(30):                      # the barcode
        x = rcx - 48 + i * 3.3 + r.random() * 1.2
        if r.random() < 0.7:
            s.ink([(x, rcy + 118 - (x - rcx) * 0.044), (x + 0.6, rcy + 150 - (x - rcx) * 0.044)], 1.2 + r.random() * 1.6, INK, 1.2, taper=(0, 0), tremor=0.15, dry=0)
    for qx in (rcx - 30, rcx + 34):          # two little QR squares
        q = [(qx - 26, rcy + 172), (qx + 26, rcy + 170), (qx + 28, rcy + 222), (qx - 24, rcy + 224)]
        path(s, q, 1.6, INK, 1.2, closed=True, over=1, r=r)
        for _ in range(16):
            px, py = qx - 18 + r.random() * 36, rcy + 178 + r.random() * 38
            s.wash(s.rect(px, py, px + 6, py + 6), INK, 1.0, wob=0.2, soft=0.3, edge=0)
    for i in range(3):
        s.ink([(rcx - 50, rcy + 246 + i * 16), (rcx + 40 - 30 * (i == 2), rcy + 242 + i * 16)], 3, SLATE, 0.6, skip=0.4, tremor=0.3)
    path(s, rc, 1.7, INK, 1.1, closed=True, over=3, r=r)
    dome(s, rcx - 92, rcy - 32, 30, RAIN, 1.1)
    dome(s, rcx, rcy - 10, 34, OCHRE, 1.0)
    dome(s, rcx + 96, rcy - 42, 28, TEAL, 0.8)
    # --- a small calendar under a strip of masking tape
    kx0, ky0, kx1, ky1 = 1846, 760, 2170, 1150
    kc = [(kx0, ky0 + 4), (kx1, ky0), (kx1 + 4, ky1), (kx0 + 2, ky1 + 3)]
    cast(s, Q(s, kc), 10, 14, UMBER, 0.42, blur=9, clip=door)
    km = paper_slip(s, kc, 0.08)
    s.wash(km, RAIN, 0.14, wob=1, bleed=16, grad=s.hgrad(kx0 + 100, kx1, 0, 1))
    s.wash(text_mask(s, "10月", kx0 + 70, ky0 + 66, 44, FONT_H), TERRA, 1.2, wob=0.4, soft=0.6, edge=0.3, gran=0.4)
    s.wash(text_mask(s, "2026", kx1 - 70, ky0 + 70, 26, FONT_P), INK, 0.9, wob=0.3, soft=0.5)
    gx0, gy0, cw, ch = kx0 + 22, ky0 + 116, (kx1 - kx0 - 44) / 7, 46
    for i, d in enumerate("日一二三四五六"):
        s.wash(text_mask(s, d, gx0 + cw * (i + 0.5), gy0 - 16, 18, FONT_P), TERRA if i in (0, 6) else INK, 0.9, wob=0.2, soft=0.4)
    nums = s.zeros()
    for dd in range(1, 32):
        c, rw = (dd + 3) % 7, (dd + 3) // 7
        nums = np.maximum(nums, text_mask(s, str(dd), gx0 + cw * (c + 0.5), gy0 + ch * (rw + 0.5), 20, FONT_P))
    s.wash(nums, INK, 0.85, wob=0.2, soft=0.4, edge=0.1)
    for j in range(6):
        seg(s, (gx0 - 2, gy0 + j * ch), (gx0 + cw * 7 + 2, gy0 + j * ch + 1), 1.0, SLATE, 0.5, skip=0.2, r=r)
    # 16 has a pen square round it; 24 has a small rose circle
    c, rw = (16 + 3) % 7, (16 + 3) // 7
    qx, qy = gx0 + cw * (c + 0.5), gy0 + ch * (rw + 0.5)
    path(s, [(qx - 20, qy - 19), (qx + 21, qy - 20), (qx + 20, qy + 20), (qx - 19, qy + 19)], 2.0, INK, 1.4, closed=True, over=5, r=r)
    c, rw = (24 + 3) % 7, (24 + 3) // 7
    oval(s, gx0 + cw * (c + 0.5), gy0 + ch * (rw + 0.5), 22, 18, 2.2, ROSE, 1.3)
    path(s, kc, 1.8, INK, 1.2, closed=True, over=3, r=r)
    tape = Q(s, [(kx0 + 90, ky0 - 34), (kx1 - 80, ky0 - 40), (kx1 - 78, ky0 + 22), (kx0 + 92, ky0 + 28)])
    s.wash(tape, OCHRE, 0.32, wob=2.2, edge=0.7, ew=2, gran=0.6, flow=0.4, fs=40)
    s.wash(tape, CREAM, 0.3, wob=2.2, edge=0.2)
    for x in (kx0 + 91, kx1 - 79):           # torn ends
        s.ink([(x + r.random() * 4, ky0 - 36 + i * 9 + (r.random() - 0.5) * 6) for i in range(8)], 1.2, SEPIA, 0.7, smooth=True, tremor=1.5)
    # an old ghost of tape that was here before
    s.wash(Q(s, [(1500, 1250), (1610, 1240), (1612, 1278), (1502, 1288)]) * door, SIENNA, 0.10, wob=2, edge=0.8, ew=2, gran=0.5)
    # --- the gas man's magnet
    gx0, gy0, gx1, gy1 = 1870, 1236, 2150, 1384
    gm = Q(s, [(gx0, gy0 + 3), (gx1, gy0), (gx1 + 2, gy1), (gx0 + 2, gy1 + 2)])
    cast(s, gm, 10, 14, UMBER, 0.5, blur=8, clip=door)
    base(s, gm, CREAM, 0.06, wob=0.8, edge=0.3)
    inner = Q(s, [(gx0 + 10, gy0 + 12), (gx1 - 10, gy0 + 10), (gx1 - 8, gy1 - 10), (gx0 + 12, gy1 - 8)])
    s.wash(gm * (1 - inner), TERRA, 1.0, wob=1, soft=1, edge=0.4)
    s.wash(text_mask(s, "大順瓦斯行", (gx0 + gx1) / 2, gy0 + 50, 44, FONT_P), TERRA, 1.3, wob=0.3, soft=0.5, edge=0.3)
    s.wash(text_mask(s, "桶裝瓦斯 快速送達", (gx0 + gx1) / 2, gy0 + 92, 20, FONT_P), INK, 0.9, wob=0.2, soft=0.4)
    s.wash(text_mask(s, "2345-0678", (gx0 + gx1) / 2, gy0 + 122, 26, FONT_P), INK, 1.1, wob=0.2, soft=0.4)
    path(s, [(gx0, gy0 + 3), (gx1, gy0), (gx1 + 2, gy1), (gx0 + 2, gy1 + 2)], 2.0, INK, 1.3, closed=True, over=3, r=r)


# ------------------------------------------------------------------ the note

def note(name, col, k, shadow=False):
    """A blank square sticky note, 430 px. Anchor top centre."""
    W, H = NS + 140, NS + 160
    s = Sheet(W, H, 211)       # one seed: every colour has the same shape
    r = rs(211)
    ax, ay = W / 2, 60
    x0, x1, y0, y1 = ax - NS / 2, ax + NS / 2, ay, ay + NS
    # the bottom edge lifts a little: the corners curl up more than the middle
    bot = [(x1 + 1, y1 - 16), (x1 - 60, y1 - 4), (ax, y1 + 2), (x0 + 60, y1 - 2), (x0 - 1, y1 - 14)]
    outline = [(x0 + 2, y0), (ax, y0 - 1), (x1 - 1, y0 + 1), (x1 + 1, (y0 + y1) / 2)] + [tuple(p) for p in spline(bot, step=6)] + [(x0 - 1, (y0 + y1) / 2)]
    m = cv2.GaussianBlur(s.poly(outline, smooth=False), (0, 0), 1.2)
    if shadow:
        sh = cv2.warpAffine(m, np.float32([[1, 0, 9], [0, 1, 13]]), (W, H))
        s.wash(cv2.GaussianBlur(sh, (0, 0), 9), SEPIA, 0.42, wob=0, soft=0, edge=0, gran=0.3, flow=0.2)
        curl = s.ellipse(ax + 6, y1 + 6, NS * 0.48, 22) * s.vgrad(y1 - 30, y1 + 4)
        s.wash(cv2.GaussianBlur(curl, (0, 0), 10), SEPIA, 0.55, wob=2, soft=0, edge=0, gran=0.3)
        s.save(name, crop=True, anchor=(ax, ay))
        return
    base(s, m, col, k, wob=0.7, edge=0.6, ew=5, gran=0.4, flow=0.35, fs=170)
    s.wash(m, col, 0.16 * k / 0.8, wob=2, bleed=18, grad=s.rgrad(ax + 60, y0 + 240, 40, 300, 1, 0), edge=0.3, clip=m)      # a second, uneven layer
    s.wash(m * s.vgrad(y0 + 58, y0 + 76, 1, 0), col, 0.30 * k / 0.8, wob=1.5, soft=2, edge=0.5, gran=0.5)           # the glue band lies flat and reads deeper
    s.wash(m * s.vgrad(y0 + 58, y0 + 78, 1, 0), OCHRE, 0.10, wob=1.5, soft=2, edge=0.3)
    s.wash(m, SLATE, 0.10, wob=0, edge=0, grad=s.hgrad(ax - 40, x1 + 10, 0, 1))                                    # away from the window
    s.lift(m * s.vgrad(y1 - 70, y1 - 10, 0, 1), 0.30, soft=4, wob=2)                                             # the curl catches the light
    s.wash(m * s.vgrad(y1 - 92, y1 - 66, 0, 1) * s.vgrad(y1 - 66, y1 - 40, 1, 0), col, 0.18 * k / 0.8, wob=2, soft=3)  # the bend
    s.ink(spline(bot), 1.4, SEPIA, 0.55, smooth=False, skip=0.2)
    # pen: the top edge, both sides, the bottom; a glue line
    s.ink([(x0 - 5, y0 + 1), (ax, y0 - 1), (x1 + 6, y0 + 2)], 1.9, INK, 1.25, taper=(0.06, 0.1))
    s.ink([(x1 + 1, y0 - 3), (x1 + 3, (y0 + y1) / 2), (x1 + 1, y1 - 14)], 1.8, INK, 1.15, skip=0.12)
    s.ink([(x0, y0 - 2), (x0 - 2, (y0 + y1) / 2), (x0 - 1, y1 - 12)], 1.8, INK, 1.2, skip=0.08)
    s.ink(spline([(x1 + 4, y1 - 18)] + bot[1:-1] + [(x0 - 4, y1 - 12)]), 1.8, INK, 1.2, smooth=False, skip=0.1)
    s.save(name, crop=True, anchor=(ax, ay), meta={"write": [-NS / 2 + 44, 104, NS / 2 - 44, NS - 70], "size": NS})


def notes():
    note("fr_note", NOTE, 0.82)
    note("fr_note_pink", ROSE, 0.30)
    note("fr_note_mint", MINT, 0.42)
    note("fr_note_blue", RAIN, 0.42)
    note("fr_note_sh", None, 0, shadow=True)


# ------------------------------------------------------------------ hands (shared with s_table)

def skin(s, hm, lx=-1.0, ly=-0.6, tips=None):
    """Paint a hand mask: the first warm wash, shade on the side away from the light, a flush at the tips, the line."""
    roi = s._roi(hm, 0)
    y0, y1, x0, x1 = roi[0].start, roi[0].stop, roi[1].start, roi[1].stop
    cx, cy, R = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) / 2
    base(s, hm, SKIN, 0.60, wob=1.2, edge=0.4, flow=0.3, fs=90, gran=0.35)
    d = ((s.X - cx) * -lx + (s.Y - cy) * -ly) / (R * math.hypot(lx, ly))
    s.wash(hm, BURNT, 0.24, wob=1.5, bleed=10, grad=np.clip(0.45 + d * 0.9, 0, 1), edge=0.2, clip=hm)
    s.wash(hm, ROSE, 0.10, wob=1.5, bleed=12, grad=np.clip(0.5 + d * 0.6, 0, 1), clip=hm)
    if tips is not None:
        for (tx, ty) in tips:
            s.wash(s.rgrad(tx, ty, 10, R * 0.28, 1, 0) * hm, ROSE, 0.22, wob=1, soft=0, edge=0)


def nail(s, tip, back, w, k=0.5):
    """A fingernail seen from the back of the finger: tip = finger tip, back = a point down the finger."""
    tx, ty = tip
    ux, uy = back[0] - tx, back[1] - ty
    L = math.hypot(ux, uy)
    ux, uy = ux / L, uy / L
    c = (tx + ux * w * 0.62, ty + uy * w * 0.62)
    ang = math.degrees(math.atan2(uy, ux))
    m = s.ellipse(c[0], c[1], w * 0.40, w * 0.30, ang)
    s.wash(m, ROSE, 0.18 * k / 0.5, wob=0.6, soft=1, edge=0.5)
    s.lift(s.ellipse(c[0] - uy * w * 0.08 - ux * w * 0.1, c[1] + ux * w * 0.08 - uy * w * 0.1, w * 0.16, w * 0.07, ang), 0.5, soft=1.2)
    s.ink(ell(c[0], c[1], w * 0.40, w * 0.30, 0.6, 2 * math.pi - 0.6, 20, math.radians(ang)), 1.3, SEPIA, 0.7)


def crease(s, a, b, k=0.75, w=1.4):
    s.ink([a, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 2), b], w, BURNT, k, taper=(0.3, 0.3))


def navy(s, m, ax, ay, k=1.0):
    """Navy jacket cloth; (ax, ay) is the unit direction along the arm (toward the shoulder)."""
    base(s, m, NAVY, 0.95 * k, wob=2, edge=0.5, flow=0.5, fs=160, gran=0.5)
    roi = s._roi(m, 0)
    cx, cy = (roi[1].start + roi[1].stop) / 2, (roi[0].start + roi[0].stop) / 2
    side = (s.X - cx) * -ay + (s.Y - cy) * ax                     # across the arm
    span = max(roi[1].stop - roi[1].start, roi[0].stop - roi[0].start) * 0.35
    s.wash(m, INDIGO, 0.55, wob=2, bleed=14, grad=sstep(-span * 0.2, span, side), edge=0.2, clip=m)
    s.lift(m * sstep(span * 0.1, -span * 0.9, side), 0.20, soft=10, wob=2)


def knit(s, m, k=1.0):
    """The cream knit cardigan: soy-cream with ochre in the shade."""
    base(s, m, CREAM, 0.62 * k, wob=1.8, edge=0.45, flow=0.45, fs=90, gran=0.55)
    s.wash(m, SOY, 0.25, wob=1, edge=0)


def ribs(s, m, a, b, n, along, w=1.4, col=SIENNA, k=0.55):
    """Knit ribs: n lines from edge a to edge b (each a (p0, p1) pair), each line running `along` px."""
    for i in range(n):
        u = (i + 0.5) / n
        p = (a[0][0] + (a[1][0] - a[0][0]) * u, a[0][1] + (a[1][1] - a[0][1]) * u)
        q = (b[0][0] + (b[1][0] - b[0][0]) * u, b[0][1] + (b[1][1] - b[0][1]) * u)
        s.ink([p, ((p[0] + q[0]) / 2 + 1.5, (p[1] + q[1]) / 2), q], w, col, k, skip=0.25, clip=m, tremor=0.9)


def sleeve_cloth(s, slv, cuff, T, ang, kind, r, slen):
    """The sleeve and cuff of a dorsal hand in local coordinates (arm along +y from the wrist at y ~ 840).
    kind 'navy' = your jacket, 'knit' = my cream cardigan."""
    ax_, ay_ = -math.sin(ang), math.cos(ang)
    if kind == "navy":
        navy(s, slv, -ax_, -ay_)
        for (a, b, c_, w) in (((-150, 1000), (-120, 1250), (-160, slen - 100), 46), ((80, 980), (120, 1300), (150, slen - 100), 60), ((-20, 1060), (10, 1400), (-10, slen), 30)):
            s.stroke(T([a, b, c_]), w, INDIGO, 0.40, wob=3, bleed=8, clip=slv, taper=(0.3, 0.1), edge=0.2)
        for (a, b) in (((-180, 1040), (-60, 1080)), ((120, 1100), (220, 1060)), ((-230, 1300), (-120, 1330))):
            s.ink(T([a, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 14), b]), 1.6, INKBLUE, 0.7, skip=0.25)
        contour(s, slv, 2.6, INKBLUE, 1.4)
        navy(s, cuff, -ax_, -ay_, 1.1)
        s.ink(T([(-196, 850), (0, 852), (198, 856)]), 1.3, MIST, 0.5, skip=0.45, tremor=0.5)        # top-stitching, pale thread
        s.ink(T([(-200, 938), (0, 940), (202, 944)]), 1.3, MIST, 0.5, skip=0.45, tremor=0.5)
        bx, by = T([(150, 896)])[0]
        s.wash(s.ellipse(bx, by, 12, 12), PAYNE, 1.0, wob=0.5, edge=0.6)                              # a button
        oval(s, bx, by, 12, 12, 1.4, INKBLUE, 1.2)
        contour(s, cuff, 2.4, INKBLUE, 1.4)
    else:
        knit(s, slv, 0.85)
        x0, x1 = T([(-300, 1100)])[0], T([(300, 1100)])[0]
        side = s.hgrad(x0[0], x1[0], 0, 1) if abs(x1[0] - x0[0]) > abs(x1[1] - x0[1]) else s.vgrad(x0[1], x1[1], 0, 1)
        s.wash(slv, OCHRE, 0.30, wob=2, bleed=16, grad=side, edge=0.2, clip=slv)
        for i in range(17):                                    # stockinette: small v's in loose columns
            x = -250 + i * 30 + r.random() * 4
            for j in range(int((slen - 960) / 36)):
                y = 990 + j * 36 + (i % 2) * 18
                if r.random() < 0.7:
                    continue
                s.ink(T([(x - 5, y - 4), (x, y + 3), (x + 5, y - 4)]), 1.0, SIENNA, 0.3, smooth=False, clip=slv, taper=(0.2, 0.2), tremor=0.3, skip=0.2)
        for (a, b, c_, w) in (((-160, 1000), (-130, 1250), (-170, slen - 100), 46), ((90, 980), (130, 1300), (160, slen - 100), 56)):
            s.stroke(T([a, b, c_]), w, OCHRE, 0.30, wob=3, bleed=8, clip=slv, taper=(0.3, 0.1), edge=0.2)
        contour(s, slv, 2.3, UMBER, 1.25)
        knit(s, cuff)
        s.wash(cuff, OCHRE, 0.30, wob=1.5, bleed=8, grad=side, clip=cuff)
        ribs(s, cuff, (T([(-196, 836)])[0], T([(196, 840)])[0]), (T([(-204, 954)])[0], T([(204, 958)])[0]), 26, 120, 1.4, SIENNA, 0.6)
        contour(s, cuff, 2.2, UMBER, 1.3)


def dorsal_hand(s, T, S, ang, kind="navy", relaxed=False, sh=((12, 16, 8, 0.45), (60, 70, 36, 0.42)), slen=1800):
    """The back of a right hand, fingers up in local coordinates (the fingertips' contact at the origin), thumb on
    the left, the wrist at y ~ 840, the sleeve running down to `slen`. T maps local points to the sheet, S is its
    scale. Returns local reference points."""
    r = rs(int(abs(ang) * 1000) + (7 if relaxed else 0))
    W, H = s.w, s.h
    fingers = [([(-128, 372), (-130, 190), (-126, 0)], 88), ([(-42, 380), (-42, 180), (-38, -26)], 92),
               ([(44, 378), (46, 190), (50, 8)], 88), ([(126, 384), (132, 240), (138, 96)], 74)]
    thumb_p = [(-150, 700), (-228, 590), (-262, 470)]
    if relaxed:      # fingers a little apart and a little curled (shorter seen from above), the thumb in closer
        spread, out = (-0.10, -0.03, 0.05, 0.13), []
        for (p, w), a in zip(fingers, spread):
            rx, ry = p[0]
            q = []
            for i, (x, y) in enumerate(p):
                dx, dy = (x - rx) * 0.9, (y - ry) * 0.86
                q.append((rx + dx * math.cos(a) - dy * math.sin(a), ry + dx * math.sin(a) + dy * math.cos(a)))
            q[1] = (q[1][0] + 6, q[1][1])
            out.append((q, w))
        fingers = out
        thumb_p = [(-150, 700), (-212, 606), (-232, 506)]
    palm_p = [(-176, 330), (-60, 346), (60, 344), (168, 352), (188, 520), (172, 700), (150, 840), (-128, 840), (-168, 720), (-182, 520)]
    cuff_p = [(-196, 830), (196, 836), (204, 960), (-204, 954)]
    slv_p = [(-206, 930), (210, 936), (262, 1300), (290, slen), (-300, slen), (-262, 1300)]
    fms = [finger(s, T(p), w * S, tip=0.86) for p, w in fingers]
    thumb = finger(s, T(thumb_p), 100 * S, tip=0.78)
    palm = s.poly(T(palm_p))
    hm = np.maximum(palm, thumb)
    for f in fms:
        hm = np.maximum(hm, f)
    slv = s.poly(T(slv_p))
    cuff = s.poly(T(cuff_p))
    arm = np.maximum(slv, cuff)
    (dx0, dy0, b0, k0), (dx1, dy1, b1, k1) = sh
    s.wash(cv2.GaussianBlur(cv2.warpAffine(hm, np.float32([[1, 0, dx0], [0, 1, dy0]]), (W, H)), (0, 0), b0), SEPIA, k0, wob=0, soft=0, edge=0, gran=0.3)
    s.wash(cv2.GaussianBlur(cv2.warpAffine(arm, np.float32([[1, 0, dx1], [0, 1, dy1]]), (W, H)), (0, 0), b1), SEPIA, k1, wob=0, soft=0, edge=0, gran=0.3)
    sleeve_cloth(s, slv, cuff, T, ang, kind, r, slen)
    # the hand
    tips = T([(p[-1][0], p[-1][1] + 30) for p, w in fingers] + [(thumb_p[-1][0], thumb_p[-1][1] + 26)])
    skin(s, hm, -1.0, -0.4, tips)
    contour(s, hm, 2.3, SEPIA, 1.35)
    off = 1 - cv2.GaussianBlur(palm, (0, 0), 6)            # no line where a finger joins the back of the hand
    for f in fms[1:]:
        contour(s, f, 1.4, SEPIA, 0.75, skip=0.15, clip=off)
    contour(s, thumb, 1.4, SEPIA, 0.8, clip=off)
    for (p, w) in fingers:
        nail(s, T([p[-1]])[0], T([p[1]])[0], w * 0.86 * S, 0.3)
    nail(s, T([thumb_p[-1]])[0], T([thumb_p[1]])[0], 88 * S, 0.3)
    for (p, w) in fingers:
        x, pip = p[1][0], p[1][1] + 10
        crease(s, *T([(x - w * 0.28, pip), (x + w * 0.24, pip + 4)]), 0.65)
        crease(s, *T([(x - w * 0.22, pip + 18), (x + w * 0.18, pip + 20)]), 0.5, 1.1)
        dip = (p[1][1] + p[2][1]) / 2 + 6
        crease(s, *T([((p[1][0] + p[2][0]) / 2 - w * 0.22, dip), ((p[1][0] + p[2][0]) / 2 + w * 0.2, dip + 2)]), 0.55, 1.1)
        kx, ky = p[0][0], p[0][1] + 6                                                           # the knuckle
        s.ink(T([(kx - w * 0.3, ky + 8), (kx, ky - 6), (kx + w * 0.3, ky + 8)]), 1.4, BURNT, 0.6, taper=(0.3, 0.3))
        s.ink(T([(kx + 2, ky + 40), (kx * 0.6, ky + 200), (kx * 0.3, ky + 380)]), 1.2, BURNT, 0.18, skip=0.45)   # tendon
    s.ink(T([(-200, 610), (-170, 640), (-150, 700)]), 1.3, BURNT, 0.6)                           # the web of the thumb
    mid = fingers[1][0]
    return {"tip": ((fingers[1][0][-1][0] + fingers[2][0][-1][0]) / 2, (fingers[1][0][-1][1] + fingers[2][0][-1][1]) / 2), "cuff": (0, 895), "end": (0, slen)}


def hand_you():
    """Your right hand pressing the note flat, back of the hand to us, from the lower right. Drawn upright in local
    coordinates (fingers up, the contact at the origin) and turned so the fingers point up and to the left."""
    W, H = 1250, 1650
    AX, AY = 300, 200
    ang = -0.48
    S = 0.8                                   # local units to sheet px
    T = lambda pts: rot([(x * S, y * S) for x, y in pts], ang, AX, AY)
    s = Sheet(W, H, 221)
    ref = dorsal_hand(s, T, S, ang, "navy")
    ax_, ay_ = -math.sin(ang), math.cos(ang)
    s.save("fr_hand_you", crop=True, anchor=(AX, AY), meta={"cuff": [round(c_, 1) for c_ in np.subtract(T([ref["cuff"]])[0], (AX, AY))],
                                                                "arm": [round(ax_, 3), round(ay_, 3)]})


def pen_side(s, P0, P1, w=34, cap=True):
    """A black felt-tip pen from the side: P0 the tip, P1 the far end (as far as it shows)."""
    ux, uy = P1[0] - P0[0], P1[1] - P0[1]
    L = math.hypot(ux, uy)
    ux, uy = ux / L, uy / L
    nx, ny = -uy, ux
    at = lambda t, o=0.0: (P0[0] + ux * t + nx * o, P0[1] + uy * t + ny * o)
    nib = s.poly([at(0, -5), at(0, 5), at(30, w * 0.24), at(30, -w * 0.24)], smooth=False)
    cone = s.poly([at(28, -w * 0.24), at(28, w * 0.24), at(72, w * 0.46), at(72, -w * 0.46)], smooth=False)
    body = s.poly([at(70, -w / 2), at(70, w / 2), at(L, w / 2), at(L, -w / 2)], smooth=False)
    base(s, nib, INK, 1.3, wob=0.4, edge=0.4)
    base(s, cone, SLATE, 0.9, wob=0.5, edge=0.4)
    base(s, body, INK, 1.25, wob=0.8, edge=0.5, gran=0.3)
    s.wash(body, INKBLUE, 0.3, wob=0.5, soft=2)
    s.lift(s.poly([at(90, -w * 0.30), at(90, -w * 0.12), at(L - 10, -w * 0.12), at(L - 10, -w * 0.30)], smooth=False), 0.55, soft=2.5, wob=1)
    band = s.poly([at(160, -w / 2 - 1), at(160, w / 2 + 1), at(176, w / 2 + 1), at(176, -w / 2 - 1)], smooth=False)
    s.lift(band, 0.85, soft=0.8)
    s.wash(band, SLATE, 0.6, wob=0.4, soft=0.8)
    for o in (-w / 2, w / 2):
        s.ink([at(70, o), at(L, o)], 2.0, INK, 1.4, taper=(0.02, 0.02))
    s.ink([at(28, -w * 0.24), at(72, -w * 0.46)], 1.6, INK, 1.2)
    s.ink([at(28, w * 0.24), at(72, w * 0.46)], 1.6, INK, 1.2)
    return np.maximum(np.maximum(nib, cone), body)


def writing_hand(s, T, ang, sh=(56, 70, 34, 0.38), slen=1500):
    """My right hand holding the felt-tip to write, back of the hand to the eye, the pen between thumb and first
    finger, the other fingers curled under, the cream knit cuff. Local coordinates: the pen tip at the origin, the
    arm hanging down (+y) to `slen`. Returns local reference points."""
    r = rs(231)
    W, H = s.w, s.h
    idx_p = [(118, 262), (80, 196), (42, 130), (22, 76)]
    th_p = [(-6, 330), (-58, 252), (-50, 164), (-20, 106)]
    stubs = [([(244, 345), (240, 296), (218, 270)], 48), ([(212, 300), (206, 240), (180, 212)], 56), ([(168, 270), (150, 204), (118, 176)], 62)]
    palm_p = [(40, 250), (130, 240), (230, 296), (262, 420), (224, 522), (90, 544), (0, 476), (-40, 364)]
    cuff_p = [(36, 516), (252, 478), (290, 612), (66, 652)]
    slv_p = [(56, 630), (288, 590), (384, 1000), (430, slen), (-70, slen), (6, 1000)]
    pen = T([(0, 0), (56, 372)])
    idx = finger(s, T(idx_p), 62, tip=0.8)
    thumb = finger(s, T(th_p), 66, tip=0.76)
    fms = [finger(s, T(p), w, tip=0.85) for p, w in stubs]
    palm = s.poly(T(palm_p))
    back = palm.copy()
    for f in fms:
        back = np.maximum(back, f)
    cuff = s.poly(T(cuff_p))
    slv = s.poly(T(slv_p))
    allm = np.maximum(np.maximum(back, np.maximum(idx, thumb)), np.maximum(cuff, slv))
    s.wash(cv2.GaussianBlur(cv2.warpAffine(allm, np.float32([[1, 0, sh[0]], [0, 1, sh[1]]]), (W, H)), (0, 0), sh[2]), SEPIA, sh[3], wob=0, soft=0, edge=0, gran=0.3)
    # the cardigan sleeve: cream knit, ribbed cuff
    across = lambda m: s.hgrad(*[T([(-40, 900)])[0][0], T([(400, 900)])[0][0]], 0, 1)
    knit(s, slv, 0.85)
    s.wash(slv, OCHRE, 0.30, wob=2, bleed=16, grad=across(slv), edge=0.2, clip=slv)
    e0, e1 = T([(0, 700), (0, slen)])
    s.wash(slv, UMBER, 0.16, wob=2, bleed=16, grad=s.vgrad(e0[1], e1[1], 0, 1) if abs(e1[1] - e0[1]) > abs(e1[0] - e0[0]) else s.hgrad(e0[0], e1[0], 0, 1), clip=slv)
    for i in range(16):                                    # stockinette: little v's in loose columns
        x = 30 + i * 24 + r.random() * 4
        for j in range(int((slen - 690) / 30)):
            y = 690 + j * 30 + (i % 2) * 15
            if r.random() < 0.7:
                continue
            p = T([(x - 4, y - 3), (x, y + 2), (x + 4, y - 3)])
            s.ink(p, 1.0, SIENNA, 0.3, smooth=False, clip=slv, taper=(0.2, 0.2), tremor=0.3, skip=0.2)
    contour(s, slv, 2.3, UMBER, 1.25)
    knit(s, cuff)
    s.wash(cuff, OCHRE, 0.32, wob=1.5, bleed=8, grad=across(cuff), clip=cuff)
    ribs(s, cuff, (T([(44, 522)])[0], T([(254, 484)])[0]), (T([(70, 648)])[0], T([(288, 608)])[0]), 20, 130, 1.4, SIENNA, 0.6)
    contour(s, cuff, 2.2, UMBER, 1.3)
    # the pen, under the hand
    pen_side(s, pen[0], pen[1], 40)
    # the back of the hand and the curled fingers
    skin(s, back, -1.0, -0.6)
    s.wash(back, BURNT, 0.16, wob=2, bleed=8, grad=s.rgrad(*T([(214, 470)])[0], 40, 200, 1, 0), clip=back)
    contour(s, back, 2.2, SEPIA, 1.3)
    for (p, w) in stubs:
        s.ink(T([(p[1][0] - w * 0.45, p[1][1] - 6), (p[2][0], p[2][1] - w * 0.18), (p[1][0] + w * 0.42, p[1][1] - 18)]), 1.6, SEPIA, 0.9)
        kx, ky = p[0]
        s.ink(T([(kx - w * 0.3, ky + 8), (kx, ky - 4), (kx + w * 0.3, ky + 8)]), 1.3, BURNT, 0.55, taper=(0.3, 0.3))
    # the pen's tail, coming out over the web of the thumb toward me
    web = T([(38, 268)])[0]
    tail = pen[1]
    s.wash(cv2.GaussianBlur(cv2.warpAffine(s.ribbon([web, tail], 40), np.float32([[1, 0, 10], [0, 1, 14]]), (W, H)), (0, 0), 6), SEPIA, 0.4, wob=0, soft=0, edge=0)
    ux, uy = tail[0] - pen[0][0], tail[1] - pen[0][1]
    L = math.hypot(ux, uy)
    t0 = L * 0.56
    tm = s.ribbon([(pen[0][0] + ux * t0 / L, pen[0][1] + uy * t0 / L), tail], 40)
    base(s, tm, INK, 1.25, wob=0.6, edge=0.5, gran=0.3)
    s.wash(tm, INKBLUE, 0.3, wob=0.5, soft=2)
    nx, ny = -uy / L, ux / L
    s.lift(s.ribbon([(pen[0][0] + ux * (t0 + 10) / L - nx * 9, pen[0][1] + uy * (t0 + 10) / L - ny * 9), (tail[0] - ux * 14 / L - nx * 9, tail[1] - uy * 14 / L - ny * 9)], 8), 0.55, soft=2)
    s.wash(s.ellipse(tail[0], tail[1], 20, 20), INK, 0.9, wob=0.4, edge=0.5)
    contour(s, tm, 2.0, INK, 1.3)
    # thumb and first finger on top of the pen
    for m in (thumb, idx):
        skin(s, m, -1.0, -0.6)
    s.wash(idx, ROSE, 0.18, wob=1, soft=0, edge=0, grad=s.rgrad(*T([(24, 84)])[0], 6, 60, 1, 0))
    s.wash(thumb, ROSE, 0.16, wob=1, soft=0, edge=0, grad=s.rgrad(*T([(-22, 112)])[0], 6, 60, 1, 0))
    contour(s, idx, 2.1, SEPIA, 1.3)
    contour(s, thumb, 2.1, SEPIA, 1.3)
    nail(s, T([(24, 70)])[0], T([(44, 130)])[0], 50, 0.35)
    nail(s, T([(-18, 100)])[0], T([(-50, 164)])[0], 54, 0.35)
    crease(s, *T([(58, 196), (100, 188)]), 0.7, 1.3)
    crease(s, *T([(26, 134), (60, 124)]), 0.5, 1.1)
    crease(s, *T([(-74, 244), (-44, 252)]), 0.5, 1.1)
    return {"tip": (0, 0), "cuff": (165, 565), "end": (180, slen)}


def hand_me():
    """My right hand writing on the door with the felt-tip, from behind and to the right. Anchor = the pen tip."""
    W, H = 1400, 1600
    AX, AY = 240, 200
    ang = -0.62
    T = lambda pts: rot(pts, ang, AX, AY)
    s = Sheet(W, H, 231)
    ref = writing_hand(s, T, ang)
    ax_, ay_ = -math.sin(ang), math.cos(ang)
    s.save("fr_hand_me", crop=True, anchor=(AX, AY), meta={"cuff": [round(c_, 1) for c_ in np.subtract(T([ref["cuff"]])[0], (AX, AY))],
                                                               "arm": [round(ax_, 3), round(ay_, 3)]})

ALL = {"fr_bg": door_bg, "fr_note": notes, "fr_hand_you": hand_you, "fr_hand_me": hand_me}
