"""The small dining table from straight above: a checked cloth, the two mugs, the pad of notes, a pen, a phone, and the
hands that come in over the edges. Sheet 3072x1728 at about 40 px per cm on the table top. The morning light comes
from the window at the left; every sprite carries its own soft shadow toward the lower right.

Layers
  tb_bg          opaque 3072x1728: gingham cloth (pale celadon/tile on cream, ruled by hand, two fold creases) over
                 x < ~2590; the bare wooden table edge to x ~2800; the terracotta floor far below beyond it.
                 meta centre [x, y] the middle of the cloth area (between the two places)
                      cloth_edge  x of the cloth's hemmed edge (at the centre row);  table_edge  x of the table's edge
  tb_mug_you     your navy mug, empty, a dry coffee ring inside; handle to the right. Anchor = centre of the rim.
                 meta rim r of the rim (px), handle [dx, dy] middle of the handle
  tb_mug_me      my mint-glazed mug with tea; same geometry. meta rim, handle, tea r of the tea surface
  tb_pad         a square pad of yellow sticky notes (stack edges showing). Anchor = centre of the top sheet.
                 meta write [x0, y0, x1, y1] the area for handwriting, relative to the anchor;  size  side in px
  tb_pen         the black felt-tip, capped, lying along x. Anchor = centre. meta nib [dx, dy] the capped end, end [dx, dy]
  tb_phone       face up, dark screen. tb_phone_back  face down (navy case). Anchor = centre; long side along y.
  tb_hand_you    your right hand palm-down, relaxed, coming in from the TOP (navy cuff). Anchor = the end of the sleeve
                 (top edge of the sprite, centre): put it at the frame edge and slide. meta tip [dx, dy] the middle
                 fingertips, cuff [dx, dy], arm [dx, dy] unit vector from the hand back along the arm
  tb_hand_me     my right hand palm-down, relaxed, from the BOTTOM (cream knit cuff). Anchor = sleeve end (bottom
                 edge). meta tip, cuff, arm as above
  tb_hand_me_pen my right hand from the BOTTOM writing with the felt-tip. Anchor = sleeve end. meta tip = the pen
                 tip, cuff, arm
"""
import math

import cv2
import numpy as np

from kit import *
from s_fridge import dorsal_hand, writing_hand, rot, rr, dense

TW, TH = 3072, 1728
CEN = (1300, 864)
TH_ = 0.021                      # the cloth lies a little askew
CLOTH_U1 = 1290                  # its hemmed edge, in cloth coordinates (u to the right of the centre)
TABLE_X = 2800
PERIOD, BANDW = 84, 42           # gingham: 1 cm bands, 2 cm repeat
LIGHT = (16, 22)                 # shadow offset for things lying on the table


def cu(u, v):
    """Cloth coordinates (u right, v down, about the centre, turned by TH_) to sheet."""
    c, s_ = math.cos(TH_), math.sin(TH_)
    return (CEN[0] + u * c - v * s_, CEN[1] + u * s_ + v * c)


def table_bg():
    s = Sheet(TW, TH, 401)
    r = rs(401)
    # the floor far below, beyond the table: terracotta squares, soft (out of focus), in the table's shadow
    fl = s.rect(TABLE_X - 20, -10, TW + 10, TH + 10)
    s.wash(fl, TERRA, 0.75, wob=4, bleed=8, edge=0.2, flow=0.5, fs=200, gran=0.5)
    for y in (300, 880, 1460):
        s.lift(s.rect(TABLE_X, y - 6, TW + 10, y + 6), 0.45, soft=5, wob=4)
    s.lift(s.rect(2990, -10, 3002, TH + 10), 0.4, soft=5, wob=4)
    s.wash(fl, SEPIA, 0.85, wob=0, edge=0, grad=s.hgrad(TABLE_X, TABLE_X + 240, 1, 0.25), gran=0.4)
    s.wash(fl, PRUSSIAN, 0.18, wob=0, edge=0)
    # the table top: wood, grain running across
    top = s.rect(-10, -10, TABLE_X, TH + 10)
    wood(s, top, SIENNA, BURNT, 0.62, ang=math.pi / 2 + 0.01, n=70, seed=41, lw=1.4, streak=0.6)
    s.wash(s.rect(TABLE_X - 26, -10, TABLE_X, TH + 10), UMBER, 0.75, wob=1, soft=4, edge=0.3)        # the rounded edge
    s.lift(s.rect(TABLE_X - 60, -10, TABLE_X - 40, TH + 10), 0.25, soft=6, wob=2)
    seg(s, (TABLE_X, -10), (TABLE_X + 2, TH + 10), 2.6, INK, 1.4, r=r)
    seg(s, (TABLE_X - 26, -10), (TABLE_X - 25, TH + 10), 1.4, SEPIA, 0.8, skip=0.3, r=r)
    # the cloth
    U0, V0, V1 = -1700, -1100, 1100
    corners = [cu(U0, V0), cu(CLOTH_U1, V0), cu(CLOTH_U1, V1), cu(U0, V1)]
    cl = Q(s, corners)
    cast(s, cl, 6, 8, SEPIA, 0.5, blur=6)
    base(s, cl, CREAM, 0.30, wob=1.0, edge=0.3, flow=0.4, fs=400, gran=0.35)
    k0 = int(U0 / PERIOD) - 1
    for i in range(k0, int(CLOTH_U1 / PERIOD) + 1):          # bands running down
        u = i * PERIOD + 10
        q = [cu(u, V0), cu(u + BANDW, V0), cu(u + BANDW, V1), cu(u, V1)]
        s.wash(Q(s, q) * cl, CELADON, 0.52, wob=1.6, soft=1.0, edge=0.55, ew=3, gran=0.5, flow=0.4, fs=120)
    for j in range(int(V0 / PERIOD) - 1, int(V1 / PERIOD) + 1):   # bands running across
        v = j * PERIOD + 30
        q = [cu(U0, v), cu(CLOTH_U1, v), cu(CLOTH_U1, v + BANDW), cu(U0, v + BANDW)]
        s.wash(Q(s, q) * cl, TILE, 0.40, wob=1.6, soft=1.0, edge=0.55, ew=3, gran=0.5, flow=0.4, fs=120)
    mm = cv2.GaussianBlur(cl, (0, 0), 1.0)
    for i in range(k0, int(CLOTH_U1 / PERIOD) + 1):          # the ruling pen, now and then
        for e in (0, BANDW):
            if r.random() < 0.16:
                u = i * PERIOD + 10 + e
                a0 = V0 + r.random() * 600
                s.ink([cu(u, a0), cu(u + (r.random() - 0.5) * 4, a0 + 900 + r.random() * 900)], 1.2, TEAL, 0.4, skip=0.45, clip=mm, tremor=1.2)
    for j in range(int(V0 / PERIOD) - 1, int(V1 / PERIOD) + 1):
        for e in (0, BANDW):
            if r.random() < 0.16:
                v = j * PERIOD + 30 + e
                a0 = U0 + r.random() * 1200
                s.ink([cu(a0, v), cu(a0 + 1200 + r.random() * 1400, v + (r.random() - 0.5) * 4)], 1.2, SLATE, 0.4, skip=0.45, clip=mm, tremor=1.2)
    # fold creases: a ridge (light) with a soft valley beside it
    for (a, b, d) in ((cu(-120, V0), cu(-110, V1), (1, 0)), (cu(U0, 70), cu(CLOTH_U1, 60), (0, 1))):
        s.stroke([a, ((a[0] + b[0]) / 2 + d[1] * 8, (a[1] + b[1]) / 2 + d[0] * 6), b], 46, UMBER, 0.14, wob=3, bleed=10, clip=cl, taper=(0, 0), edge=0.1)
        s.lift(s.ribbon([(a[0] - d[0] * 18, a[1] - d[1] * 18), (b[0] - d[0] * 18, b[1] - d[1] * 18)], 8) * cl, 0.22, soft=4, wob=2)
    # a soft rumple or two, and a faint old tea ring
    for (x, y, rx_, ry_) in ((560, 300, 260, 90), (2100, 1350, 300, 110)):
        s.wash(s.blob(x, y, rx_, ry_, 0.3) * cl, UMBER, 0.10, wob=4, bleed=30, edge=0.2)
    ring = s.ellipse(760, 1240, 150, 146) * (1 - s.ellipse(762, 1242, 141, 137))
    s.wash(ring * cl, SIENNA, 0.32, wob=2.5, edge=0.8, ew=2, gran=0.6, flow=0.6, fs=40)
    # the hem along the cloth's edge, stitched
    hem = Q(s, [cu(CLOTH_U1 - 40, V0), cu(CLOTH_U1, V0), cu(CLOTH_U1, V1), cu(CLOTH_U1 - 40, V1)])
    s.wash(hem, CELADON, 0.30, wob=1, edge=0.4)
    s.wash(hem, UMBER, 0.12, wob=1, soft=2)
    s.ink([cu(CLOTH_U1 - 34, V0), cu(CLOTH_U1 - 36, 0), cu(CLOTH_U1 - 34, V1)], 1.4, SLATE, 0.8, skip=0.55, tremor=0.5)
    s.ink([cu(CLOTH_U1, V0), cu(CLOTH_U1 + 2, 0), cu(CLOTH_U1, V1)], 2.4, INK, 1.4)
    s.ink([cu(CLOTH_U1 - 40, V0 + 200), cu(CLOTH_U1 - 41, 0), cu(CLOTH_U1 - 40, V1 - 300)], 1.2, SEPIA, 0.6, skip=0.4)
    # the morning: warm from the left, cooler and a little darker toward the right
    s.wash(s.full(), OCHRE, 0.12, wob=0, soft=0, edge=0, gran=0.2, flow=0.3, grad=s.hgrad(0, 2400, 1, 0))
    s.wash(s.full(), UMBER, 0.12, wob=0, soft=0, edge=0, gran=0.3, flow=0.4, grad=s.rgrad(1200, 860, 900, 2000, 0, 1, sy=0.7))
    s.save("tb_bg", opaque=True, q=90, meta={"centre": list(CEN), "cloth_edge": round(cu(CLOTH_U1, 0)[0]), "table_edge": TABLE_X})


def shadow(s, m, k=0.5, blur=12, off=LIGHT):
    s.wash(cv2.GaussianBlur(cv2.warpAffine(m, np.float32([[1, 0, off[0]], [0, 1, off[1]]]), (s.w, s.h)), (0, 0), blur), SEPIA, k, wob=0, soft=0, edge=0, gran=0.3)


def mug(name, glaze, inner, k_out, tea=False):
    s = Sheet(700, 560, 411 if not tea else 412)
    c = (270, 270)
    R, RI, RB = 172, 156, 112                      # outer rim, inner rim, the bottom seen from above
    hp = [(c[0] + 160, c[1] - 40), (c[0] + 236, c[1] - 52), (c[0] + 290, c[1] - 24), (c[0] + 290, c[1] + 24), (c[0] + 236, c[1] + 52), (c[0] + 160, c[1] + 40)]
    hd = s.ribbon(hp, 30)
    body = s.ellipse(c[0], c[1], R, R)
    shadow(s, np.maximum(body, hd), 0.6, 14, (22, 30))
    base(s, hd, glaze, k_out, wob=0.8, edge=0.5, gran=0.4)
    s.wash(hd, SEPIA, 0.25, wob=0.5, soft=3, grad=s.vgrad(c[1] - 50, c[1] + 50))
    s.lift(s.ribbon(hp[:3], 6), 0.45, soft=2)
    contour(s, hd, 1.8, INK, 1.3)
    base(s, body, glaze, k_out, wob=0.8, edge=0.5, ew=4, gran=0.5, flow=0.3)
    s.wash(body, SEPIA, 0.2, wob=0.5, soft=4, grad=s.rgrad(c[0] + 60, c[1] + 70, 40, 230, 1, 0) * 0 + s.hgrad(c[0] - 120, c[0] + 180, 0, 1))
    s.lift(s.ellipse(c[0], c[1], R, R) * (1 - s.ellipse(c[0] + 6, c[1] + 8, R - 4, R - 4)), 0.5, soft=2)          # light on the upper-left lip
    ins = s.ellipse(c[0], c[1], RI, RI)
    base(s, ins, inner, 0.32 if not tea else 0.5, wob=0.6, edge=0.4)
    # the inner wall: in shade on the side toward the light, lit on the far side
    wall = ins * (1 - s.ellipse(c[0] + 12, c[1] + 16, RB, RB))
    s.wash(wall, SEPIA if not tea else LEAF, 0.30, wob=1, soft=6, grad=s.rgrad(c[0] - 110, c[1] - 120, 40, 260, 1, 0.1))
    s.wash(ins, SLATE, 0.10, wob=0, edge=0)
    if tea:
        tea_m = s.ellipse(c[0], c[1], RI - 10, RI - 10)
        base(s, tea_m, SIENNA, 0.46, wob=0.8, edge=0.7, ew=6, gran=0.4, flow=0.4, fs=60)
        s.wash(tea_m, OCHRE, 0.30, wob=1, soft=3, clip=tea_m)
        s.wash(tea_m, UMBER, 0.16, wob=1, soft=3, grad=s.rgrad(c[0] - 60, c[1] - 60, 30, 200, 1, 0), clip=tea_m)
        s.wash(tea_m, SIENNA, 0.45, wob=1, bleed=10, grad=s.rgrad(c[0] + 50, c[1] + 50, 20, 180, 1, 0.2), clip=tea_m)
        s.wash(tea_m * (1 - s.ellipse(c[0] - 6, c[1] - 6, RI - 18, RI - 18)), UMBER, 0.5, wob=0.5, soft=1.5)      # the meniscus
        refl = Q(s, [(c[0] - 86, c[1] - 92), (c[0] - 20, c[1] - 96), (c[0] - 16, c[1] - 36), (c[0] - 82, c[1] - 32)])
        s.lift(refl, 0.55, soft=5, wob=2)                                        # the window, a small square, in the tea
        s.lift(s.ellipse(c[0] + 70, c[1] + 80, 30, 8, -40), 0.3, soft=3)
        oval(s, c[0], c[1], RI - 10, RI - 10, 1.4, UMBER, 0.9, skip=0.2)
    else:
        bot = s.ellipse(c[0] + 12, c[1] + 16, RB, RB)
        s.lift(bot, 0.2, soft=8)
        ringm = s.ellipse(c[0] + 14, c[1] + 18, RB - 8, RB - 10) * (1 - s.ellipse(c[0] + 15, c[1] + 19, RB - 16, RB - 18))
        s.wash(ringm, UMBER, 0.75, wob=2, edge=0.9, ew=2, gran=0.6, flow=0.6, fs=30)         # the dry coffee ring
        s.wash(s.ellipse(c[0] + 14, c[1] + 18, RB - 10, RB - 12), SIENNA, 0.12, wob=2, bleed=6, edge=0.6)
        s.wash(s.blob(c[0] + 50, c[1] + 60, 30, 18, 0.4), UMBER, 0.4, wob=1.5, edge=0.9, ew=2)                  # the last drop, dried
        for a in (0.4, 1.3, 2.2):                                                    # dried trickles down the inner wall
            x0, y0 = c[0] + math.cos(a) * (RI - 4), c[1] + math.sin(a) * (RI - 4)
            x1, y1 = c[0] + 12 + math.cos(a) * (RB - 4), c[1] + 16 + math.sin(a) * (RB - 4)
            s.ink([(x0, y0), (x1, y1)], 3.0, UMBER, 0.45, taper=(0.4, 0.1), dry=0.4)
        oval(s, c[0] + 12, c[1] + 16, RB, RB, 1.2, SEPIA, 0.5, skip=0.45)
    oval(s, c[0], c[1], R, R, 2.6, INK, 1.5, twice=True)
    oval(s, c[0], c[1], RI, RI, 1.8, INK, 1.1, skip=0.1)
    meta = {"rim": R, "handle": [236, 0]}
    if tea:
        meta["tea"] = RI - 10
    s.save(name, crop=True, anchor=c, meta=meta)


def mugs():
    mug("tb_mug_you", NAVY, CREAM, 1.05)
    mug("tb_mug_me", MINT, MINT, 0.85, tea=True)


def pad():
    """A pad of yellow sticky notes from above: the top sheet, and the sheets under it showing along two edges."""
    s = Sheet(520, 520, 421)
    r = rs(421)
    c, N = (250, 250), 304
    x0, y0 = c[0] - N / 2, c[1] - N / 2
    full = s.rect(x0, y0, x0 + N + 10, y0 + N + 12)
    shadow(s, full, 0.55, 8, (12, 16))
    for i in range(6, 0, -1):                       # the stack: each sheet a hair out of line
        dx, dy = i * 1.6 + (r.random() - 0.5) * 1.5, i * 2.0 + (r.random() - 0.5) * 1.5
        m = s.rect(x0 + dx, y0 + dy, x0 + N + dx, y0 + N + dy)
        base(s, m, NOTE, 0.85, wob=0.3, edge=0.3, gran=0.3)
        s.wash(m, OCHRE, 0.15 + i * 0.03, wob=0.3, soft=1)
        s.ink([(x0 + N + dx, y0 + dy + 6), (x0 + N + dx, y0 + N + dy), (x0 + dx + 6, y0 + N + dy)], 1.0, SEPIA, 0.55, smooth=False, skip=0.3, tremor=0.3)
    top = s.rect(x0, y0, x0 + N, y0 + N)
    base(s, top, NOTE, 0.82, wob=0.6, edge=0.6, ew=5, gran=0.4, flow=0.35, fs=120)
    s.wash(top * s.vgrad(y0 + 40, y0 + 54, 1, 0), NOTE, 0.25, wob=1, soft=2, edge=0.5)          # the glue band
    s.wash(top, SLATE, 0.08, wob=0, edge=0, grad=s.hgrad(c[0], x0 + N, 0, 1))
    s.lift(top * s.rgrad(x0 + 60, y0 + 80, 10, 200, 1, 0), 0.15, soft=0, wob=0)
    path(s, [(x0, y0), (x0 + N, y0), (x0 + N, y0 + N), (x0, y0 + N)], 1.9, INK, 1.25, closed=True, over=4, r=r)
    s.save("tb_pad", crop=True, anchor=c, meta={"write": [-N / 2 + 30, -N / 2 + 74, N / 2 - 30, N / 2 - 30], "size": N})


def pen():
    """The felt-tip lying on the cloth, capped, the clip up."""
    s = Sheet(700, 180, 431)
    r = rs(431)
    cy = 86
    xa, xb = 80, 620                                # capped end (left), back end (right)
    body = s.poly(dense([(xa + 150, cy - 23), (xb - 16, cy - 22), (xb, cy - 14), (xb + 4, cy), (xb, cy + 14), (xb - 16, cy + 22), (xa + 150, cy + 23)], 8), smooth=False)
    cap = s.poly(dense([(xa + 10, cy - 20), (xa + 156, cy - 25), (xa + 156, cy + 25), (xa + 10, cy + 20), (xa, cy + 10), (xa - 2, cy - 10)], 8), smooth=False)
    shadow(s, np.maximum(body, cap), 0.55, 6, (10, 14))
    base(s, body, INK, 1.2, wob=0.6, edge=0.5, gran=0.3)
    s.wash(body, INKBLUE, 0.3, wob=0.4, soft=2)
    base(s, cap, INK, 1.25, wob=0.5, edge=0.5)
    s.lift(s.rect(xa + 170, cy - 15, xb - 30, cy - 9), 0.55, soft=2, wob=1)
    s.lift(s.rect(xa + 14, cy - 16, xa + 146, cy - 10), 0.5, soft=2, wob=1)
    band = s.rect(xa + 260, cy - 24, xa + 330, cy + 24)
    s.lift(band, 0.9, soft=0.8)
    s.wash(band, MIST, 0.6, wob=0.4, soft=0.6)
    for i in range(4):
        s.ink([(xa + 270 + i * 6, cy - 12), (xa + 300 + i * 6, cy - 12)], 1.0, SLATE, 0.8, tremor=0.2)
    clip = s.poly([(xa + 30, cy - 6), (xa + 150, cy - 6), (xa + 156, cy + 2), (xa + 30, cy + 2)], smooth=False)
    base(s, clip, MIST, 0.7, wob=0.3, edge=0.4)
    s.lift(s.rect(xa + 34, cy - 5, xa + 146, cy - 3), 0.6, soft=0.8)
    contour(s, clip, 1.2, INK, 1.0)
    contour(s, np.maximum(body, cap), 2.0, INK, 1.4)
    seg(s, (xa + 154, cy - 25), (xa + 156, cy + 25), 1.6, INK, 1.2, r=r)
    s.save("tb_pen", crop=True, anchor=((xa + xb) / 2, cy), meta={"nib": [xa - (xa + xb) / 2, 0], "end": [xb - (xa + xb) / 2, 0]})


def phone():
    """Your phone: face up (dark glass; the page can light it) and face down (a navy case)."""
    for name, back in (("tb_phone", False), ("tb_phone_back", True)):
        s = Sheet(460, 780, 441)
        r = rs(441)
        cx, cy, hw, hh = 220, 380, 144, 300
        P = rr(cx - hw, cy - hh, cx + hw, cy + hh, 40, n=9, step=12)
        body = s.poly(P, smooth=False)
        shadow(s, body, 0.6, 10, (16, 22))
        if back:
            base(s, body, NAVY, 0.85, wob=0.8, edge=0.5, flow=0.4, fs=120, gran=0.5)
            s.wash(body, INDIGO, 0.35, wob=0.5, soft=4, grad=s.hgrad(cx - hw, cx + hw, 0, 1))
            s.lift(body * s.hgrad(cx - hw, cx - hw + 50, 1, 0), 0.25, soft=2)
            cam = s.poly(rr(cx - hw + 26, cy - hh + 26, cx - hw + 136, cy - hh + 136, 26, n=7, step=8), smooth=False)   # the camera's little square
            base(s, cam, PAYNE, 1.0, wob=0.4, edge=0.5)
            for (lx, ly) in ((cx - hw + 56, cy - hh + 56), (cx - hw + 106, cy - hh + 56), (cx - hw + 56, cy - hh + 106)):
                s.wash(s.ellipse(lx, ly, 18, 18), INK, 1.2, wob=0.3, edge=0.5)
                s.lift(s.ellipse(lx - 5, ly - 6, 5, 4), 0.6, soft=1)
                oval(s, lx, ly, 18, 18, 1.3, INK, 1.1)
            s.wash(s.ellipse(cx - hw + 106, cy - hh + 106, 8, 8), CREAM, 0.6, wob=0.2)
            s.ink(dense(rr(cx - hw + 26, cy - hh + 26, cx - hw + 136, cy - hh + 136, 26, n=7, step=8), 6), 1.6, INK, 1.2, closed=True)
        else:
            base(s, body, PAYNE, 1.1, wob=0.8, edge=0.4)
            scr = s.poly(rr(cx - hw + 14, cy - hh + 14, cx + hw - 14, cy + hh - 14, 30, n=9, step=12), smooth=False)
            s.wash(scr, INKBLUE, 1.25, wob=0.6, edge=0.3, flow=0.3, fs=100)
            s.lift(Q(s, [(cx - hw + 30, cy - hh + 40), (cx - hw + 150, cy - hh + 40), (cx - hw + 70, cy - hh + 330), (cx - hw + 30, cy - hh + 330)]), 0.16, soft=10)
            s.lift(s.rect(cx + 20, cy - 120, cx + 110, cy - 30), 0.10, soft=6)
            s.wash(s.ellipse(cx, cy - hh + 34, 9, 9), INK, 1.0, wob=0.2)
        s.ink(dense(P, 6), 2.4, INK, 1.5, closed=True)
        s.save(name, crop=True, anchor=(cx, cy))


def hands():
    # yours, from the top: the dorsal hand turned upside down (fingers down, thumb at the right)
    S = 0.8
    ang = math.pi + 0.10
    W, H = 760, 1320
    AX, AY = 360, 30
    slen = 1500
    T0 = lambda pts: rot([(x * S, y * S) for x, y in pts], ang, 0, 0)
    ex, ey = T0([(0, slen)])[0]
    T = lambda pts: [(x - ex + AX, y - ey + AY) for x, y in T0(pts)]
    s = Sheet(W, H, 451)
    ref = dorsal_hand(s, T, S, ang, "navy", relaxed=True, sh=((10, 14, 8, 0.42), (16, 22, 16, 0.38)), slen=slen)
    meta_ = lambda ref, T, ang: {"tip": [round(v, 1) for v in np.subtract(T([ref["tip"]])[0], T([ref["end"]])[0])],
                                 "cuff": [round(v, 1) for v in np.subtract(T([ref["cuff"]])[0], T([ref["end"]])[0])],
                                 "arm": [round(-math.sin(ang), 3), round(math.cos(ang), 3)]}
    s.save("tb_hand_you", crop=True, anchor=T([ref["end"]])[0], meta=meta_(ref, T, ang))
    # mine, from the bottom, palm down
    ang = -0.08
    W, H = 760, 1320
    AX, AY = 380, 1290
    T0 = lambda pts: rot([(x * S, y * S) for x, y in pts], ang, 0, 0)
    ex, ey = T0([(0, slen)])[0]
    T = lambda pts: [(x - ex + AX, y - ey + AY) for x, y in T0(pts)]
    s = Sheet(W, H, 452)
    ref = dorsal_hand(s, T, S, ang, "knit", relaxed=True, sh=((10, 14, 8, 0.42), (16, 22, 16, 0.38)), slen=slen)
    s.save("tb_hand_me", crop=True, anchor=T([ref["end"]])[0], meta=meta_(ref, T, ang))
    # mine again, writing
    ang = -0.30
    W, H = 1000, 1300
    slen = 1080
    AX, AY = 660, 1270
    T0 = lambda pts: rot(pts, ang, 0, 0)
    ex, ey = T0([(180, slen)])[0]
    T = lambda pts: [(x - ex + AX, y - ey + AY) for x, y in T0(pts)]
    s = Sheet(W, H, 453)
    ref = writing_hand(s, T, ang, sh=(34, 46, 22, 0.4), slen=slen)
    s.save("tb_hand_me_pen", crop=True, anchor=T([ref["end"]])[0], meta=meta_(ref, T, ang))


ALL = {"tb_bg": table_bg, "tb_mugs": mugs, "tb_pad": pad, "tb_pen": pen, "tb_phone": phone, "tb_hands": hands}
