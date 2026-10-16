"""The old lane (巷子), seen from inside it: one-point perspective, running away from the camera and bending to the
right at the far end. That bend is the "halfway corner": a little red shrine sits on the wall the lane turns along,
a round traffic mirror on a pole watches the corner.
Painted through a real camera (CAM, eye 1.5 m, no pitch), in neutral overcast daylight values with no cast light:
the page darkens it for dusk and night and adds the lamp and the windows.

Sheet 3072x1728. Layers (back to front) and suggested parallax depth p:
  ln_sky_dusk  opaque  blue-hour sky                                           p 0.35
  ln_sky_day   opaque  bright washed sky after rain                            p 0.35
  ln_far       sprite  the bend: the wall the lane turns along (with the shrine), the turned lane's floor, roofs beyond  p 0.7
  ln_ground    sprite  the wet concrete lane up to the bend: puddles left pale (they mirror the sky), drain, 慢, edge lines  p 1.0
  ln_left      sprite  the left wall and everything on it                     p 1.0
  ln_right     sprite  the right wall, the wall lamp, the corner mirror, the wires across  p 1.0
  ln_fg        sprite  very near: a parked scooter at the left edge, a big pot and a bucket at the right edge  p 1.4
All sprites: crop with anchor (0, 0), so L() places them by their ox/oy.
Meta (on ln_ground, repeated on every layer): path (15 points along the ground from the camera round the bend; the last
two go behind the corner, so draw the line under ln_right), bend, vanish, vanish2 (the turned lane's), lamp (the wall
lamp's head), shrine (its little lamp), windows (rects of the glass where warm light can be shown: on ln_left, ln_right and
ln_far each lists its own; empty on ln_ground), stand [x, y, h] (a figure's feet
at mid-distance and its height in px there), stand2 (a little farther), p (parallax per layer).
"""
import math

import cv2
import numpy as np

from kit import *
from s_block import (Sub, sub_of, J, U, box, quad, line, put, streaks, mottle, hl, cable, iron, bump, Cam, Fac, ctr, tilework, vline,
                     hline, plant_pot, acunit, shutter, canopy, downpipe, laundry, cage, PINK, GREY, GREY2, RUST, DOOR_RED, DOOR_GREEN, FONT,
                     HAND)

W, H = 3072, 1728
CAM = Cam(1450.0, 1420.0, 900.0, (-0.5, 1.5, 0.0), 0.0)
TH = math.radians(40)
TURN = 22.6
L1 = Fac(CAM, (-2.1, 0, 0.3), (-2.1, 0, TURN), 1)
L2 = Fac(CAM, (-2.1, 0, TURN), (-2.1 + 30 * math.sin(TH), 0, TURN + 30 * math.cos(TH)), 1)
R1 = Fac(CAM, (2.1, 0, 0.3), (2.1, 0, 19.6), -1)
PLASTER = (0.92, 0.86, 0.76)
BLUEGREY = (0.76, 0.80, 0.82)
PEACH = (0.94, 0.80, 0.70)
LIGHTS = {}          # filled while painting: windows, lamp, shrine


def P3(x, y, z):
    return tuple(float(v) for v in CAM((x, y, z)))


def meta_all():
    path = []
    for z in (2.7, 3.4, 4.3, 5.4, 6.8, 8.4, 10.2, 12.2, 14.3, 16.4, 18.4):
        path.append(P3(-0.1 + 0.12 * math.sin(z * 0.5), 0, z))
    for x, z in ((0.05, 20.0), (0.45, 21.4), (1.12, 23.2), (2.08, 24.3), (3.05, 25.5)):      # round the bend, along the turned lane's middle
        path.append(P3(x, 0, z))
    m = {"path": [[round(a), round(b)] for a, b in path], "bend": [round(v) for v in P3(0.3, 0, 21.2)], "vanish": [round(v) for v in P3(-0.5, 1.5, 1e6)],
         "vanish2": [round(v) for v in P3(-0.5 + 1e6 * math.sin(TH), 1.5, 1e6 * math.cos(TH))],
         "stand": [round(v) for v in P3(-0.1, 0, 9.0)] + [round(1.65 * CAM.f / 9.0)],
         "stand2": [round(v) for v in P3(-0.1, 0, 13.0)] + [round(1.65 * CAM.f / 13.0)],
         "p": {"ln_sky_dusk": 0.35, "ln_sky_day": 0.35, "ln_far": 0.7, "ln_ground": 1.0, "ln_left": 1.0, "ln_right": 1.0, "ln_fg": 1.4}}
    hx, hy = R1.p(9.2, 3.28, 0.95)
    m["lamp"] = [round(hx), round(hy + 0.06 * R1.ppm(9.2, 3.2, 0.95))]
    m["shrine"] = [round(v) for v in L2.p(2.0, 1.18, 0.4)]
    m["windows"] = LIGHTS.get("windows", [])
    return m


# ---------------------------------------------------------------------------------------------- skies
def sky_dusk():
    s = Sheet(W, H, 201)
    r = rs(201)
    F = s.full()
    s.wash(F, INDIGO, 0.55, wob=0, soft=0, edge=0, grad=s.vgrad(-200, 1100, 1, 0.25), flow=0.4, fs=500, gran=0.06)
    s.wash(F, VIOLET, 0.32, wob=0, soft=0, edge=0, grad=bump(s, 200, 1300, 0.55), flow=0.6, fs=400, gran=0.06)
    s.wash(F, DUSK, 0.45, wob=0, soft=0, edge=0, flow=0.6, fs=380, gran=0.06)
    s.wash(F, ROSE, 0.2, wob=0, soft=0, edge=0, grad=s.vgrad(600, 1200, 0, 1), flow=0.7, fs=300, gran=0.05)
    s.wash(F, OCHRE, 0.16, wob=0, soft=0, edge=0, grad=s.vgrad(850, 1300, 0, 1), flow=0.7, fs=260, gran=0.05)
    for cx, cy, rx, ry, k in ((700, 160, 700, 70, 0.3), (2100, 110, 800, 70, 0.28), (1400, 380, 600, 50, 0.22), (2600, 470, 500, 40, 0.18),
                              (400, 560, 500, 36, 0.16), (1700, 690, 700, 34, 0.14)):
        s.wash(s.blob(cx, cy, rx, ry, irr=0.4, n=15), INDIGO, k, wob=9, bleed=30, edge=0.25, ew=10, flow=0.5, fs=200, gran=0.1)
        s.wash(s.blob(cx + J(r, 60), cy + ry * 0.6, rx * 0.7, ry * 0.4, irr=0.4, n=13), VIOLET, k * 0.6, wob=6, bleed=16, edge=0.4, ew=8, gran=0.1)
        s.wash(s.blob(cx + J(r, 80), cy + ry * 0.9, rx * 0.6, ry * 0.25, irr=0.4, n=11), ROSE, k * 0.35, wob=6, bleed=12, edge=0.3, gran=0.1)
    s.wash(F, None, 0.22, wob=0, soft=0, edge=0, gran=0, flow=0.6, fs=300, grad=s.vgrad(900, 1500, 0, 1), lift=True)
    s.save("ln_sky_dusk", opaque=True, q=88, meta={"p": 0.35})


def sky_day():
    s = Sheet(W, H, 202)
    r = rs(202)
    F = s.full()
    s.wash(F, RAIN, 0.42, wob=0, soft=0, edge=0, grad=s.vgrad(-200, 1100, 1, 0.1), flow=0.4, fs=500, gran=0.06)
    s.wash(F, PRUSSIAN, 0.16, wob=0, soft=0, edge=0, grad=s.vgrad(-100, 700, 1, 0), flow=0.5, fs=400, gran=0.06)
    s.wash(F, CELADON, 0.12, wob=0, soft=0, edge=0, grad=s.vgrad(500, 1300, 0, 1), flow=0.6, fs=300, gran=0.05)
    for cx, cy, rx, ry in ((600, 240, 520, 110), (1500, 140, 420, 90), (2400, 300, 600, 120), (1200, 560, 520, 70), (2800, 640, 360, 60),
                           (300, 720, 420, 60), (1900, 760, 500, 50)):
        bm = s.blob(cx, cy, rx, ry, irr=0.35, n=15)
        s.lift(bm, 0.85, bleed=22, wob=8)
        s.wash(s.blob(cx + J(r, 40), cy + ry * 0.55, rx * 0.8, ry * 0.4, irr=0.35, n=13), SLATE, 0.14, wob=6, bleed=18, edge=0.35, ew=8, gran=0.15)
        s.wash(s.blob(cx + J(r, 40), cy + ry * 0.7, rx * 0.6, ry * 0.25, irr=0.35, n=11), DUSK, 0.08, wob=6, bleed=14, edge=0.3, gran=0.1)
    s.wash(F, None, 0.35, wob=0, soft=0, edge=0, gran=0, flow=0.6, fs=300, grad=s.vgrad(850, 1500, 0, 1), lift=True)
    s.save("ln_sky_day", opaque=True, q=88, meta={"p": 0.35})


# ---------------------------------------------------------------------------------------------- the wall's vocabulary
def house(S, F, r, s0, s1, top, col, tiled=True, storeys=(3.3, 6.3, 9.3, 12.3), last=False, ts=0.1):
    """A house front on a wall plane: its first wash, wet-in-wet, tiles or old plaster, slab bands, the party-wall line, coping."""
    ext = F.q(s0 - 0.2, -0.8, s1 + 0.2, top + 1.2) + F.q(s0 - 0.2, -0.8, s1 + 0.2, top + 1.2, 1.4)
    s = sub_of(S, ext, 50)
    m = Q(s, F.q(s0, -0.8, s1, top))
    s.lift(m, 1.0, soft=0.5, wob=0)
    m = s.wash(m, col, 0.62, wob=0.8, soft=0.7, edge=0.35, ew=6, flow=0.5, fs=200, gran=0.35, body=1)
    sv, yv = F.sy(s)
    s.wash(m, DUSK, 0.18, wob=0, soft=0, edge=0, grad=sstep(top, 0, yv).astype(np.float32), flow=0.6, fs=150, gran=0.3)
    s.wash(m, col, 0.18, wob=3, bleed=10, edge=0.15, flow=0.7, fs=120, gran=0.4, clip=m)
    bb = np.asarray(F.q(s0, 0, s1, top), float)
    mottle(s, r, bb[:, 0].min(), bb[:, 1].min(), bb[:, 0].max(), bb[:, 1].max(), [DUSK, SLATE, col, ROSE, MOSS], 7, 0.1, size=(30, 110), clip=m)
    if tiled:
        tilework(s, r, F, m * sstep(storeys[0] + 0.1, storeys[0] + 0.3, yv), sv, yv, ts, k=0.3, seed=int(s0 * 37) + 3, oddk=0.1, lift=0.12)
        s.wash(Q(s, F.q(s0, -0.8, s1, storeys[0])), GREY, 0.3, wob=1, soft=1, edge=0.2, clip=m)        # the ground storey: stone-ish, greyer
    else:                        # old plaster: patches fallen off, cracks
        for _ in range(int(r.integers(2, 5))):
            cs, cy = U(r, s0 + 0.3, s1 - 0.3), U(r, 0.3, top - 0.5)
            pts = [F.p(cs + 0.5 * math.cos(t) * U(r, 0.5, 1.3), cy + 0.35 * math.sin(t) * U(r, 0.5, 1.3)) for t in np.linspace(0, 2 * math.pi, 10)[:-1]]
            pm = Q(s, pts)
            s.wash(pm, GREY, 0.4, wob=1.5, soft=0.8, edge=0.6, ew=3, gran=0.5)
            s.wash(pm, SEPIA, 0.1, wob=2, soft=1, edge=0.3, flow=0.8, fs=20)
            quad(s, r, pts, 1.1, SEPIA, 0.8, over=1, dbl=0, j=0.5, skip=0.3)
        for _ in range(3):
            a = (U(r, s0, s1), U(r, 0.5, top))
            pts = [F.p(a[0], a[1])]
            for k in range(5):
                a = (a[0] + J(r, 0.3), a[1] - U(r, 0.1, 0.4))
                pts.append(F.p(*a))
            s.ink(pts, 1.0, SEPIA, 0.7, smooth=False, skip=0.3, tremor=0.8)
    for y in storeys[1:]:
        if y < top - 0.3:
            s.wash(Q(s, F.q(s0, y - 0.1, s1, y + 0.15)), GREY, 0.25, wob=0.8, soft=0.8, edge=0.4, ew=3)
            hline(s, r, F, s0, s1, y - 0.1, 0, pw(F, s0, y), SEPIA, 1.2, over=5, dbl=0.4)
            hline(s, r, F, s0, s1, y + 0.15, 0, pw(F, s0, y, 0, 0.012), SEPIA, 0.9, over=2, skip=0.3)
    # the coping, the line of the party wall, the foot
    put(s, Q(s, F.q(s0, top - 0.15, s1, top, 0.05)), GREY, 0.35, lift=0.8, wob=0.5, edge=0.4, ew=2)
    hline(s, r, F, s0, s1, top, 0.05, pw(F, s0, top), SEPIA, 1.4, over=6, dbl=0.4)
    hline(s, r, F, s0, s1, top - 0.15, 0.05, pw(F, s0, top, 0, 0.012), SEPIA, 1.0, over=2, skip=0.2)
    vline(s, r, F, s0, -0.6, top, 0, pw(F, s0, 2), SEPIA, 1.4, over=8, dbl=0.4)
    if last:
        vline(s, r, F, s1, -0.6, top, 0, pw(F, s1, 2), SEPIA, 1.5, over=8, dbl=0.5)
    s.wash(Q(s, F.q(s0, -0.8, s1, 0.45)), SEPIA, 0.25, wob=3, bleed=10, edge=0.1, clip=m)
    streaks(s, r, [F.p(U(r, s0, s1), top - 0.1) for _ in range(8)], F.ppm(s0, top) * 1.2, 6, PAYNE, 0.14, (3, 9), clip=m)
    return s


def pw(F, sv, y, d=0.0, base=0.02, lo=1.0, hi=3.0):
    return float(np.clip(F.ppm(sv, y, d) * base, lo, hi))


def door(s, r, F, a, b, h, col, grille=True, steps=0):
    pen = pw(F, a, 1.0)
    dm = Q(s, F.q(a, 0, b, h, -0.1))
    put(s, dm, col, 0.8, lift=0.95, wob=0.8, edge=0.5, ew=4, flow=0.35, fs=max(50, F.ppm(a, 1.0) * 0.6), gran=0.45)
    s.wash(dm, PAYNE, 0.15, wob=1, soft=2, edge=0, grad=s.vgrad(F.p(a, h)[1], F.p(a, 0)[1], 0.2, 1))
    jam = Q(s, F.qs(b, 0, h, 0, -0.1))
    head = Q(s, F.qh(a, b, h, 0, -0.1))
    put(s, np.maximum(jam, head), GREY, 0.45, lift=0.8, wob=0.4)
    s.wash(jam, PAYNE, 0.3, wob=0.4, soft=0.8)
    quad(s, r, F.q(a, 0, b, h, -0.1), pen, SEPIA, 1.5, over=4, dbl=0.4)
    if grille:
        gq = F.q(a + 0.18, h * 0.55, b - 0.18, h * 0.88, -0.1)
        s.wash(Q(s, gq), INKBLUE, 0.5, wob=0.5, soft=0.8, edge=0.4, ew=2)
        quad(s, r, gq, pen * 0.8, SEPIA, 1.3, over=2)
        for q in np.linspace(0.15, 0.85, 5):
            vline(s, r, F, a + 0.18 + q * (b - a - 0.36), h * 0.55, h * 0.88, -0.1, pen * 0.7, SEPIA, 1.2, over=0)
    quad(s, r, F.q(a + 0.14, 0.18, b - 0.14, h * 0.48, -0.1), pen * 0.7, SEPIA, 1.0, over=2, dbl=0.2)
    hl(s, [F.p(a + 0.05, h - 0.05, -0.1), F.p(a + 0.05, 0.1, -0.1)], max(2, pen * 1.4), 0.3, dry=0.3)
    put(s, Q(s, F.q(b - 0.16, 1.0, b - 0.08, 1.12, -0.08)), OCHRE, 0.5, lift=0.6, wob=0.2)
    for i in range(steps):
        sq = F.q(a - 0.15, i * 0.16, b + 0.15, (i + 1) * 0.16, 0.25 * (steps - i))
        put(s, Q(s, sq), GREY, 0.45, lift=0.9, wob=0.5, edge=0.4, ew=2)
        quad(s, r, sq, pen, SEPIA, 1.3, over=2)
        top_ = F.qh(a - 0.15, b + 0.15, (i + 1) * 0.16, 0, 0.25 * (steps - i))
        put(s, Q(s, top_), GREY, 0.3, lift=0.9, wob=0.4)
        line(s, r, top_[3], top_[2], pen * 0.8, SEPIA, 1.2)


def window(s, r, F, a, b, y0, y1, glass=True, frame=MIST, col=GREY):
    """A lane window, painted glass (the page lights it): reveal, aluminium frame, two panes, sill. Returns its rect."""
    pen = pw(F, a, y0)
    jam = Q(s, F.qs(b, y0, y1, 0, -0.12))
    head = Q(s, F.qh(a, b, y1, 0, -0.12))
    put(s, np.maximum(jam, head), col, 0.5, lift=0.9, wob=0.4, edge=0.4, ew=2)
    s.wash(np.maximum(jam, head), PAYNE, 0.3, wob=0.5, soft=0.8)
    fm = Q(s, F.q(a, y0, b, y1, -0.12))
    put(s, fm, frame, 0.5, lift=0.9, wob=0.4, edge=0.45, ew=2)
    gq = F.q(a + 0.06, y0 + 0.06, b - 0.06, y1 - 0.06, -0.12)
    gm = Q(s, gq)
    s.wash(gm, SLATE, 0.75, wob=0.5, soft=0.8, edge=0.5, ew=3, flow=0.5, fs=20)
    s.wash(gm, INKBLUE, 0.35, wob=1, soft=2, grad=s.vgrad(gq[3][1], gq[0][1], 0.2, 1), clip=gm)
    s.lift(Q(s, [gq[3], F.p(a + (b - a) * 0.45, y1 - 0.06, -0.12), F.p(a + 0.06, y0 + (y1 - y0) * 0.45, -0.12)]), 0.35, soft=2, wob=1)     # the sky in it
    if r.random() < 0.5:        # a net curtain
        s.wash(Q(s, F.q(a + 0.06, y0 + 0.06, a + (b - a) * U(r, 0.3, 0.5), y1 - 0.06, -0.12)), CREAM, 0.25, wob=1, soft=1.5, edge=0.4)
    m = (a + b) / 2
    vline(s, r, F, m, y0 + 0.06, y1 - 0.06, -0.12, pen * 0.8, SEPIA, 1.2, over=1)
    quad(s, r, F.q(a, y0, b, y1, 0), pen, SEPIA, 1.4, over=4, dbl=0.3, j=0.6)
    quad(s, r, gq, pen * 0.6, SEPIA, 0.9, over=1, dbl=0, j=0.5, skip=0.2)
    sm = Q(s, F.q(a - 0.06, y0 - 0.07, b + 0.06, y0, 0.05))
    put(s, sm, GREY, 0.35, lift=0.8, wob=0.4, edge=0.4, ew=2)
    hline(s, r, F, a - 0.07, b + 0.07, y0 - 0.07, 0.05, pen, SEPIA, 1.3, over=2)
    streaks(s, r, [F.p(U(r, a, b), y0 - 0.07) for _ in range(4)], F.ppm(a, y0) * U(r, 0.5, 1.2), 3, PAYNE, 0.18, (2, 6))
    q = np.asarray(gq)
    LIGHTS.setdefault("windows", []).append([int(q[:, 0].min()), int(q[:, 1].min()), int(math.ceil(q[:, 0].max())), int(math.ceil(q[:, 1].max()))])


def wcage(s, r, F, a, b, y0, y1, plants=1, deco=0, roof=True, rust=False):
    fl = dict(a=a, b=b, wins=[dict(y0=y0, y1=y1)], dd=U(r, 0.32, 0.42), plants=plants, vine=r.random() < 0.3, roof=roof, deco=deco, rust=rust)
    cage(s, r, F, fl)


def boxes(s, r, F, a, y, n=3, m_=2, w=0.24, h=0.28, d=0.14, cols=(GREY, RUST, SLATE)):
    """Mailboxes or electricity meters: a little grid of tin boxes on the wall."""
    pen = pw(F, a, y, 0, 0.014)
    for i in range(n):
        for j in range(m_):
            sa, ya = a + i * (w + 0.02), y + j * (h + 0.02)
            fq = F.q(sa, ya, sa + w, ya + h, d)
            sq = F.qs(sa, ya, ya + h, 0, d)
            put(s, Q(s, sq), SLATE, 0.5, lift=0.8, wob=0.3)
            s.wash(Q(s, sq), PAYNE, 0.25, wob=0.3, soft=0.8)
            put(s, Q(s, fq), cols[(i + j) % len(cols)], 0.5, lift=0.85, wob=0.3, edge=0.5, ew=1)
            quad(s, r, fq, pen, SEPIA, 1.2, over=1, dbl=0, j=0.3)
            hline(s, r, F, sa + 0.04, sa + w - 0.04, ya + h * 0.7, d, pen * 0.7, SEPIA, 1.0, over=0)


def vsign(s, r, F, sv, y0, y1, d0, d1, text, col, tcol=(0.98, 0.96, 0.9), font=FONT):
    """A vertical sign board standing out from the wall into the lane, its face turned to us; the characters set on it."""
    q = F.qs(sv, y0, y1, d0, d1)                 # wall-bottom, out-bottom, out-top, wall-top
    pen = pw(F, sv, y0, d1)
    m = put(s, Q(s, q), col, 0.75, lift=1.0, wob=0.5, edge=0.5, ew=3, flow=0.5, fs=40)
    s.wash(m, PAYNE, 0.18, wob=1, soft=2, edge=0, grad=s.vgrad(q[2][1], q[1][1], 0, 1))
    # set the text on a flat card and lay it onto the board's quad
    from PIL import Image, ImageDraw, ImageFont
    n = len(text)
    cw, ch = 200, 200 * n + 80
    im = Image.new("L", (cw, ch), 0)
    d = ImageDraw.Draw(im)
    fnt = ImageFont.truetype(font, 150)
    for i, c in enumerate(text):
        d.text((cw / 2, 40 + 100 + i * 200), c, 255, fnt, anchor="mm")
    card = np.asarray(im, np.float32) / 255
    if q[1][0] < q[0][0]:
        card = card[:, ::-1].copy()
    src = np.float32([[0, ch], [cw, ch], [cw, 0], [0, 0]])
    dst = np.float32([(x - s.ox, y - s.oy) for x, y in (q[0], q[1], q[2], q[3])])
    Hm = cv2.getPerspectiveTransform(src, dst)
    scale = max(1e-3, float(np.hypot(*(dst[1] - dst[0]))) / cw)
    if scale < 0.6:
        card = cv2.GaussianBlur(card, (0, 0), 0.45 / scale)
    tm = cv2.warpPerspective(card, Hm, (s.w, s.h), flags=cv2.INTER_LINEAR)
    s.lift(tm * m, 0.9, soft=0.4, wob=0.3)
    s.wash(tm * m, tcol, 0.4, wob=0.3, soft=0.5, edge=0.3)
    quad(s, r, q, pen, SEPIA, 1.5, over=3, dbl=0.4, j=0.4)
    for y in (y0 + 0.15, y1 - 0.15):
        line(s, r, F.p(sv, y, 0), F.p(sv, y, d0), pen * 0.9, INKBLUE, 1.3, over=1)
    return q


def scooter(s, r, F, sv, d, col, facing=1, kick=0.0):
    """A scooter parked along the wall (its length along s), seen from behind and the lane side; sv = its rear."""
    L = 1.75 * facing
    so = d + 0.62                   # its lane-side face
    dm = d + 0.31                   # its middle
    pen = pw(F, sv, 0.6, so, 0.016)
    P = lambda u, y, dd=so: F.p(sv + u * L, y, dd)
    # wheels (in the plane of the scooter), dark
    for u0 in (0.15, 0.82):
        cxy = np.array(P(u0, 0.24, dm))
        rx = abs(np.array(P(u0 + 0.13, 0.24, dm))[0] - cxy[0]) + 0.5
        ry = abs(np.array(P(u0, 0.48, dm))[1] - cxy[1])
        wm = s.ellipse(cxy[0], cxy[1], max(rx, ry * 0.18), ry)
        iron(s, wm, 0.8, SEPIA, INKBLUE)
    # body side panel and the rear
    body = [P(0.0, 0.5), P(0.02, 0.78), P(0.12, 0.86), P(0.55, 0.86), P(0.6, 0.45), P(0.72, 0.42), P(0.78, 1.05), P(0.86, 1.1), P(0.9, 0.55), P(0.98, 0.42), P(0.95, 0.3), P(0.05, 0.3)]
    bm = put(s, Q(s, body), col, 0.75, lift=1.0, wob=0.5, edge=0.5, ew=3, flow=0.5, fs=30)
    s.wash(bm, PAYNE, 0.25, wob=1, soft=2, edge=0, grad=s.vgrad(P(0, 0.86)[1], P(0, 0.3)[1], 0, 1))
    rear = [P(0.0, 0.32, d + 0.08), P(0.0, 0.32, so), P(0.0, 0.82, so), P(0.0, 0.82, d + 0.08)]
    rm = put(s, Q(s, rear), col, 0.6, lift=1.0, wob=0.4, edge=0.5, ew=2)
    s.wash(rm, PAYNE, 0.45, wob=0.5, soft=1)
    tl = [P(0.0, 0.62, d + 0.2), P(0.0, 0.62, d + 0.44), P(0.0, 0.72, d + 0.44), P(0.0, 0.72, d + 0.2)]
    put(s, Q(s, tl), (0.82, 0.30, 0.26), 0.8, lift=1.0, wob=0.3)
    seat = [P(0.04, 0.86, d + 0.1), P(0.04, 0.86, so - 0.05), P(0.56, 0.88, so - 0.05), P(0.56, 0.88, d + 0.1)]
    seat2 = [P(0.04, 0.86, so - 0.05), P(0.56, 0.88, so - 0.05), P(0.54, 0.95, so - 0.08), P(0.06, 0.95, so - 0.08)]
    sm = put(s, np.maximum(Q(s, seat), Q(s, seat2)), SEPIA, 1.0, lift=1.0, wob=0.4, edge=0.4, ew=2)
    s.wash(sm, INKBLUE, 0.4, wob=0.5, soft=1)
    hl(s, [P(0.08, 0.93, so - 0.07), P(0.5, 0.94, so - 0.07)], max(1.5, pen * 1.2), 0.3)
    # handlebar, mirrors
    hb = [P(0.86, 1.1, d + 0.0), P(0.86, 1.12, so + 0.05)]
    s.ink(hb, pen * 2.2, INK, 1.4)
    for dd in (d - 0.02, so + 0.07):
        a, b = P(0.86, 1.12, dd), P(0.84, 1.42, dd)
        s.ink([a, b], pen * 0.9, INK, 1.3)
        s.wash(s.ellipse(b[0], b[1], max(2, pen * 2.2), max(2, pen * 3)), SEPIA, 1.0, wob=0.2, soft=0.5, body=1)
    ctr(s, bm, pen, INK, 1.3, step=3)
    quad(s, r, rear, pen, INK, 1.2, over=1, dbl=0, j=0.3)
    hl(s, [P(0.2, 0.84), P(0.5, 0.84)], max(1.5, pen), 0.35)
    # its shadow on the wet ground
    g = [F.p(sv - 0.05 * L, 0, d - 0.05), F.p(sv + 1.05 * L, 0, d - 0.05), F.p(sv + 1.05 * L, 0, so + 0.15), F.p(sv - 0.05 * L, 0, so + 0.15)]
    s.wash(Q(s, g), PAYNE, 0.35, wob=2, soft=4, edge=0)


def bucket(s, r, F, sv, d, rad=0.18, h=0.36, col=(0.40, 0.58, 0.78)):
    c0 = np.array(F.p(sv, 0, d))
    c1 = np.array(F.p(sv, h, d))
    sc = F.ppm(sv, h / 2, d)
    rx = rad * sc
    ry = rx * min(0.7, max(0.08, (1.5 - h) / float(CAM.depth(F.w3(sv, h, d)))))
    bm = np.maximum(s.poly([(c1[0] - rx, c1[1]), (c1[0] + rx, c1[1]), (c0[0] + rx * 0.85, c0[1]), (c0[0] - rx * 0.85, c0[1])], smooth=False), s.ellipse(c0[0], c0[1], rx * 0.85, ry))
    bm = put(s, bm, col, 0.8, lift=1.0, wob=0.4, edge=0.5, ew=2)
    s.wash(bm, PAYNE, 0.35, wob=1, soft=2, grad=s.hgrad(c1[0] - rx, c1[0] + rx, 0.1, 1))
    top = s.ellipse(c1[0], c1[1], rx, ry)
    put(s, top, PAYNE, 0.6, lift=1.0, wob=0.3)
    s.lift(s.ellipse(c1[0], c1[1] + ry * 0.2, rx * 0.8, ry * 0.6), 0.3, soft=1)
    pen = max(1.0, sc * 0.012)
    s.ink(ell(c1[0], c1[1], rx, ry, 0, 2 * math.pi, 30), pen, SEPIA, 1.3)
    s.ink([(c1[0] - rx, c1[1]), (c0[0] - rx * 0.85, c0[1])], pen, SEPIA, 1.3)
    s.ink([(c1[0] + rx, c1[1]), (c0[0] + rx * 0.85, c0[1])], pen, SEPIA, 1.3)
    s.ink(ell(c0[0], c0[1], rx * 0.85, ry, 0, math.pi, 20), pen, SEPIA, 1.3)


def bigpot(s, r, F, sv, d, size=1.0, leaf=MOSS, pot=TERRA):
    """A big pot of something on the ground by a door: a few stems and broad leaves."""
    plant_pot(s, r, F, sv, 0, d, 2.4 * size)


def clutter(s, r, F, s0, s1, y_top=4.5):
    """What grows on a near wall: a grey conduit with its saddles, black wires dropping from the bundle to a box, stains."""
    sc = F.ppm((s0 + s1) / 2, 1.5)
    cv_ = s0 + (s1 - s0) * U(r, 0.2, 0.4)
    a, b = F.p(cv_, 0.3, 0.05), F.p(cv_, y_top, 0.05)
    cm = s.ribbon([a, b], max(3, 0.05 * sc), smooth=False)
    put(s, cm, GREY, 0.45, lift=0.8, wob=0.4, edge=0.4, ew=2)
    s.wash(cm, PAYNE, 0.25, wob=0.5, soft=1, grad=s.hgrad(a[0] - 0.03 * sc, a[0] + 0.03 * sc, 0, 1))
    s.ink([a, b], max(1.0, sc * 0.006), SEPIA, 1.1)
    y = 0.6
    while y < y_top:
        p_ = F.p(cv_, y, 0.05)
        line(s, r, (p_[0] - 0.05 * sc, p_[1]), (p_[0] + 0.05 * sc, p_[1]), max(1.2, sc * 0.008), SEPIA, 1.2, over=0)
        y += U(r, 0.7, 1.1)
    bx = s0 + (s1 - s0) * U(r, 0.55, 0.75)
    q = F.q(bx, 2.0, bx + 0.32, 2.4, 0.08)
    put(s, Q(s, q), SLATE, 0.5, lift=0.9, wob=0.4, edge=0.45, ew=2)
    quad(s, r, q, max(1.0, sc * 0.008), SEPIA, 1.2, over=1, dbl=0)
    for k in range(3):
        a = F.p(bx + 0.06 + k * 0.1, 2.4, 0.1)
        b = F.p(bx + 0.06 + k * 0.1 + J(r, 0.15), y_top - 1.2, 0.12)
        s.ink([a, ((a[0] + b[0]) / 2 + J(r, 8), (a[1] + b[1]) / 2), b], max(1.0, sc * 0.006), INKBLUE, 1.2, tremor=1.0)
    for _ in range(4):
        x = U(r, s0, s1)
        streaks(s, r, [F.p(x, U(r, 2.5, y_top))], sc * U(r, 0.8, 2.0), 1, SEPIA, 0.12, (sc * 0.04, sc * 0.12), dry=0.5)


def notice(s, r, F, sv, y, w, h, text, paper_, ink_):
    """A paper notice pasted on the wall (a flat to let, a plumber's sticker): curling corners, the characters on it."""
    q = F.q(sv, y, sv + w, y + h, 0.01)
    m = put(s, Q(s, q), paper_, 0.25, lift=1.0, wob=0.8, edge=0.5, ew=3)
    s.wash(m, SEPIA, 0.08, wob=2, bleed=4, edge=0.3, flow=0.9, fs=12)
    from PIL import Image, ImageDraw, ImageFont
    n = len(text)
    vert = h > w
    cw, ch = (240, 240 * n + 60) if vert else (240 * n + 60, 240)
    im = Image.new("L", (cw, ch), 0)
    d = ImageDraw.Draw(im)
    fnt = ImageFont.truetype(FONT, 180)
    for i, c in enumerate(text):
        d.text((cw / 2, 30 + 120 + i * 240) if vert else (30 + 120 + i * 240, ch / 2), c, 255, fnt, anchor="mm")
    card = np.asarray(im, np.float32) / 255
    dst = np.float32([(x - s.ox, y_ - s.oy) for x, y_ in (q[3], q[2], q[1], q[0])])
    pad = 0.12
    cx_, cy_ = dst.mean(0)
    dst = (dst - (cx_, cy_)) * (1 - pad) + (cx_, cy_)
    scale = max(1e-3, float(np.hypot(*(dst[1] - dst[0]))) / cw)
    if scale < 0.6:
        card = cv2.GaussianBlur(card, (0, 0), 0.45 / scale)
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]]), dst)
    tm = cv2.warpPerspective(card, Hm, (s.w, s.h), flags=cv2.INTER_LINEAR)
    s.wash(tm * m, ink_, 0.9, wob=0.5, soft=0.5, edge=0.2)
    quad(s, r, q, pw(F, sv, y, 0, 0.01), SEPIA, 0.9, over=1, dbl=0, j=0.4, skip=0.2)
    streaks(s, r, [F.p(sv + w * U(r, 0.2, 0.8), y) for _ in range(2)], F.ppm(sv, y) * 0.3, 2, SEPIA, 0.12, (2, 6))


def wall_lamp(s, r, F, sv, y):
    """A street lamp on an arm off the wall, the lane's own: bracket, curved arm, a hooded head."""
    pen = pw(F, sv, y, 0.5)
    arm = [F.p(sv, y - 0.35, 0.0), F.p(sv, y + 0.05, 0.3), F.p(sv, y + 0.2, 0.7), F.p(sv, y + 0.12, 0.95)]
    iron(s, s.ribbon(arm, max(3, pen * 2.4)), 0.9)
    s.ink(arm, pen, INK, 1.3)
    s.ink([F.p(sv, y - 0.35, 0.0), F.p(sv, y + 0.1, 0.0)], pen * 1.6, INK, 1.4)
    hx, hy = F.p(sv, y + 0.08, 0.95)
    sc = F.ppm(sv, y, 0.95)
    hood = [(hx - 0.22 * sc, hy + 0.02 * sc), (hx - 0.12 * sc, hy - 0.1 * sc), (hx + 0.12 * sc, hy - 0.1 * sc), (hx + 0.22 * sc, hy + 0.02 * sc)]
    iron(s, s.poly(hood), 1.0)
    bulb = s.ellipse(hx, hy + 0.06 * sc, 0.1 * sc, 0.05 * sc)
    s.lift(bulb, 1.0, soft=0.5)
    s.wash(bulb, CREAM, 0.3, wob=0.2, soft=0.5, body=1)
    s.ink(hood + [hood[0]], pen, INK, 1.4, smooth=False)
    s.ink(ell(hx, hy + 0.06 * sc, 0.1 * sc, 0.05 * sc, 0, math.pi, 16), pen * 0.8, INK, 1.2)
    return (hx, hy + 0.06 * sc)


def gcables(s, r, F, s0, s1, y, n=3, step=4.0):
    hooks = np.arange(s0, s1 + 0.1, step)
    for j in range(n):
        for h0, h1 in zip(hooks, hooks[1:]):
            a, b = F.p(h0, y - j * 0.07, 0.12), F.p(h1, y - j * 0.07, 0.12)
            sag = F.ppm((h0 + h1) / 2, y) * U(r, 0.08, 0.2) * (1 + j * 0.3)
            cable(s, r, a, b, sag, pw(F, h0, y, 0, 0.012, 0.9, 2.2), INKBLUE, U(r, 1.0, 1.3))




def balc(s, r, F, a, b, yf, yt, col, laundry_=True, plants=1):
    from s_block import balcony
    balcony(s, r, F, dict(a=a, b=b, yf=yf, yt=yt, dd=U(r, 0.75, 0.95), laundry=laundry_, plants=plants), col)


def tin_top(S, r, F, s0, s1, y, h=2.4, setback=0.6):
    """A rusty corrugated room on a low roof, its front set back from the wall."""
    Fo = F.off(-setback)
    s = sub_of(S, Fo.q(s0, y, s1, y + h + 0.4) + Fo.q(s0, y, s1, y + h + 0.4, 0.8), 40)
    pen = pw(Fo, s0, y + h)
    fm = put(s, Q(s, Fo.q(s0, y, s1, y + h)), SIENNA, 0.55, lift=1.0, wob=0.8, edge=0.4, ew=4, flow=0.6, fs=50, gran=0.5)
    s.wash(fm, RUST, 0.45, wob=3, bleed=10, edge=0.2, flow=0.8, fs=40, gran=0.5, clip=fm)
    sv = s0 + 0.05
    while sv < s1:
        line(s, r, Fo.p(sv, y, 0), Fo.p(sv, y + h, 0), pen * 0.5, SEPIA, U(r, 0.4, 0.8), over=0, skip=0.25)
        sv += 0.08
    streaks(s, r, [Fo.p(U(r, s0, s1), y + h - 0.1) for _ in range(8)], Fo.ppm(s0, y + h) * 1.4, 6, BURNT, 0.3, (2, 7), clip=fm)
    rq = [Fo.p(s0 - 0.2, y + h + 0.1, 0), Fo.p(s1 + 0.2, y + h + 0.1, 0), Fo.p(s1 + 0.2, y + h - 0.1, 0.5), Fo.p(s0 - 0.2, y + h - 0.1, 0.5)]
    rm = put(s, Q(s, rq), RUST, 0.6, lift=1.0, wob=0.5, edge=0.45, ew=3)
    s.wash(rm, PAYNE, 0.35, wob=0.8, soft=1.5)
    quad(s, r, rq, pen, SEPIA, 1.4, over=4, dbl=0.4)
    quad(s, r, Fo.q(s0, y, s1, y + h), pen, SEPIA, 1.3, over=4, dbl=0.3)
    return s


def left():
    S = Sheet(W, H, 210)
    LIGHTS["windows"] = []
    r = rs(210)
    F = L1
    # b3: two storeys of blue-grey plaster at the bend, a tin room on top, a noodle shop below
    s = house(S, F, r, 16.2, TURN, 6.6, BLUEGREY, tiled=False, storeys=(3.3, 6.6), last=True)
    tin_top(S, r, F, 17.0, 21.6, 6.6, 2.3)
    s = Sub(S, *s_box(F, 16.0, TURN + 0.2, 0, 9.5))
    shutter(s, r, F, 17.6, 21.4, 2.5, GREY, 0.5)
    window(s, r, F, 16.7, 17.3, 1.0, 2.1)
    window(s, r, F, 17.8, 19.8, 4.2, 5.5)
    wcage(s, r, F, 17.8, 19.8, 4.2, 5.5, plants=2, deco=1)
    acunit(s, r, F, "box", 20.4, 4.5)
    canopy(s, r, F, 16.6, 21.8, 2.85, 0.9)
    vsign(s, r, F, 16.4, 2.0, 3.6, 0.15, 0.85, "麵", (0.94, 0.88, 0.70), tcol=(0.80, 0.26, 0.22))
    bucket(s, r, F, 16.6, 0.4)
    # b2: four storeys of pink tile, a shutter, two scooters, a salon's sign
    s = house(S, F, r, 10.6, 16.2, 12.6, PINK, tiled=True)
    s = Sub(S, *s_box(F, 10.4, 16.4, 0, 13))
    shutter(s, r, F, 11.0, 15.8, 2.8, SLATE, 0.55)
    for y0 in (4.3, 7.3, 10.3):
        window(s, r, F, 12.6, 14.8, y0, y0 + 1.4)
    balc(s, r, F, 11.0, 15.8, 9.3, 12.3, PINK, True, 1)
    wcage(s, r, F, 12.6, 14.8, 4.3, 5.7, plants=1, deco=0)
    acunit(s, r, F, "split", 11.2, 6.6)
    wcage(s, r, F, 12.6, 14.8, 7.3, 8.7, plants=2, deco=2, rust=True)
    acunit(s, r, F, "box", 15.2, 4.6)
    vsign(s, r, F, 11.4, 3.5, 6.2, 0.15, 0.85, "美髮", (0.84, 0.32, 0.30))
    downpipe(s, r, F, 10.75, 0.1, 12.5)
    gcables(s, r, F, 10.6, 16.2, 3.25, 3, 2.8)
    scooter(s, r, F, 13.6, 0.12, (0.86, 0.85, 0.80), 1)
    scooter(s, r, F, 11.5, 0.12, (0.52, 0.66, 0.74), 1)
    # b1: old peach plaster, three storeys, a red door up two steps, a balcony with washing
    s = house(S, F, r, 5.2, 10.6, 9.6, PEACH, tiled=False, storeys=(3.3, 6.3, 9.6))
    s = Sub(S, *s_box(F, 5.0, 10.8, 0, 10.5))
    door(s, r, F, 5.9, 6.9, 2.2, DOOR_RED, steps=2)
    window(s, r, F, 7.6, 9.4, 1.0, 2.2)
    wcage(s, r, F, 7.6, 9.4, 1.0, 2.2, plants=3, deco=0)
    balc(s, r, F, 5.6, 10.2, 3.3, 6.3, PEACH, True, 2)
    window(s, r, F, 7.0, 9.0, 7.3, 8.6)
    wcage(s, r, F, 7.0, 9.0, 7.3, 8.6, plants=1, deco=1)
    boxes(s, r, F, 7.0, 1.3, 1, 2, 0.26, 0.3)
    bigpot(s, r, F, 9.8, 0.35, 1.0)
    bucket(s, r, F, 10.25, 0.3, col=(0.84, 0.36, 0.30))
    plant_pot(s, r, F, 7.3, 0, 0.3, 1.6)
    gcables(s, r, F, 5.2, 10.6, 3.15, 3, 2.7)
    # b0: celadon tile, four storeys, a green door and its mailboxes, meters, a downpipe
    s = house(S, F, r, 0.0, 5.2, 12.6, CELADON, tiled=True)
    s = Sub(S, *s_box(F, -0.2, 5.4, 0, 6))
    door(s, r, F, 3.2, 4.2, 2.2, DOOR_GREEN, steps=1)
    boxes(s, r, F, 4.45, 1.2, 2, 2, 0.22, 0.26)
    downpipe(s, r, F, 0.15, 0.1, 6.0)
    acunit(s, r, F, "split", 4.0, 3.5)
    clutter(s, r, F, 0.4, 3.0)
    notice(s, r, F, 1.7, 1.45, 0.5, 0.62, "吉屋出租", (0.97, 0.95, 0.9), (0.80, 0.24, 0.22))
    notice(s, r, F, 2.45, 1.7, 0.28, 0.2, "通水管", (0.96, 0.9, 0.6), (0.15, 0.15, 0.2))
    notice(s, r, F, 2.5, 1.2, 0.22, 0.3, "收", (0.95, 0.94, 0.9), (0.3, 0.3, 0.4))
    gcables(s, r, F, 2.0, 5.2, 3.1, 4, 2.5)
    S.save("ln_left", crop=True, anchor=(0, 0), q=90, meta=meta_all())


def s_box(F, s0, s1, y0, y1, dmax=1.4):
    P = np.asarray(F.q(s0, y0, s1, y1) + F.q(s0, y0, s1, y1, dmax), float)
    return (P[:, 0].min() - 40, P[:, 1].min() - 40, P[:, 0].max() + 40, P[:, 1].max() + 40)


def mirror(s, r, F, sv, d, h=2.9):
    """The round convex traffic mirror on its orange pole at the corner, looking up the lane."""
    a, b = F.p(sv, 0, d), F.p(sv, h, d)
    sc = F.ppm(sv, h, d)
    pen = max(1.2, sc * 0.014)
    pm = s.ribbon([a, b], max(3, 0.07 * sc), smooth=False)
    put(s, pm, (0.92, 0.56, 0.28), 0.75, lift=1.0, wob=0.3, edge=0.4, ew=2)
    s.wash(pm, PAYNE, 0.3, wob=0.5, soft=1, grad=s.hgrad(a[0] - 0.04 * sc, a[0] + 0.04 * sc, 0, 1))
    s.ink([a, b], pen, INK, 1.3)
    cx, cy = b[0] - 0.1 * sc, b[1] - 0.25 * sc
    rx, ry = 0.36 * sc, 0.4 * sc
    rim = s.ellipse(cx, cy, rx, ry)
    put(s, rim, (0.92, 0.50, 0.24), 0.85, lift=1.0, wob=0.3, edge=0.5, ew=2)
    face = s.ellipse(cx, cy, rx * 0.84, ry * 0.86)
    put(s, face, RAIN, 0.4, lift=1.0, wob=0.3)
    s.wash(face, SLATE, 0.35, wob=1, soft=2, grad=s.vgrad(cy - ry, cy + ry, 0, 1))
    s.wash(s.ellipse(cx - rx * 0.2, cy + ry * 0.3, rx * 0.5, ry * 0.25), SEPIA, 0.3, wob=1, soft=2)       # the lane, bent round in it
    hl(s, [(cx - rx * 0.5, cy - ry * 0.5), (cx - rx * 0.2, cy - ry * 0.7)], max(2, sc * 0.03), 0.6)
    s.ink(ell(cx, cy, rx, ry, 0, 2 * math.pi, 40), pen * 1.2, INK, 1.4)
    s.ink(ell(cx, cy, rx * 0.84, ry * 0.86, 0.4, 0.4 + 2 * math.pi, 40), pen * 0.7, INK, 1.0, skip=0.2)
    s.ink([(cx + rx * 0.6, cy + ry * 0.4), b], pen, INK, 1.3)


def right():
    S = Sheet(W, H, 211)
    LIGHTS["windows"] = []
    r = rs(211)
    F = R1
    # c2: the corner building, celadon tile, three storeys; the mirror and the lane's name plate at its corner
    s = house(S, F, r, 11.8, 19.3, 9.8, TILE, tiled=True, storeys=(3.3, 6.5, 9.8), last=True)
    s = Sub(S, *s_box(F, 11.6, 19.6, 0, 11))
    shutter(s, r, F, 12.4, 15.6, 2.7, (0.76, 0.66, 0.56), 0.5)
    door(s, r, F, 16.3, 17.3, 2.2, DOOR_RED, grille=True)
    canopy(s, r, F, 16.0, 17.7, 2.7, 0.7)
    for y0 in (4.3, 7.5):
        window(s, r, F, 12.6, 14.6, y0, y0 + 1.35)
        wcage(s, r, F, 12.6, 14.6, y0, y0 + 1.35, plants=int(r.integers(0, 3)), deco=int(r.integers(0, 3)), rust=r.random() < 0.5)
        window(s, r, F, 16.2, 18.2, y0, y0 + 1.35)
        wcage(s, r, F, 16.2, 18.2, y0, y0 + 1.35, plants=int(r.integers(0, 3)), deco=int(r.integers(0, 3)))
    acunit(s, r, F, "split", 15.0, 4.0)
    acunit(s, r, F, "box", 18.5, 7.8)
    vsign(s, r, F, 15.6, 3.4, 5.9, 0.15, 0.8, "水電行", (0.44, 0.62, 0.66))
    plate = F.q(18.6, 2.3, 19.15, 2.6, 0.02)
    put(s, Q(s, plate), PRUSSIAN, 0.85, lift=1.0, wob=0.2, edge=0.4, ew=1)
    hl(s, [F.p(18.65, 2.45, 0.02), F.p(19.1, 2.45, 0.02)], 1.6, 0.6)
    quad(s, r, plate, 1.2, SEPIA, 1.2, over=1, dbl=0)
    mirror(s, r, F, 19.1, 0.45)
    gcables(s, r, F, 11.8, 19.3, 3.1, 3, 2.6)
    # c1: two storeys of blue-grey plaster; the lane's lamp on its arm; a green door, pots on the step
    s = house(S, F, r, 5.7, 11.8, 6.6, BLUEGREY, tiled=False, storeys=(3.3, 6.6))
    s = Sub(S, *s_box(F, 5.5, 12.0, 0, 8))
    door(s, r, F, 6.4, 7.4, 2.2, DOOR_GREEN, steps=1)
    window(s, r, F, 8.4, 10.4, 1.0, 2.3)
    wcage(s, r, F, 8.4, 10.4, 1.0, 2.3, plants=2, deco=1)
    window(s, r, F, 6.6, 8.6, 4.2, 5.5)
    wcage(s, r, F, 6.6, 8.6, 4.2, 5.5, plants=3, deco=0)
    acunit(s, r, F, "box", 9.6, 4.5)
    for sv, d, sz in ((7.75, 0.35, 1.5), (8.05, 0.75, 1.0), (11.2, 0.3, 2.0)):
        plant_pot(s, r, F, sv, 0, d, sz, trail=False)
    bucket(s, r, F, 10.9, 0.55)
    LIGHTS["lamp"] = wall_lamp(s, r, F, 9.2, 3.2)
    # pots along the parapet
    for sv in (6.3, 7.4, 10.8):
        plant_pot(s, r, F, sv, 6.6, 0.05, 1.3)
    gcables(s, r, F, 5.7, 11.8, 3.0, 3, 3.0)
    # c0: tile, three storeys: a red door, the meters, a caged window
    s = house(S, F, r, 0.0, 5.7, 9.6, TILE, tiled=True)
    s = Sub(S, *s_box(F, -0.2, 5.9, 0, 6))
    door(s, r, F, 2.0, 3.0, 2.2, DOOR_RED, steps=1)
    boxes(s, r, F, 3.25, 1.45, 1, 2, 0.28, 0.3, 0.18, (GREY, SLATE))
    window(s, r, F, 3.9, 5.4, 1.0, 2.2)
    wcage(s, r, F, 3.9, 5.4, 1.0, 2.2, plants=2, deco=2)
    downpipe(s, r, F, 0.2, 0.1, 6.0)
    clutter(s, r, F, 0.2, 1.9, 4.0)
    notice(s, r, F, 1.0, 1.5, 0.42, 0.55, "停車", (0.97, 0.96, 0.92), (0.80, 0.24, 0.22))
    notice(s, r, F, 3.25, 2.2, 0.26, 0.18, "抓漏", (0.95, 0.9, 0.62), (0.15, 0.15, 0.2))
    gcables(s, r, F, 1.5, 5.7, 3.0, 4, 2.4)
    # wires across the lane, slung from wall to wall
    s = Sub(S, 0, 0, W, H)
    for zl, zr, yl, yr, sag in ((6.0, 7.5, 6.2, 6.0, 40), (8.5, 9.5, 6.8, 6.3, 50), (12.0, 11.0, 7.2, 6.4, 30), (15.0, 16.5, 7.0, 6.6, 26),
                                (19.0, 18.5, 6.4, 6.0, 18), (10.0, 13.0, 8.4, 6.5, 40), (5.0, 9.0, 9.0, 6.4, 60)):
        a = P3(-2.0, yl, zl)
        b = P3(2.0, yr, zr)
        cable(s, r, a, b, sag, max(1.0, 1450 * 0.012 / ((zl + zr) / 2)) * 1.4, INKBLUE, 1.3)
    m = meta_all()
    m["lamp"] = [round(v) for v in LIGHTS["lamp"]]
    S.save("ln_right", crop=True, anchor=(0, 0), q=90, meta=m)


def shrine(s, r, F, sv, d=0.0):
    """A little 土地公 shrine against the wall: a red box under a curved tiled roof, two red lanterns, an incense pot."""
    sc = F.ppm(sv, 1.0, d)
    pen = max(1.0, sc * 0.015)
    a, b = sv - 0.45, sv + 0.45
    # the stand
    st = F.q(a + 0.05, 0, b - 0.05, 0.6, d + 0.5)
    put(s, Q(s, st), GREY, 0.5, lift=1.0, wob=0.4, edge=0.4, ew=2)
    put(s, Q(s, F.qs(a + 0.05, 0, 0.6, d, d + 0.5)), GREY, 0.5, lift=1.0, wob=0.4)
    s.wash(Q(s, F.qs(a + 0.05, 0, 0.6, d, d + 0.5)), PAYNE, 0.3, wob=0.4, soft=1)
    quad(s, r, st, pen, SEPIA, 1.3, over=2)
    # the box
    bx = F.q(a + 0.1, 0.6, b - 0.1, 1.45, d + 0.45)
    bm = put(s, Q(s, bx), (0.78, 0.26, 0.22), 0.85, lift=1.0, wob=0.4, edge=0.5, ew=2)
    sd = F.qs(a + 0.1, 0.6, 1.45, d, d + 0.45)
    put(s, Q(s, sd), (0.62, 0.20, 0.18), 0.85, lift=1.0, wob=0.4)
    inner = F.q(a + 0.24, 0.7, b - 0.24, 1.25, d + 0.45)
    s.wash(Q(s, inner), SEPIA, 0.9, wob=0.3, soft=0.8, edge=0.4, ew=1)
    gx, gy = F.p(sv, 0.9, d + 0.3)
    s.wash(s.ellipse(gx, gy, 0.06 * sc, 0.09 * sc), OCHRE, 0.7, wob=0.2, soft=0.6)          # the little god, gilt, no face
    s.lift(s.ellipse(*F.p(sv, 1.18, d + 0.4), 0.05 * sc, 0.03 * sc), 0.8, soft=1)            # its lamp
    quad(s, r, bx, pen, SEPIA, 1.4, over=2, dbl=0.3)
    quad(s, r, inner, pen * 0.8, SEPIA, 1.2, over=1)
    # roof: curved, green tiles, upturned ends
    rl, rr = F.p(a - 0.05, 1.45, d + 0.6), F.p(b + 0.05, 1.45, d + 0.6)
    rt = F.p(sv, 1.78, d + 0.3)
    roof = [(rl[0] - 0.08 * sc, rl[1] - 0.08 * sc), rl, rr, (rr[0] + 0.08 * sc, rr[1] - 0.08 * sc), (rt[0] + 0.25 * sc, rt[1]), (rt[0] - 0.25 * sc, rt[1])]
    rm = put(s, s.poly(roof), (0.40, 0.58, 0.48), 0.8, lift=1.0, wob=0.4, edge=0.5, ew=2)
    s.wash(rm, PAYNE, 0.3, wob=0.5, soft=1, grad=s.vgrad(rt[1], rl[1], 0, 1))
    for q in np.linspace(0.1, 0.9, 7):
        x = rl[0] + (rr[0] - rl[0]) * q
        s.ink([(x, rl[1]), (rt[0] + (x - rt[0]) * 0.55, rt[1] + 2)], pen * 0.6, SEPIA, 0.9)
    s.ink(roof + [roof[0]], pen, SEPIA, 1.4)
    # lanterns, an incense pot, oranges
    for sx in (a - 0.12, b + 0.12):
        top = F.p(sx, 1.7, d + 0.5)
        c = F.p(sx, 1.42, d + 0.5)
        s.ink([top, (c[0], c[1] - 0.12 * sc)], pen * 0.6, SEPIA, 1.0)
        lm = s.ellipse(c[0], c[1], 0.1 * sc, 0.13 * sc)
        put(s, lm, (0.86, 0.28, 0.24), 0.8, lift=1.0, wob=0.3, edge=0.5, ew=2)
        s.wash(lm, OCHRE, 0.3, wob=0.5, soft=2, grad=s.rgrad(c[0], c[1], 0, 0.1 * sc, 1, 0))
        s.ink(ell(c[0], c[1], 0.1 * sc, 0.13 * sc, 0, 2 * math.pi, 20), pen * 0.8, SEPIA, 1.2)
        s.ink([(c[0], c[1] + 0.13 * sc), (c[0], c[1] + 0.22 * sc)], pen * 0.6, (0.86, 0.6, 0.2), 1.0)
    px, py = F.p(sv, 0.62, d + 0.5)
    put(s, s.ellipse(px, py - 0.04 * sc, 0.09 * sc, 0.05 * sc), (0.70, 0.56, 0.30), 0.7, lift=1.0, wob=0.2)
    for k in range(3):
        s.ink([(px + (k - 1) * 0.03 * sc, py - 0.06 * sc), (px + (k - 1) * 0.035 * sc, py - 0.3 * sc)], max(0.8, pen * 0.5), SEPIA, 0.9)
    for k in range(3):
        ox, oy = F.p(sv + 0.18 + k * 0.07, 0.64, d + 0.48)
        put(s, s.ellipse(ox, oy - 0.03 * sc, 0.035 * sc, 0.03 * sc), OSMAN, 0.8, lift=1.0, wob=0.2)
    return F.p(sv, 1.18, d + 0.4)


def far():
    S = Sheet(W, H, 212)
    LIGHTS["windows"] = []
    r = rs(212)
    F = L2
    # beyond: taller blocks over the roofs at the end of the lane, pale with distance
    s = Sub(S, 900, 0, 2400, 1000)
    for x0, x1, top, z in ((-6, 3, 19, 70), (3, 12, 15, 64), (10, 24, 21, 80), (-14, -5, 14, 60)):
        q = [P3(x0, 0, z), P3(x1, 0, z), P3(x1, top, z), P3(x0, top, z)]
        m = put(s, Q(s, q), MIST, 0.6, lift=0, wob=0.6, edge=0.4, ew=3, flow=0.5, fs=60, gran=0.4)
        s.wash(m, SLATE, 0.35, wob=1, soft=2, edge=0, grad=s.vgrad(q[3][1], q[0][1], 1, 0.4))
        s.wash(m, DUSK, 0.22, wob=2, bleed=8, edge=0.1)
        quad(s, r, q, 1.2, SLATE, 1.1, over=3, dbl=0.2)
        for k in range(2, int(top / 3)):
            yy = k * 3.0
            line(s, r, P3(x0, yy, z), P3(x1, yy, z), 1.0, SLATE, 0.8, over=1, skip=0.3)
            for xx in np.arange(x0 + 0.8, x1 - 1.2, 2.2):
                wq = [P3(xx, yy + 0.8, z), P3(xx + 1.2, yy + 0.8, z), P3(xx + 1.2, yy + 2.1, z), P3(xx, yy + 2.1, z)]
                s.wash(Q(s, wq), SLATE, 0.5, wob=0.4, soft=0.6, edge=0.4, ew=1)
        tx = (x0 + x1) / 2
        q = [P3(tx - 0.8, top, z), P3(tx + 0.8, top, z), P3(tx + 0.8, top + 1.6, z), P3(tx - 0.8, top + 1.6, z)]
        put(s, Q(s, q), SLATE, 0.5, lift=0.6, wob=0.4, edge=0.4, ew=2)
        quad(s, r, q, 1.0, SLATE, 1.0, over=1)
    # the turned lane's floor
    g = [P3(-2.6, 0, 18.0), P3(-2.1, 0, TURN), tuple(L2.p(16, 0)), P3(2.1 + 16 * math.sin(TH), 0, 19.6 + 16 * math.cos(TH)), P3(2.6, 0, 17.0)]
    s = Sub(S, 1000, 700, 2600, 1150)
    gm = put(s, Q(s, g), SLATE, 0.6, lift=0, wob=1, edge=0.3, ew=4, flow=0.6, fs=120, gran=0.45)
    s.wash(gm, PAYNE, 0.3, wob=2, bleed=8, edge=0.1, flow=0.7, fs=60)
    for x, z, rx in ((0.4, 22.0, 0.8), (2.4, 24.5, 1.0)):
        pts = [P3(x + rx * math.cos(t), 0, z + rx * 0.6 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 12)[:-1]]
        s.lift(s.poly(pts), 0.6, soft=1, wob=1.5)
    # the wall the lane turns along: an old two-storey house with the shrine at its foot, then a tiled block
    s = house(S, F, r, 6.5, 16, 9.8, TILE, tiled=True, storeys=(3.3, 6.5, 9.8), last=True)
    s = Sub(S, *s_box(F, 6.3, 16, 0, 11))
    shutter(s, r, F, 7.2, 10.8, 2.6, SLATE, 0.5)
    for y0 in (4.2, 7.4):
        window(s, r, F, 7.4, 9.4, y0, y0 + 1.3)
        wcage(s, r, F, 7.4, 9.4, y0, y0 + 1.3, plants=1, deco=int(r.integers(0, 3)))
    balc(s, r, F, 10.0, 14.0, 3.3, 6.5, TILE, True, 1)
    s = house(S, F, r, -0.6, 6.5, 6.8, PLASTER, tiled=False, storeys=(3.2, 6.8))
    s = Sub(S, *s_box(F, -0.8, 6.7, 0, 8))
    window(s, r, F, 3.6, 5.4, 1.0, 2.1)
    wcage(s, r, F, 3.6, 5.4, 1.0, 2.1, plants=2, deco=0)
    window(s, r, F, 1.2, 3.2, 4.2, 5.4)
    wcage(s, r, F, 1.2, 3.2, 4.2, 5.4, plants=3, deco=1, rust=True)
    acunit(s, r, F, "split", 4.2, 4.3)
    canopy(s, r, F, -0.4, 6.4, 2.75, 0.8)
    LIGHTS["shrine"] = shrine(s, r, F, 2.0, 0.0)
    for sv, d, sz in ((0.6, 0.35, 1.6), (3.2, 0.4, 1.2), (5.9, 0.3, 2.0)):
        plant_pot(s, r, F, sv, 0, d, sz)
    for sv in (1.0, 4.0, 5.6):
        plant_pot(s, r, F, sv, 6.8, 0.05, 1.3)
    gcables(s, r, F, -0.6, 16, 3.0, 3, 3.0)
    m = meta_all()
    m["shrine"] = [round(v) for v in LIGHTS["shrine"]]
    S.save("ln_far", crop=True, anchor=(0, 0), q=90, meta=m)


def ground():
    """The lane floor up to the bend: wet concrete, puddles left pale (they will hold the sky), a grate, 慢, the edge lines."""
    S = Sheet(W, H, 213)
    r = rs(213)
    s = Sub(S, 0, 860, W, H)
    g = [P3(-2.6, 0, 0.25), P3(-2.6, 0, 20.4), P3(0.0, 0, 21.6), P3(2.6, 0, 19.2), P3(2.6, 0, 0.25)]
    gm = Q(s, g)
    gm = s.wash(gm, GREY, 0.75, wob=2, soft=1, edge=0.3, ew=6, flow=0.6, fs=220, gran=0.5, body=1)
    s.wash(gm, SLATE, 0.45, wob=0, soft=0, edge=0, flow=0.7, fs=160, gran=0.45, clip=gm)
    s.wash(gm, SEPIA, 0.15, wob=3, bleed=14, edge=0.1, flow=0.9, fs=90, gran=0.4, clip=gm)
    s.wash(gm, PAYNE, 0.3, wob=0, soft=0, edge=0, grad=s.vgrad(1000, H, 0, 1), flow=0.6, fs=120, clip=gm)
    # concrete slabs: joints across the lane, a long crack, patches
    for z in (3.2, 5.4, 8.1, 11.3, 15.0):
        a, b = P3(-2.6, 0, z), P3(2.6, 0, z + J(r, 0.1))
        line(s, r, a, b, max(1.0, 1450 * 0.006 / z * 3), SEPIA, 0.9, over=0, skip=0.25)
    for _ in range(3):
        x, z = U(r, -1.8, 1.8), U(r, 3, 12)
        pts = [P3(x, 0, z)]
        for k in range(6):
            x, z = x + J(r, 0.35), z + U(r, 0.2, 0.7)
            pts.append(P3(x, 0, z))
        s.ink(pts, 1.4, SEPIA, 0.8, smooth=False, skip=0.25, tremor=1.0)
    # puddles: lifted pale shapes on the ground plane
    for x, z, rx, rz in ((-0.6, 3.6, 0.9, 0.5), (0.9, 5.2, 1.1, 0.6), (-1.2, 7.4, 0.8, 0.5), (0.4, 9.4, 1.3, 0.8), (1.4, 12.0, 0.7, 0.5),
                         (-0.8, 13.6, 1.0, 0.7), (0.3, 16.8, 1.2, 0.9), (-1.6, 4.8, 0.5, 0.3), (1.6, 7.0, 0.4, 0.3)):
        pts = [P3(x + rx * math.cos(t) * U(r, 0.8, 1.15), 0, z + rz * math.sin(t) * U(r, 0.8, 1.15)) for t in np.linspace(0, 2 * math.pi, 14)[:-1]]
        pm = s.poly(pts)
        s.lift(pm, 0.62, soft=2.0, wob=3)
        s.wash(pm, MIST, 0.1, wob=2, soft=2, edge=0.7, ew=4)
        s.wash(pm, SLATE, 0.12, wob=3, bleed=6, edge=0.3, flow=0.8, fs=30)
        ctr(s, pm, max(1.0, 1450 * 0.004 / z * 3), SEPIA, 0.6, step=4, skip=0.45)
    # the drain grate by the right wall, a manhole in the middle
    q = [P3(1.6, 0, 6.0), P3(2.1, 0, 6.0), P3(2.1, 0, 7.4), P3(1.6, 0, 7.4)]
    put(s, Q(s, q), SEPIA, 0.6, lift=0.8, wob=0.5, edge=0.4, ew=2)
    for k in range(10):
        z = 6.0 + 1.4 * (k + 0.5) / 10
        line(s, r, P3(1.6, 0, z), P3(2.1, 0, z), 1.6, INK, 1.2, over=0)
    quad(s, r, q, 1.8, INK, 1.3, over=2)
    mx, mz, mr = -0.2, 11.0, 0.35
    pts = [P3(mx + mr * math.cos(t), 0, mz + mr * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 30)[:-1]]
    put(s, s.poly(pts), SLATE, 0.6, lift=0.8, wob=0.4, edge=0.5, ew=2)
    s.ink(pts + [pts[0]], 1.4, SEPIA, 1.2)
    # the red no-parking lines along both walls
    for x0 in (-2.0, 1.95):
        for k, dx in enumerate((0.0, 0.12)):
            a, b = P3(x0 + (dx if x0 < 0 else -dx), 0, 0.6), P3(x0 + (dx if x0 < 0 else -dx), 0, 19.5)
            s.stroke([a, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), b], 9 if k == 0 else 5, (0.80, 0.30, 0.26), 0.7, dry=0.35, edge=0.3, wob=1.5,
                     taper=(0.0, 0.3), profile=lambda t: 1 - 0.9 * t)
    # 慢, painted large on the ground and squashed by the distance
    from PIL import Image, ImageDraw, ImageFont
    card = Image.new("L", (600, 600), 0)
    ImageDraw.Draw(card).text((300, 300), "慢", 255, ImageFont.truetype(FONT, 520), anchor="mm")
    card = np.asarray(card, np.float32) / 255
    zc, wd, ln = 6.2, 1.5, 2.6
    dst = np.float32([P3(-0.45 - wd / 2, 0, zc + ln / 2), P3(-0.45 + wd / 2, 0, zc + ln / 2), P3(-0.45 + wd / 2, 0, zc - ln / 2), P3(-0.45 - wd / 2, 0, zc - ln / 2)])
    dst -= (s.ox, s.oy)
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [600, 0], [600, 600], [0, 600]]), dst)
    card = cv2.GaussianBlur(card, (0, 0), 1.2)
    tm = cv2.warpPerspective(card, Hm, (s.w, s.h), flags=cv2.INTER_LINEAR)
    tm *= sstep(0.3, 0.7, s.P + 0.4 * fbm(s.h, s.w, 8, 5, 2))                        # worn by tyres
    s.lift(tm, 0.8, soft=0.6, wob=1.2)
    s.wash(tm, CREAM, 0.15, wob=1, soft=0.8, edge=0.3)
    # the wet: the walls' pale shapes trailing down into the floor
    for x, z0, z1 in ((-2.4, 1.5, 9), (2.4, 2.0, 12), (-2.4, 10, 18), (2.4, 12, 18)):
        pts = [P3(x * (1 - 0.15 * q), 0, z0 + (z1 - z0) * q) for q in np.linspace(0, 1, 6)]
        s.stroke(pts, 40, None, 0.18, taper=(0.2, 0.5), wob=3, edge=0, gran=0, flow=0, lift=True, dry=0.3)
    m = meta_all()
    m["windows"] = []          # the windows are listed on ln_left, ln_right and ln_far, each its own
    S.save("ln_ground", crop=True, anchor=(0, 0), q=90, meta=m)


def fg():
    """Very near things on the edges: a scooter parked at the left (seen from behind), a big glazed pot, a styrofoam box
    of spring onions and a blue bucket at the right."""
    S = Sheet(W, H, 214)
    r = rs(214)
    s = Sub(S, 0, 700, 1300, H)
    # ---- the scooter: its own frame, rear at z0, its long axis down the lane, close to the left wall
    xc, z0 = -1.5, 1.95

    def SP(lx, ly, lz):
        return P3(xc + lx, ly, z0 + lz)

    def face(pts, col, k=0.8, lift=1.0, **kw):
        return put(s, Q(s, [SP(*p) for p in pts]), col, k, lift=lift, wob=0.6, edge=0.5, ew=4, **kw)

    def pen(pts, w=2.4, c=INK, k=1.4, closed=False, **kw):
        P = [SP(*p) for p in pts]
        if closed:
            P = P + [P[0]]
        s.ink(P, w, c, k, **kw)

    BODY = (0.90, 0.87, 0.80)
    # the front wheel and the apron, farthest, first
    fw = [SP(-0.06, 0.0, 1.55), SP(0.06, 0.0, 1.55), SP(0.06, 0.5, 1.55), SP(-0.06, 0.5, 1.55)]
    iron(s, Q(s, fw), 0.9)
    face([(-0.27, 0.32, 1.25), (0.27, 0.32, 1.25), (0.3, 0.98, 1.3), (-0.3, 0.98, 1.3)], BODY, 0.6)
    face([(0.27, 0.32, 1.25), (0.3, 0.98, 1.3), (0.3, 1.0, 1.6), (0.27, 0.3, 1.6)], BODY, 0.6)
    s.wash(Q(s, [SP(0.27, 0.32, 1.25), SP(0.3, 0.98, 1.3), SP(0.3, 1.0, 1.6), SP(0.27, 0.3, 1.6)]), RAIN, 0.35, wob=1, soft=2)
    head = [(-0.22, 0.98, 1.32), (0.22, 0.98, 1.32), (0.2, 1.12, 1.45), (-0.2, 1.12, 1.45)]
    face(head, BODY, 0.55)
    pen([(-0.27, 0.32, 1.25), (-0.3, 0.98, 1.3), (0.3, 0.98, 1.3), (0.27, 0.32, 1.25)], 2.2)
    pen(head, 2.0, closed=True)
    # handlebar, grips, round mirrors on long stalks
    hb = [SP(-0.36, 1.1, 1.42), SP(-0.1, 1.14, 1.42), SP(0.1, 1.14, 1.42), SP(0.36, 1.1, 1.42)]
    iron(s, s.ribbon(hb, 12), 0.9)
    for sd in (-1, 1):
        iron(s, s.ribbon([SP(sd * 0.36, 1.1, 1.42), SP(sd * 0.27, 1.11, 1.42)], 20, smooth=False), 1.0)
        a, b = SP(sd * 0.24, 1.12, 1.42), SP(sd * 0.3, 1.48, 1.45)
        s.ink([a, ((a[0] + b[0]) / 2 + sd * 6, (a[1] + b[1]) / 2), b], 4, INK, 1.5)
        mm = s.ellipse(b[0], b[1] - 16, 34, 26)
        iron(s, mm, 0.9)
        s.wash(s.ellipse(b[0] - 3, b[1] - 18, 24, 17), RAIN, 0.4, wob=0.3, soft=1)
        hl(s, [(b[0] - 18, b[1] - 26), (b[0] - 4, b[1] - 32)], 4, 0.5)
        s.ink(ell(b[0], b[1] - 16, 34, 26, 0, 2 * math.pi, 30), 2.4, INK, 1.4)
    # floorboard (we look down on it), with a mat
    face([(-0.19, 0.3, 0.72), (0.2, 0.3, 0.72), (0.2, 0.3, 1.26), (-0.19, 0.3, 1.26)], SLATE, 0.7)
    s.wash(Q(s, [SP(-0.15, 0.31, 0.8), SP(0.16, 0.31, 0.8), SP(0.16, 0.31, 1.2), SP(-0.15, 0.31, 1.2)]), SEPIA, 0.8, wob=0.5, soft=1)
    # the side panel along the lane side and the tail, the rear wheel under it
    rw = [SP(-0.065, 0.0, 0.28), SP(0.065, 0.0, 0.28), SP(0.065, 0.52, 0.3), SP(-0.065, 0.52, 0.3)]
    rwm = iron(s, s.poly(rw, smooth=True), 1.0)
    side = [(0.26, 0.34, 0.02), (0.27, 0.34, 0.72), (0.24, 0.62, 0.78), (0.25, 0.86, 0.72), (0.26, 0.86, 0.05)]
    sm = face(side, BODY, 0.6, flow=0.5, fs=40)
    s.wash(sm, RAIN, 0.4, wob=1, soft=3, edge=0, grad=s.vgrad(SP(0.26, 0.86, 0.4)[1], SP(0.26, 0.34, 0.4)[1], 0.2, 1))
    s.wash(sm, SEPIA, 0.15, wob=2, bleed=6, edge=0.2, flow=0.8, fs=20)
    rear = [(-0.25, 0.4, 0.0), (0.26, 0.4, 0.0), (0.26, 0.86, 0.04), (-0.25, 0.86, 0.04)]
    rm = face(rear, BODY, 0.6)
    s.wash(rm, SLATE, 0.3, wob=1, soft=3, edge=0, grad=s.hgrad(SP(-0.25, 0.6, 0)[0], SP(0.26, 0.6, 0)[0], 0.8, 0.2))
    # tail light, indicators, the plate
    tl = [(-0.2, 0.68, 0.02), (0.21, 0.68, 0.02), (0.2, 0.78, 0.03), (-0.19, 0.78, 0.03)]
    tm = face(tl, (0.80, 0.24, 0.22), 0.9)
    hl(s, [SP(-0.16, 0.76, 0.03), SP(0.0, 0.765, 0.03)], 5, 0.45)
    for sd in (-1, 1):
        face([(sd * 0.21, 0.68, 0.02), (sd * 0.26, 0.68, 0.02), (sd * 0.26, 0.76, 0.03), (sd * 0.21, 0.76, 0.03)], (0.95, 0.62, 0.24), 0.8)
    plate = [(-0.11, 0.42, -0.02), (0.11, 0.42, -0.02), (0.11, 0.56, -0.01), (-0.11, 0.56, -0.01)]
    pm = face(plate, (0.97, 0.96, 0.92), 0.25)
    from PIL import Image, ImageDraw, ImageFont
    card = Image.new("L", (440, 280), 0)
    ImageDraw.Draw(card).text((220, 150), "MKX-3721", 255, ImageFont.truetype(FONT, 92), anchor="mm")
    card = cv2.GaussianBlur(np.asarray(card, np.float32) / 255, (0, 0), 1.0)
    dst = (np.float32([SP(*plate[3]), SP(*plate[2]), SP(*plate[1]), SP(*plate[0])]) - (s.ox, s.oy)).astype(np.float32)
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [440, 0], [440, 280], [0, 280]]), dst)
    s.wash(cv2.warpPerspective(card, Hm, (s.w, s.h)), INK, 0.9, wob=0.3, soft=0.4, edge=0)
    pen(plate, 1.8, closed=True)
    # the seat: dark, a wet shine along it; a helmet left on it
    seat_top = [(-0.2, 0.88, 0.06), (0.21, 0.88, 0.06), (0.2, 0.9, 0.7), (-0.19, 0.9, 0.7)]
    seat_side = [(0.21, 0.82, 0.04), (0.21, 0.88, 0.06), (0.2, 0.9, 0.7), (0.2, 0.84, 0.72)]
    seat_back = [(-0.2, 0.82, 0.04), (0.21, 0.82, 0.04), (0.21, 0.88, 0.06), (-0.2, 0.88, 0.06)]
    stm = face(seat_top, SEPIA, 1.1)
    s.wash(stm, INKBLUE, 0.5, wob=1, soft=2, clip=stm)
    face(seat_side, SEPIA, 1.2)
    face(seat_back, SEPIA, 1.3)
    hl(s, [SP(0.05, 0.9, 0.12), SP(0.08, 0.91, 0.45), SP(0.06, 0.91, 0.62)], 10, 0.4, dry=0.25)
    hx, hy = SP(-0.02, 1.02, 0.42)
    hm = s.blob(hx, hy, 118, 92, irr=0.05, n=14)
    hm = put(s, hm, (0.95, 0.93, 0.86), 0.35, lift=1.0, wob=0.5, edge=0.5, ew=5)
    s.wash(hm, RAIN, 0.5, wob=1, soft=4, edge=0, grad=s.hgrad(hx - 118, hx + 118, 0.1, 1), clip=hm)
    s.wash(s.ribbon([(hx - 60, hy - 86), (hx - 10, hy - 30), (hx + 40, hy + 40)], 30), TERRA, 0.6, wob=0.6, soft=1, clip=hm)     # its stripe
    rimm = s.ribbon(ell(hx, hy + 20, 116, 60, 0.1, math.pi - 0.1, 30), 16)
    put(s, rimm, SEPIA, 1.0, lift=0.8, wob=0.4)
    hl(s, [(hx - 70, hy - 50), (hx - 30, hy - 74)], 8, 0.55)
    ctr(s, hm, 2.6, INK, 1.4, step=3)
    s.ink([(hx + 90, hy + 50), (hx + 110, hy + 120), (hx + 80, hy + 170)], 3, INK, 1.4)                                       # its strap
    # the grab rail round the back of the seat
    rail = [SP(-0.21, 0.86, 0.5), SP(-0.23, 0.9, 0.08), SP(0.0, 0.92, -0.02), SP(0.23, 0.9, 0.08), SP(0.22, 0.86, 0.5)]
    iron(s, s.ribbon(rail, 13), 0.9)
    s.ink(rail, 2.4, INK, 1.4)
    hl(s, rail[1:4], 3, 0.4)
    # pen round the body, the panel line, the wheel
    pen(side, 2.6)
    pen(rear, 2.4, closed=True)
    pen([(0.26, 0.6, 0.1), (0.26, 0.58, 0.68)], 1.6, k=0.9, skip=0.2)
    pen([(0.27, 0.34, 0.72), (0.27, 0.32, 1.25)], 2.2)
    # its shadow on the wet ground
    gq = [SP(-0.4, 0.0, -0.15), SP(0.45, 0.0, -0.15), SP(0.45, 0.0, 1.8), SP(-0.4, 0.0, 1.8)]
    s.wash(Q(s, gq), PAYNE, 0.4, wob=2, soft=10, edge=0)

    # ---- the right edge: a big glazed pot of broad leaves, spring onions in a foam box, a blue bucket
    t = Sub(S, 2000, 500, W, H)
    F = R1

    def leaf(cx, cy, L, wd, ang, col=MOSS, k=0.9):
        ca, sa = math.cos(ang), math.sin(ang)
        pts = []
        for q in np.linspace(0, 1, 9):
            w_ = wd * math.sin(math.pi * q) ** 0.8
            pts.append((cx + ca * L * q - sa * w_, cy + sa * L * q + ca * w_))
        for q in np.linspace(1, 0, 9)[1:-1]:
            w_ = wd * math.sin(math.pi * q) ** 0.8
            pts.append((cx + ca * L * q + sa * w_, cy + sa * L * q - ca * w_))
        m = t.poly(pts)
        m = put(t, m, col, k, lift=1.0, wob=0.6, edge=0.55, ew=4, flow=0.6, fs=30, gran=0.4)
        t.wash(m, PRUSSIAN, 0.5, wob=1, soft=3, edge=0, grad=t.hgrad(cx - L, cx + L, 0.2, 1), clip=m)
        t.wash(m, LEAF, 0.35, wob=2, bleed=6, edge=0.2, flow=0.9, fs=14, clip=m)
        hl(t, [(cx + ca * L * 0.1, cy + sa * L * 0.1), (cx + ca * L * 0.85, cy + sa * L * 0.85)], 3, 0.35)
        ctr(t, m, 2.0, INK, 1.3, step=3)
        t.ink([(cx, cy), (cx + ca * L * 0.9, cy + sa * L * 0.9)], 1.4, INK, 0.9, skip=0.15)
        return m

    # the foam box at the back, white, with the onions standing in it
    bx = [P3(1.25, 0, 3.3), P3(1.95, 0, 3.3), P3(1.95, 0.32, 3.3), P3(1.25, 0.32, 3.3)]
    btop = [P3(1.25, 0.32, 3.3), P3(1.95, 0.32, 3.3), P3(1.95, 0.32, 3.8), P3(1.25, 0.32, 3.8)]
    bm = put(t, Q(t, bx), (0.97, 0.96, 0.93), 0.2, lift=1.0, wob=0.5, edge=0.6, ew=4)
    t.wash(bm, RAIN, 0.3, wob=1, soft=3, edge=0, grad=t.hgrad(bx[0][0], bx[1][0], 0, 1))
    tm_ = put(t, Q(t, btop), SEPIA, 0.6, lift=1.0, wob=0.5)
    path(t, bx, 2.2, INK, 1.3, closed=True, over=3, r=r)
    path(t, [btop[3], btop[0], btop[1], btop[2]], 1.8, INK, 1.2, over=2, r=r)
    for i in range(36):
        x0 = U(r, 1.3, 1.9)
        z0_ = U(r, 3.35, 3.75)
        a = P3(x0, 0.32, z0_)
        b = P3(x0 + J(r, 0.06), 0.32 + U(r, 0.35, 0.6), z0_ + J(r, 0.05))
        c = (b[0] + J(r, 30), b[1] + U(r, 10, 60))
        t.stroke([a, b, c] if r.random() < 0.3 else [a, b], U(r, 6, 10), MOSS if r.random() < 0.6 else LEAF, U(r, 0.7, 1.0), taper=(0.0, 0.8), wob=0.5, edge=0.4, body=1)
        t.ink([a, b], 1.0, INK, 0.7, skip=0.3)
    # the bucket
    bucket(t, r, F, 2.75, 1.0, 0.21, 0.42, (0.36, 0.56, 0.80))
    # the big pot, glazed blue-green, nearest
    px, pz, pr, ph = 1.5, 2.45, 0.3, 0.52
    top_c, bot_c = P3(px, ph, pz), P3(px, 0, pz)
    sc = 1450 / pz
    L_SCALE = sc / 590
    rx = pr * sc
    ry = rx * 0.32
    potm = np.maximum(t.poly([(top_c[0] - rx, top_c[1]), (top_c[0] + rx, top_c[1]), (bot_c[0] + rx * 0.72, bot_c[1]), (bot_c[0] - rx * 0.72, bot_c[1])], smooth=False),
                      t.ellipse(bot_c[0], bot_c[1], rx * 0.72, ry * 0.8))
    potm = put(t, potm, TEAL, 0.9, lift=1.0, wob=0.6, edge=0.5, ew=5, flow=0.6, fs=40, gran=0.5)
    t.wash(potm, PRUSSIAN, 0.6, wob=1, soft=4, edge=0, grad=t.hgrad(top_c[0] - rx, top_c[0] + rx, 0.1, 1), clip=potm)
    t.wash(potm, SEPIA, 0.2, wob=2, bleed=6, edge=0.2, flow=0.9, fs=16, clip=potm)
    hl(t, [(top_c[0] - rx * 0.55, top_c[1] + 30), (bot_c[0] - rx * 0.45, bot_c[1] - 30)], 12, 0.45, dry=0.2)
    t.ink([(top_c[0] - rx, top_c[1]), (bot_c[0] - rx * 0.72, bot_c[1])], 2.6, INK, 1.4)
    t.ink([(top_c[0] + rx, top_c[1]), (bot_c[0] + rx * 0.72, bot_c[1])], 2.8, INK, 1.5)
    t.ink(ell(bot_c[0], bot_c[1], rx * 0.72, ry * 0.8, 0, math.pi, 30), 2.4, INK, 1.4)
    rim = t.ribbon(ell(top_c[0], top_c[1], rx, ry, 0, 2 * math.pi, 40), 18, closed=True)
    put(t, rim, TEAL, 0.9, lift=1.0, wob=0.4)
    put(t, t.ellipse(top_c[0], top_c[1], rx * 0.92, ry * 0.85), SEPIA, 1.0, lift=1.0, wob=0.4)
    t.ink(ell(top_c[0], top_c[1], rx, ry, 0, 2 * math.pi, 40), 2.6, INK, 1.4)
    # broad leaves out of it, every way, some hanging over the rim
    for i in range(16):
        ang = U(r, -2.9, -0.2) if i < 12 else U(r, 0.3, 1.2) + (math.pi if r.random() < 0.5 else 0)
        L = U(r, 160, 300) * L_SCALE
        bxs = top_c[0] + J(r, rx * 0.5)
        stem_end = (bxs + math.cos(ang) * L * 0.35, top_c[1] - 20 + math.sin(ang) * L * 0.35)
        t.stroke([(bxs, top_c[1] - 5), stem_end], 7, LEAF, 0.9, taper=(0, 0.3), wob=0.4, edge=0.3, body=1)
        leaf(stem_end[0], stem_end[1], L * 0.75, L * 0.24, ang + J(r, 0.25), MOSS if i % 3 else LEAF, U(r, 0.75, 1.0))
    S.save("ln_fg", crop=True, anchor=(0, 0), q=90, meta={"p": 1.4})


ALL = {"lane_sky_dusk": sky_dusk, "lane_sky_day": sky_day, "lane_far": far, "lane_ground": ground, "lane_left": left, "lane_right": right, "lane_fg": fg}
