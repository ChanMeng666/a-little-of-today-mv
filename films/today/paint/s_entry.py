"""The entry (玄关) of the old flat on a grey morning, and the umbrella that came home bent.

entry, sheet 2560x1440. One-point perspective from inside the flat, eye 1.75 m up (a little high), looking at the
front door: x = 1280 + F*X/Z, y = 315 + F*(EYE - h)/Z with F = 1344 (X metres right of centre, h metres up, Z metres
away). The back wall (door wall) stands at Z = 3.2 (420 px per metre there; its floor line is y 1050). The entry
floor is small square tiles; the flat's wooden floor is a step up, its edge across the bottom at y ~1364.
  en_bg        opaque  door wall with the green steel security door (peephole, chain, lever, an upside-down 福), the
                       light-wood shoe cabinet on the right with keys in a dish, the frosted window above it, the
                       canvas tote on a hook and the intercom on the left wall, sneakers, slippers, the doormat, the
                       step. meta: umb [x, y, deg] where en_umb's tip stands and its lean (deg clockwise from up),
                       umb_scale (scale to apply to en_umb / en_umb_fixed here), puddle [x, y, rx, ry] where the drips
                       pool, window [x0, y0, x1, y1] (frosted window bbox, the cool light), door [x0, y0, x1, y1],
                       peephole [x, y], keys [x, y].
  en_umb       sprite  your long umbrella, painted upright (rotate by umb deg), navy, loosely furled, one bent rib
                       sticking out, the strap hanging open, wet. Anchor: the tip on the floor.
                       meta tip_drip [dx, dy] (the lowest point of the fabric, where drops fall from), rib [dx, dy]
                       (the bent rib's end), strap [dx, dy] (the strap's loose end, it can sway), len (tip to handle).
  en_umb_fixed sprite  the same umbrella neatly furled, strap fastened, rib straight. Same anchor and meta keys.

fix, sheet 2560x1440: the umbrella opened on the entry floor, seen from above at an angle, inside up.
  fx_bg        opaque  wet small-square floor tiles, the doormat's edge (top left), a towel (bottom right).
  fx_umb       sprite  the opened canopy from the inside, ribs and stretchers showing, one rib bent so its panel sags.
                       Anchor: the hub. meta ribtips [[dx, dy]..] (rib ends, the drip points), bent [dx, dy] (the
                       kink in the bent rib), shaft [dx, dy] (where the shaft leaves the frame side).
  fx_hands     sprite  both my hands (cream cuffs) gripping the bent rib, wet. Anchor: the grip point (place it on
                       fx_umb's bent). meta drips [[dx, dy]..] where water runs off the hands.
"""
import math

import cv2
import numpy as np

from kit import *

SW, SH = 2560, 1440
F, EYE, VPX, VPY = 1344.0, 1.75, 1280.0, 315.0
ZB = 3.2                       # the door wall
WL, WR = -1.1, 1.1             # side walls
ZS, STEP = 2.05, 0.15          # the step up into the flat: its edge, its height


def P(X, h, Z):
    return (VPX + F * X / Z, VPY + F * (EYE - h) / Z)


def _lift(s, m, amt, clip=None, soft=1.5, wob=1.2, bleed=0.0, dry=0.0, gran=0.15, flow=0.0, fs=80):
    return s.wash(m, None, amt, wob=wob, soft=soft, edge=0, gran=gran, flow=flow, fs=fs, bleed=bleed, dry=dry, lift=True, clip=clip)


def _opaque(s, m, color, k, **kw):
    inner = cv2.erode(m, np.ones((3, 3), np.uint8))
    s.A *= (1 - cv2.GaussianBlur(inner, (0, 0), 0.6))[..., None]
    return s.wash(m, color, k, body=1, **kw)


def QP(s, pts):
    return Q(s, [P(*p) for p in pts])


def _floor_fields(s, m, h, cell, ox=0.0, oz=0.0):
    """Tile fields on a floor plane at height h: per-tile random, grout lines (screen px from the nearest line)."""
    Z = F * (EYE - h) / np.maximum(s.Y - VPY, 1.0)
    X = (s.X - VPX) * Z / F
    u, v = (X - ox) / cell, (Z - oz) / cell
    iu, iv = np.floor(u), np.floor(v)
    hh = np.sin(iu * 12.9898 + iv * 78.233) * 43758.5453
    rnd = (hh - np.floor(hh)).astype(np.float32)
    du = np.minimum(u - iu, 1 - (u - iu)) * cell * F / Z
    dv = np.minimum(v - iv, 1 - (v - iv)) * cell * F * (EYE - h) / (Z * Z)
    return rnd, np.minimum(du, dv).astype(np.float32), Z, X


# ============================================================================================== the entry
DOOR = (-0.60, 0.25, 2.05)     # X from, X to, height
UMB_TIP = (-0.80, 3.05)        # X, Z of the umbrella's tip on the floor
UMB_TOP = (-0.95, 0.93, 3.17)  # X, h, Z of the top of the handle, resting against the wall
CAB = (0.72, 2.35, 0.92)       # shoe cabinet: front face X, near end Z, height
WIN = (2.48, 3.0, 1.32, 1.92)  # frosted window on the right wall: Z from, Z to, h from, h to


def bg():
    s = Sheet(SW, SH, 301)
    r = rs(301)
    bx0, by0 = P(WL, 2.7, ZB)
    bx1, by1 = P(WR, 0, ZB)
    back = Q(s, [(bx0, -10), (bx1, -10), (bx1, by1), (bx0, by1)])
    lw = Q(s, [(-10, -10), (bx0, -10), (bx0, by1), P(WL, 0, 1.2)])
    rw = Q(s, [(bx1, -10), (SW + 10, -10), (SW + 10, P(WR, 0, 1.25)[1]), (bx1, by1)])
    floor = Q(s, [P(WL, 0, ZB), P(WR, 0, ZB), P(WR, 0, 1.2), P(WL, 0, 1.2)])
    # ---- walls: pale plaster; the cool window light from the right falls on the left wall and the door wall
    walls = np.clip(back + lw + rw, 0, 1)
    s.wash(walls, CREAM, 0.36, wob=0, soft=0, edge=0, flow=0.3, fs=420, gran=0.4)
    s.wash(walls, MIST, 0.20, wob=0, soft=0, edge=0, flow=0.3, fs=500, gran=0.4)
    s.wash(rw, SLATE, 0.32, wob=0, soft=0, edge=0, flow=0.4, fs=300)                                   # the window wall, against its own light
    s.wash(back, SLATE, 0.10, wob=0, soft=0, edge=0, grad=s.hgrad(bx0, bx1, 0.2, 1.0))
    s.wash(lw, PAYNE, 0.22, wob=0, soft=0, edge=0, grad=s.hgrad(0, bx0, 1, 0), flow=0.3)
    s.wash(walls, PAYNE, 0.16, wob=0, soft=0, edge=0, grad=s.vgrad(0, 300, 1, 0))                      # toward the ceiling
    for _ in range(8):
        s.wash(s.blob(r.random() * SW, r.random() * 1100, 160 + 160 * r.random(), 80 + 80 * r.random(), 0.3) * walls, SLATE, 0.035, wob=6, bleed=26, edge=0.7, ew=12)
    # scuffs low on the walls where shoes and bags knock
    for _ in range(14):
        x = r.random() * SW
        y = 900 + r.random() * 300
        s.stroke([(x, y), (x + 20 + 40 * r.random(), y + (r.random() - 0.5) * 8)], 3 + 4 * r.random(), UMBER, 0.15, wob=1, soft=1, dry=0.4, clip=walls, taper=(0.3, 0.4))
    # skirting: one row of floor tile up the walls
    sk = np.maximum(np.maximum(QP(s, [(WL, 0.10, ZB), (WR, 0.10, ZB), (WR, 0, ZB), (WL, 0, ZB)]), QP(s, [(WL, 0.10, ZB), (WL, 0.10, 1.2), (WL, 0, 1.2), (WL, 0, ZB)])), QP(s, [(WR, 0.10, ZB), (WR, 0.10, 1.2), (WR, 0, 1.2), (WR, 0, ZB)]))
    s.wash(sk, TILE, 0.55, wob=0.8, soft=0.8, edge=0.5, gran=0.4)
    # ---- floor: small square tiles, pale grey-green with some cream and some darker, wet by the door
    rnd, gd, Zf, Xf = _floor_fields(s, floor, 0.0, 0.10, ox=0.03)
    base(s, floor, TILE, 0.55, wob=0.8, soft=0.8, edge=0.3, flow=0.4, fs=200, gran=0.5)
    s.wash(floor, CREAM, 0.5, wob=0, soft=0, edge=0, grad=np.clip((rnd - 0.6) * 3, 0, 1), gran=0.3)
    s.wash(floor, SLATE, 0.45, wob=0, soft=0, edge=0, grad=np.clip((0.25 - rnd) * 4, 0, 1), gran=0.4)
    s.wash(floor, CELADON, 0.35, wob=0, soft=0, edge=0, grad=np.clip(1 - np.abs(rnd - 0.45) * 8, 0, 1), gran=0.3)
    gro = np.clip(1.5 - gd, 0, 1) * floor
    s.wash(gro, UMBER, 0.35, wob=0.5, soft=0.6, edge=0, gran=0.3, flow=0.6, fs=60)
    s.wash(floor, PAYNE, 0.25, wob=0, soft=0, edge=0, grad=s.hgrad(VPX + 700, VPX - 500, 0.1, 1) * s.vgrad(1050, 1400, 0.2, 1), flow=0.4)
    _lift(s, floor * s.rgrad(*P(0.5, 0, 2.7), 50, 700, 1, 0, sy=0.3), 0.18, soft=0, wob=0)          # the window's light lying on the tiles
    # ---- the step up into the flat: light wood boards, a nosing along the edge
    ey = P(0, STEP, ZS)[1]
    wood_m = Q(s, [P(WL, STEP, ZS), P(WR, STEP, ZS), (SW + 10, SH + 10), (-10, SH + 10)])
    s.lift(wood_m, 1.0, soft=0.6, wob=0.5)
    base(s, wood_m, SIENNA, 0.55, wob=0.6, soft=0.6, edge=0.4, flow=0.5, fs=200, gran=0.5)
    for X in np.arange(-2.0, 2.0, 0.12):
        a, b = P(X, STEP, ZS), P(X * 1.0, STEP, 1.6)
        s.ink([a, b], 1.2, BURNT, 0.6, skip=0.3, clip=wood_m, tremor=0.8)
    s.wash(wood_m, UMBER, 0.25, wob=0, soft=0, edge=0, grad=s.vgrad(ey, SH, 0, 1))
    nose = Q(s, [P(WL, STEP, ZS), P(WR, STEP, ZS), P(WR, STEP, ZS - 0.05), P(WL, STEP, ZS - 0.05)])
    _lift(s, nose, 0.4, soft=1)
    s.wash(nose, OCHRE, 0.3, wob=0.5, soft=1, edge=0.5)
    seg(s, P(WL, STEP, ZS), P(WR, STEP, ZS), 2.4, INK, 1.4, r=r, over=0)
    s.wash(Q(s, [P(WL, 0, ZS + 0.25), P(WR, 0, ZS + 0.25), P(WR, 0, ZS + 0.12), P(WL, 0, ZS + 0.12)]) * floor, PAYNE, 0.2, wob=1, soft=8)
    # ---- the door: frame, leaf, panels, the hardware
    dX0, dX1, dH = DOOR
    jamb = 0.06
    fr = QP(s, [(dX0 - jamb, dH + jamb, ZB), (dX1 + jamb, dH + jamb, ZB), (dX1 + jamb, 0, ZB), (dX0 - jamb, 0, ZB)])
    leaf = QP(s, [(dX0, dH, ZB), (dX1, dH, ZB), (dX1, 0.02, ZB), (dX0, 0.02, ZB)])
    cast(s, fr, -16, 6, PAYNE, 0.30, blur=12, clip=back)
    base(s, fr, MOSS, 0.55, wob=0.8, soft=0.8, edge=0.5, flow=0.4)
    s.wash(fr, TEAL, 0.45, wob=0.8, soft=0.8, edge=0.3)
    base(s, leaf, MOSS, 0.65, wob=1.0, soft=0.8, edge=0.55, ew=6, flow=0.5, fs=160, gran=0.55)
    s.wash(leaf, TEAL, 0.55, wob=1.0, soft=1, edge=0.3, flow=0.5, fs=200, gran=0.5)
    s.wash(leaf, PAYNE, 0.25, wob=0, soft=0, edge=0, grad=s.hgrad(*[P(dX1, 0, ZB)[0], P(dX0, 0, ZB)[0]], 0.0, 1.0))
    _lift(s, leaf * s.hgrad(P(dX1 - 0.25, 0, ZB)[0], P(dX1, 0, ZB)[0], 0, 1), 0.18, soft=4, wob=1)
    # raised panels pressed in the steel: two, with a lit edge and a shaded edge
    for (h0, h1) in ((1.12, 1.92), (0.14, 0.98)):
        pm = QP(s, [(dX0 + 0.09, h1, ZB), (dX1 - 0.09, h1, ZB), (dX1 - 0.09, h0, ZB), (dX0 + 0.09, h0, ZB)])
        x0p, y0p = P(dX0 + 0.09, h1, ZB)
        x1p, y1p = P(dX1 - 0.09, h0, ZB)
        _lift(s, s.rect(x1p - 6, y0p, x1p, y1p), 0.3, soft=1.5)
        _lift(s, s.rect(x0p, y1p - 5, x1p, y1p), 0.2, soft=1.5)
        s.wash(s.rect(x0p, y0p, x0p + 7, y1p), PAYNE, 0.45, wob=0.6, soft=1.2)
        s.wash(s.rect(x0p, y0p, x1p, y0p + 6), PAYNE, 0.40, wob=0.6, soft=1.2)
        path(s, [(x0p, y0p), (x1p, y0p), (x1p, y1p), (x0p, y1p)], 1.6, INKBLUE, 0.9, closed=True, over=4, r=r, skip=0.15)
    # chips and rust where feet and keys have knocked it
    rust = s.zeros()
    for _ in range(40):
        x = P(dX0, 0, ZB)[0] + r.random() * (P(dX1, 0, ZB)[0] - P(dX0, 0, ZB)[0])
        y = P(0, 0.35, ZB)[1] + r.random() * 130 if r.random() < 0.7 else P(0, 2.0, ZB)[1] + r.random() * 800
        rust = np.maximum(rust, s.blob(x, y, 2 + 6 * r.random(), 1.5 + 3 * r.random(), 0.4, 7))
    s.wash(rust * leaf, SIENNA, 0.65, wob=0.6, soft=0.6, edge=0.7, ew=1.5)
    _lift(s, s.rect(*P(dX0 + 0.1, 0.12, ZB), *P(dX1 - 0.1, 0.03, ZB)) * s.noise(8, 2) * leaf, 0.3, soft=0.8, wob=1, dry=0.3)
    path(s, [P(dX0 - jamb, dH + jamb, ZB), P(dX1 + jamb, dH + jamb, ZB), P(dX1 + jamb, 0, ZB)], 2.4, INK, 1.4, over=6, r=r)
    seg(s, P(dX0 - jamb, dH + jamb, ZB), P(dX0 - jamb, 0, ZB), 2.4, INK, 1.4, r=r, over=6)
    path(s, [P(dX0, 0.02, ZB), P(dX0, dH, ZB), P(dX1, dH, ZB), P(dX1, 0.02, ZB)], 2.0, INK, 1.3, over=4, r=r)
    # threshold
    th_ = QP(s, [(dX0 - jamb, 0.03, ZB), (dX1 + jamb, 0.03, ZB), (dX1 + jamb, 0.0, ZB - 0.06), (dX0 - jamb, 0.0, ZB - 0.06)])
    base(s, th_, MIST, 0.55, wob=0.5, soft=0.6, edge=0.6)
    seg(s, P(dX0 - jamb, 0.0, ZB - 0.06), P(dX1 + jamb, 0.0, ZB - 0.06), 1.6, INK, 1.0, r=r)
    # the closer at the top hinge side
    cx_, cy_ = P(dX0 + 0.05, dH - 0.08, ZB)
    cl = s.rect(cx_, cy_ - 14, cx_ + 92, cy_ + 12)
    base(s, cl, SLATE, 0.6, wob=0.5, soft=0.6, edge=0.6)
    seg(s, (cx_ + 60, cy_ + 2), (cx_ + 10, cy_ - 46), 5, SLATE, 1.5, bow=0)
    path(s, [(cx_, cy_ - 14), (cx_ + 92, cy_ - 14), (cx_ + 92, cy_ + 12), (cx_, cy_ + 12)], 1.5, INK, 1.1, closed=True, over=3, r=r)
    # hinges
    for hh in (0.25, 1.0, 1.8):
        hx, hy = P(dX0 - 0.01, hh, ZB)
        hm = s.rect(hx - 6, hy - 22, hx + 6, hy + 22)
        base(s, hm, SLATE, 0.7, wob=0.4, soft=0.6, edge=0.5)
    # lever, deadbolt, latch, the chain
    lx, ly = P(dX1 - 0.08, 1.0, ZB)
    plate = s.rect(lx - 11, ly - 48, lx + 11, ly + 48)
    base(s, plate, MIST, 0.5, wob=0.5, soft=0.6, edge=0.6)
    s.wash(plate, SLATE, 0.3, wob=0.5, soft=1, grad=s.hgrad(lx - 11, lx + 11, 0, 1))
    lev = s.ribbon([(lx, ly - 12), (lx - 30, ly - 10), (lx - 70, ly - 4)], 12, (0.0, 0.2))
    base(s, lev, MIST, 0.6, wob=0.4, soft=0.6, edge=0.6)
    s.wash(lev, SLATE, 0.4, wob=0.4, soft=1, grad=s.vgrad(ly - 16, ly - 4, 0, 1))
    contour(s, lev, 1.5, INK, 1.2)
    path(s, [(lx - 11, ly - 48), (lx + 11, ly - 48), (lx + 11, ly + 48), (lx - 11, ly + 48)], 1.4, INK, 1.1, closed=True, over=2, r=r)
    for (yy, rr_) in ((ly - 110, 15), (ly - 190, 13)):
        dm = s.ellipse(lx, yy, rr_, rr_)
        base(s, dm, OCHRE, 0.55, wob=0.4, soft=0.6, edge=0.6)
        s.wash(dm, BURNT, 0.4, wob=0.4, soft=1, grad=s.rgrad(lx - 4, yy - 4, 2, rr_, 0, 1))
        s.wash(s.rect(lx - 3, yy - rr_ + 4, lx + 3, yy + rr_ - 4), SEPIA, 0.9, wob=0.3, soft=0.6)
        oval(s, lx, yy, rr_, rr_, 1.4, INK, 1.1)
    # the chain: a slot on the leaf, links sagging to the keeper on the frame
    k0 = (lx - 40, ly - 260)
    k1 = P(dX1 + jamb * 0.5, 1.36, ZB)
    slot = s.rect(k0[0] - 34, k0[1] - 7, k0[0] + 12, k0[1] + 7)
    base(s, slot, MIST, 0.6, wob=0.4, soft=0.6, edge=0.6)
    path(s, [(k0[0] - 34, k0[1] - 7), (k0[0] + 12, k0[1] - 7), (k0[0] + 12, k0[1] + 7), (k0[0] - 34, k0[1] + 7)], 1.3, INK, 1.0, closed=True, over=2)
    t_ = np.linspace(0, 1, 11)
    cp = np.stack([k0[0] + (k1[0] - k0[0]) * t_, k0[1] + (k1[1] - k0[1]) * t_ + 70 * np.sin(np.pi * t_)], 1)
    for i in range(10):
        a, b = cp[i], cp[i + 1]
        c_ = (a + b) / 2
        ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        oval(s, c_[0], c_[1], 9, 4.5 if i % 2 else 2.5, 1.5, SLATE, 1.4, rot=math.radians(ang))
    s.wash(s.ellipse(k1[0], k1[1], 8, 14), SLATE, 0.8, wob=0.3, soft=0.6, body=1)
    # peephole
    px_, py_ = P((dX0 + dX1) / 2, 1.55, ZB)
    s.wash(s.ellipse(px_, py_, 13, 13), OCHRE, 0.6, wob=0.3, soft=0.6, edge=0.6)
    s.wash(s.ellipse(px_, py_, 6, 6), INKBLUE, 1.0, wob=0.3, soft=0.6)
    _lift(s, s.ellipse(px_ - 2, py_ - 2, 2, 2), 0.8, soft=0.5, wob=0)
    oval(s, px_, py_, 13, 13, 1.4, INK, 1.1)
    # the 福 square, put up upside down for luck, faded by many summers
    fx_, fy_ = P((dX0 + dX1) / 2, 1.30, ZB)
    hs = 62
    dia = [(fx_, fy_ - hs), (fx_ + hs, fy_), (fx_, fy_ + hs), (fx_ - hs, fy_)]
    fm = Q(s, dia)
    s.lift(fm, 0.9, soft=0.6, wob=0.5)
    s.wash(fm, ROSE, 0.55, wob=1.0, soft=0.8, edge=0.6, ew=3, flow=0.5, gran=0.5, body=1)
    s.wash(fm, PERSIMMON, 0.25, wob=1.0, soft=1, edge=0.2, grad=s.noise(40, 3))
    _lift(s, fm * s.noise(30, 3), 0.25, soft=2, wob=1)
    tm = text_mask(s, "福", fx_, fy_, 64, "D:/github_repository/a-little-of-today-mv/assets/fonts/LXGWWenKai-Regular.ttf", rot=180)
    s.wash(tm, OCHRE, 0.75, wob=0.5, soft=0.6, edge=0.4)
    s.wash(tm, SEPIA, 0.2, wob=0.5, soft=0.6)
    path(s, dia, 1.4, SEPIA, 0.9, closed=True, over=3, r=r)
    s.wash(s.ellipse(fx_, fy_ - hs + 6, 6, 4), CREAM, 0.6, wob=0.3, soft=0.5, body=1)       # the tape at its corner
    # ---- the intercom handset on the left wall, the hook and the canvas tote
    im = QP(s, [(WL, 1.58, 3.14), (WL, 1.58, 2.92), (WL, 1.30, 2.92), (WL, 1.30, 3.14)])
    cast(s, im, 10, 8, PAYNE, 0.3, blur=6, clip=lw)
    base(s, im, CREAM, 0.55, wob=0.5, soft=0.6, edge=0.6)
    s.wash(im, OCHRE, 0.2, wob=0.5, soft=1)
    hs_ = QP(s, [(WL + 0.02, 1.55, 3.12), (WL + 0.02, 1.55, 3.06), (WL + 0.02, 1.33, 3.06), (WL + 0.02, 1.33, 3.12)])
    base(s, hs_, CREAM, 0.75, wob=0.4, soft=0.6, edge=0.6)
    s.wash(hs_, UMBER, 0.25, wob=0.4, soft=1)
    contour(s, hs_, 1.3, INK, 1.1)
    for hh in np.linspace(1.36, 1.44, 4):
        seg(s, P(WL, hh, 3.02), P(WL, hh, 2.95), 1.0, SEPIA, 0.8, bow=0)
    s.wash(s.ellipse(*P(WL, 1.52, 2.97), 4, 5), ROSE, 0.6, wob=0.2, soft=0.5)
    contour(s, im, 1.6, INK, 1.2)
    hk = P(WL, 1.62, 2.45)
    tz0, tz1 = 2.30, 2.62
    tote = QP(s, [(WL, 1.30, tz0), (WL, 1.30, tz1), (WL + 0.04, 0.84, tz1 + 0.01), (WL + 0.06, 0.82, tz0 - 0.02)])
    cast(s, tote, 22, 14, PAYNE, 0.35, blur=14, clip=lw)
    for (za, zb) in ((tz0 + 0.05, tz0 + 0.12), (tz1 - 0.05, tz1 - 0.13)):
        a, b = P(WL, 1.30, za), P(WL, 1.30, zb)
        s.stroke([a, hk, b], 9, CREAM, 0.6, taper=(0, 0), wob=0.6, edge=0.5, body=1)
        s.ink([a, hk, b], 1.4, SEPIA, 1.0)
    base(s, tote, SOY, 0.75, wob=1.2, edge=0.5, flow=0.5, fs=100, gran=0.6)
    s.wash(tote, OCHRE, 0.25, wob=1.5, bleed=10, grad=s.vgrad(P(0, 1.3, 2.45)[1], P(0, 0.84, 2.45)[1], 0.2, 1))
    s.wash(tote, SLATE, 0.2, wob=1.5, bleed=10, grad=s.hgrad(P(WL, 0, tz0)[0], P(WL, 0, tz1)[0], 0, 1))
    sq = QP(s, [(WL + 0.02, 1.16, 2.40), (WL + 0.02, 1.16, 2.52), (WL + 0.03, 1.02, 2.52), (WL + 0.03, 1.02, 2.40)])   # a small printed square
    s.wash(sq, LEAF, 0.45, wob=0.8, soft=0.8, edge=0.6)
    for _ in range(3):
        a = P(WL + 0.03, 1.25 - r.random() * 0.4, tz0 + r.random() * 0.3)
        s.ink([a, (a[0] + 6, a[1] + 40 + 30 * r.random())], 1.1, SEPIA, 0.5, skip=0.3)
    contour(s, tote, 1.8, SEPIA, 1.2)
    hm_ = s.ellipse(hk[0], hk[1], 7, 7)
    base(s, hm_, SLATE, 0.8, wob=0.3, soft=0.5)
    # ---- the frosted window on the right wall (patterned glass), and the light it lets in
    wz0, wz1, wh0, wh1 = WIN
    wo = QP(s, [(WR, wh1 + 0.04, wz0 - 0.04), (WR, wh1 + 0.04, wz1 + 0.04), (WR, wh0 - 0.04, wz1 + 0.04), (WR, wh0 - 0.04, wz0 - 0.04)])
    wg = QP(s, [(WR, wh1, wz0), (WR, wh1, wz1), (WR, wh0, wz1), (WR, wh0, wz0)])
    base(s, wo, MIST, 0.5, wob=0.5, soft=0.6, edge=0.6)
    s.lift(wg, 1.0, soft=0.8, wob=0.6)
    s.wash(wg, MIST, 0.22, wob=1, soft=1, edge=0.6, ew=4, flow=0.6, fs=60, body=1)
    s.wash(wg, CELADON, 0.12, wob=1, bleed=6, grad=s.vgrad(P(WR, wh1, wz0)[1], P(WR, wh0, wz0)[1], 0, 1))
    # 海棠 pattern: little four-petal flowers pressed in the glass, a lattice of them
    gx0_, gy0_ = P(WR, wh1, wz1)
    gx1_, gy1_ = P(WR, wh0, wz0)
    pat = s.zeros()
    for zz in np.linspace(wz0 + 0.04, wz1 - 0.04, 5):
        for hh in np.linspace(wh0 + 0.05, wh1 - 0.05, 7):
            cxp, cyp = P(WR, hh, zz)
            sc_ = F / zz * 0.022
            for a in range(4):
                an = a * math.pi / 2 + math.pi / 4
                pat = np.maximum(pat, s.ellipse(cxp + math.cos(an) * sc_ * 0.55, cyp + math.sin(an) * sc_ * 0.9, sc_ * 0.5 * 0.6, sc_ * 0.5, math.degrees(an)))
    s.wash(pat * wg, SLATE, 0.22, wob=0.4, soft=0.8, edge=0.8, ew=1.5)
    _lift(s, cv2.warpAffine(pat, np.float32([[1, 0, -1.5], [0, 1, -1.5]]), (SW, SH)) * wg, 0.2, soft=0.6, wob=0.3)
    _lift(s, s.rgrad((gx0_ + gx1_) / 2, (gy0_ + gy1_) / 2, 20, 900, 1, 0, sy=1.2) * walls * (1 - wg), 0.16, soft=0, wob=0)
    path(s, [P(WR, wh1 + 0.04, wz0 - 0.04), P(WR, wh1 + 0.04, wz1 + 0.04), P(WR, wh0 - 0.04, wz1 + 0.04), P(WR, wh0 - 0.04, wz0 - 0.04)], 2.0, INK, 1.3, closed=True, over=5, r=r)
    path(s, [P(WR, wh1, wz0), P(WR, wh1, wz1), P(WR, wh0, wz1), P(WR, wh0, wz0)], 1.5, INK, 1.0, closed=True, over=3, r=r)
    seg(s, P(WR, wh1, (wz0 + wz1) / 2), P(WR, wh0, (wz0 + wz1) / 2), 3.0, SLATE, 1.2, r=r)
    # ---- the shoe cabinet: light wood, two doors on its long face, its near end toward us
    cX, cZ, cH = CAB
    longf = QP(s, [(cX, cH, ZB), (cX, cH, cZ), (cX, 0.08, cZ), (cX, 0.08, ZB)])
    endf = QP(s, [(cX, cH, cZ), (WR, cH, cZ), (WR, 0.0, cZ), (cX, 0.0, cZ)])
    topf = QP(s, [(cX, cH, ZB), (WR, cH, ZB), (WR, cH, cZ), (cX, cH, cZ)])
    kick = QP(s, [(cX + 0.03, 0.08, ZB), (cX + 0.03, 0.08, cZ), (cX + 0.03, 0.0, cZ), (cX + 0.03, 0.0, ZB)])
    s.wash(kick, SEPIA, 0.7, wob=0.6, soft=0.8, edge=0.3, body=1)
    shadow_c = QP(s, [(cX - 0.25, 0, ZB), (cX, 0, ZB), (cX, 0, cZ), (cX - 0.12, 0, cZ - 0.05)])
    s.wash(shadow_c, PAYNE, 0.35, wob=2, bleed=14, edge=0.1)
    wood(s, longf, CREAM, SIENNA, 0.65, streak=0.5, ang=math.atan2(P(cX, cH, cZ)[1] - P(cX, cH, ZB)[1], P(cX, cH, cZ)[0] - P(cX, cH, ZB)[0]), n=18, seed=3, lw=1.1)
    s.wash(longf, OCHRE, 0.25, wob=0.6, soft=1, edge=0.3, flow=0.5)
    s.wash(longf, UMBER, 0.22, wob=0.6, soft=1, edge=0.2)                                   # facing away from the window
    s.wash(longf, SLATE, 0.15, wob=0.6, soft=1, edge=0)
    wood(s, endf, CREAM, SIENNA, 0.65, ang=math.pi / 2, n=12, seed=4, lw=1.1, streak=0.5)
    s.wash(endf, OCHRE, 0.22, wob=0.6, soft=1, edge=0.3, flow=0.5)
    s.wash(endf, UMBER, 0.18, wob=0.6, soft=1, edge=0.2, grad=s.vgrad(P(0, cH, cZ)[1], P(0, 0, cZ)[1], 0, 1))
    wood(s, topf, CREAM, SIENNA, 0.55, ang=0.2, n=10, seed=5, lw=1.0)
    s.wash(topf, OCHRE, 0.18, wob=0.6, soft=1, edge=0.3)
    _lift(s, topf, 0.22, soft=2, wob=1)
    zm = (ZB + cZ) / 2
    for (za, zb) in ((ZB - 0.03, zm + 0.01), (zm - 0.01, cZ + 0.03)):
        door_m = QP(s, [(cX, cH - 0.04, za), (cX, cH - 0.04, zb), (cX, 0.12, zb), (cX, 0.12, za)])
        path(s, [P(cX, cH - 0.04, za), P(cX, cH - 0.04, zb), P(cX, 0.12, zb), P(cX, 0.12, za)], 1.5, SEPIA, 1.0, closed=True, over=3, r=r)
        kx, ky = P(cX, cH - 0.14, (za + zb) / 2 + (0.08 if za > zb else -0.08))
        dm = s.ellipse(kx, ky, 6, 7)
        base(s, dm, OCHRE, 0.7, wob=0.3, soft=0.5, edge=0.6)
        oval(s, kx, ky, 6, 7, 1.2, INK, 1.0)
    for (a, b) in ((P(cX, cH, ZB), P(cX, cH, cZ)), (P(cX, cH, cZ), P(cX, 0, cZ)), (P(cX, cH, cZ), P(WR, cH, cZ)), (P(WR, cH, cZ), P(WR, 0, cZ)), (P(cX, 0.08, cZ), P(cX, 0.08, ZB)), (P(cX, 0, cZ), P(WR, 0, cZ))):
        seg(s, a, b, 2.2, INK, 1.4, r=r, over=5)
    # keys in a little dish on top, a couple of letters
    kx, ky = P(0.90, cH, 2.62)
    letters = QP(s, [(0.80, cH + 0.005, 2.95), (1.02, cH + 0.005, 3.0), (1.04, cH + 0.005, 2.80), (0.82, cH + 0.005, 2.76)])
    base(s, letters, CREAM, 0.25, wob=0.5, soft=0.6, edge=0.6)
    contour(s, letters, 1.3, SEPIA, 0.9)
    s.wash(s.ellipse(kx + 4, ky + 6, 52, 12), UMBER, 0.35, wob=1, bleed=5)
    dish = s.ellipse(kx, ky, 50, 13)
    base(s, dish, CELADON, 0.7, wob=0.4, soft=0.6, edge=0.6)
    s.wash(s.ellipse(kx, ky - 2, 40, 9), SLATE, 0.3, wob=0.4, soft=1)
    oval(s, kx, ky, 50, 13, 1.5, INK, 1.1)
    for (dx_, dy_, a_) in ((-18, -4, 0.3), (6, -2, -0.5), (20, -6, 0.9)):
        k_ = s.ribbon([(kx + dx_, ky + dy_), (kx + dx_ + 26 * math.cos(a_), ky + dy_ + 8 * math.sin(a_))], 5, (0, 0.4))
        base(s, k_, MIST, 0.7, wob=0.3, soft=0.5, edge=0.6)
        s.ink([(kx + dx_, ky + dy_), (kx + dx_ + 26 * math.cos(a_), ky + dy_ + 8 * math.sin(a_))], 1.0, INK, 0.9)
        oval(s, kx + dx_ - 3, ky + dy_, 6, 4, 1.1, SLATE, 1.1)
    oval(s, kx + 30, ky - 8, 10, 6, 1.3, OCHRE, 1.4)                     # a brass key ring
    # ---- the doormat in front of the door
    mat = QP(s, [(-0.62, 0.0, ZB - 0.07), (0.30, 0.0, ZB - 0.07), (0.32, 0.0, 2.72), (-0.64, 0.0, 2.72)])
    base(s, mat, SIENNA, 0.75, wob=1.0, soft=0.8, edge=0.5, flow=0.5, fs=60, gran=0.7)
    s.wash(mat, UMBER, 0.35, wob=0, soft=0, edge=0, grad=s.noise(6, 2), gran=0.6)
    s.wash(mat, OCHRE, 0.3, wob=1, soft=1, dry=0.5, gran=0.6)
    for zz in np.linspace(2.74, ZB - 0.09, 9):
        s.ink([P(-0.62, 0, zz), P(0.30, 0, zz)], 1.0, UMBER, 0.6, skip=0.35, clip=mat)
    contour(s, mat, 1.8, SEPIA, 1.2)
    # ---- shoes: my white sneakers by the cabinet, toes to the door; the slippers by the step, toes to us
    def foot(u):
        """Half-width of a sole (metres) along its length u (-0.135 heel .. +0.135 toe)."""
        return np.interp(u, [-0.135, -0.12, -0.08, 0.0, 0.06, 0.10, 0.125, 0.135], [0.0, 0.026, 0.037, 0.034, 0.047, 0.044, 0.030, 0.0])

    def sole_pts(Xc, Zc, ang, h, k=1.0, n=40):
        ca, sa = math.cos(ang), math.sin(ang)
        us = np.linspace(-0.135, 0.135, n)
        side = [(u, foot(u) * k) for u in us] + [(u, -foot(u) * k) for u in us[::-1]]
        return [P(Xc + v * ca + u * sa, h, Zc + u * ca - v * sa) for u, v in side]

    def sneaker(Xc, Zc, ang):
        ca, sa = math.cos(ang), math.sin(ang)

        def w2(u, v, h):
            return P(Xc + v * ca + u * sa, h, Zc + u * ca - v * sa)
        top = lambda u: np.interp(u, [-0.135, -0.12, -0.09, -0.04, 0.0, 0.08, 0.12, 0.135], [0.03, 0.075, 0.068, 0.062, 0.072, 0.050, 0.034, 0.02])
        pts = sole_pts(Xc, Zc, ang, 0.0) + sole_pts(Xc, Zc, ang, 0.026, 1.04)
        us = np.linspace(-0.13, 0.13, 20)
        pts += [w2(u, foot(u) * 0.75 * vv, top(u)) for u in us for vv in (-1, 0, 1)]
        hp = cv2.convexHull(np.round(np.array(pts) * 4).astype(np.int32)).reshape(-1, 2) / 4.0
        hm = Q(s, hp)
        cast(s, hm, -12, 6, PAYNE, 0.38, blur=7)
        base(s, hm, CREAM, 0.22, wob=0.5, soft=0.6, edge=0.6, ew=2)
        hx0, hx1 = hp[:, 0].min(), hp[:, 0].max()
        s.wash(hm, SLATE, 0.32, wob=0.6, bleed=4, grad=s.hgrad(hx0, hx1, 1, 0.0))        # the side away from the window
        # the rubber sole band: the outline at the sole's top, its lower edge darker
        band = Q(s, sole_pts(Xc, Zc, ang, 0.0)) * (1 - Q(s, sole_pts(Xc, Zc, ang, 0.026, 0.98)))
        _lift(s, band * hm, 0.4, soft=0.6, wob=0.3)
        s.ink(sole_pts(Xc, Zc, ang, 0.026, 1.03)[2:-2], 1.2, SLATE, 0.9, smooth=False)
        # collar and the dark opening, the tongue, laces, toe cap, heel tab
        op = s.poly([w2(-0.085 + 0.045 * math.cos(t), 0.022 * math.sin(t), 0.066 + 0.004 * math.cos(t)) for t in np.linspace(0, 2 * math.pi, 18, endpoint=False)])
        s.wash(op, PAYNE, 0.85, wob=0.3, soft=0.5, edge=0.5)
        s.wash(op * s.vgrad(*sorted([w2(-0.13, 0, 0.07)[1], w2(-0.04, 0, 0.07)[1]]), 1, 0), SEPIA, 0.3, wob=0.3, soft=1)
        for j in range(5):
            u = -0.025 + j * 0.022
            s.ink([w2(u, -0.022, top(u) - 0.002), w2(u + 0.012, 0.022, top(u + 0.012) - 0.002)], 1.2, SLATE, 1.0)
            s.ink([w2(u, 0.022, top(u) - 0.002), w2(u + 0.012, -0.022, top(u + 0.012) - 0.002)], 1.0, SLATE, 0.7)
        s.ink([w2(0.085, -0.04, 0.03), w2(0.10, 0.0, 0.045), w2(0.085, 0.04, 0.03)], 1.1, SLATE, 0.9)
        s.wash(s.ellipse(*w2(-0.132, 0, 0.06), 4, 7), NAVY, 0.6, wob=0.3, soft=0.5)
        contour(s, hm, 1.6, INK, 1.2)
        s.ink([w2(-0.09, -0.035, 0.064), w2(-0.05, -0.04, 0.062), w2(-0.02, -0.03, 0.07)], 1.1, INK, 0.8)

    sneaker(0.36, 2.86, 1.30)
    sneaker(0.40, 2.66, 1.05)

    def slipper(Xc, Zc, ang):
        ca, sa = math.cos(ang), math.sin(ang)

        def w2(u, v, h):
            return P(Xc + v * ca + u * sa, h, Zc + u * ca - v * sa)
        so = sole_pts(Xc, Zc, ang, 0.014, 1.12)
        sm = Q(s, so)
        cast(s, sm, -8, 4, PAYNE, 0.32, blur=5)
        base(s, sm, CELADON, 0.7, wob=0.5, soft=0.6, edge=0.6)
        s.wash(sm, SLATE, 0.22, wob=0.5, soft=1)
        sd = Q(s, sole_pts(Xc, Zc, ang, 0.0, 1.12)) * (1 - sm)
        s.wash(sd, SLATE, 0.6, wob=0.3, soft=0.6)
        # the band over the toes: nearest to us (toe at u = +0.135, turned toward the camera)
        strap = s.poly([w2(0.02, -0.06, 0.02), w2(0.02, 0.06, 0.02), w2(0.075, 0.058, 0.02), w2(0.07, 0.0, 0.045), w2(0.075, -0.058, 0.02)])
        base(s, strap, NAVY, 0.75, wob=0.5, soft=0.6, edge=0.6)
        s.wash(strap, INDIGO, 0.3, wob=0.5, soft=1)
        _lift(s, s.ribbon([w2(0.035, -0.04, 0.035), w2(0.045, 0.04, 0.04)], 3, (0.3, 0.3)), 0.3, soft=1)
        s.ink(so, 1.5, INK, 1.1, smooth=False, closed=True)
        contour(s, strap, 1.3, INK, 1.1)
    slipper(-0.21, 2.36, math.pi - 0.15)
    slipper(-0.04, 2.37, math.pi + 0.12)
    # the umbrella's soft shadow on the wall, thrown left by the window (it suits the furled one too)
    ux_, uy_ = P(UMB_TIP[0], 0, UMB_TIP[1])
    ux1_, uy1_ = P(*UMB_TOP)
    shm = s.ribbon([(ux_ - 20, uy_ - 30), ((ux_ + ux1_) / 2 - 60, (uy_ + uy1_) / 2), (ux1_ - 70, uy1_ + 20)], 46, (0.2, 0.3))
    s.wash(cv2.GaussianBlur(shm, (0, 0), 14) * back, PAYNE, 0.22, wob=0, soft=0, edge=0, gran=0.4)
    # ---- the puddle where the umbrella drips
    tx, ty = P(UMB_TIP[0], 0, UMB_TIP[1])
    pm = s.blob(tx + 6, ty + 6, 74, 15, 0.25, 9)
    pm = np.maximum(pm, s.blob(tx - 30, ty + 10, 40, 9, 0.3, 8))
    s.wash(cv2.GaussianBlur(pm, (0, 0), 12) * 1.4 * (1 - pm), PAYNE, 0.18, wob=2, bleed=6, edge=0.1)
    _lift(s, pm, 0.35, soft=1.5, wob=1.5)
    s.wash(pm, SLATE, 0.32, wob=1.5, soft=1, edge=1.0, ew=5, flow=0.4)
    s.wash(pm * s.vgrad(ty - 10, ty + 20, 0, 1), NAVY, 0.2, wob=1, soft=2)                          # the umbrella's colour in the water
    _lift(s, s.ribbon([(tx - 40, ty + 2), (tx + 10, ty - 1), (tx + 50, ty + 3)], 3, (0.4, 0.4)), 0.5, clip=pm, soft=1, wob=0.5)
    cs_, _ = cv2.findContours((pm > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cs_:
        c_ = max(cs_, key=len)[:, 0, :].astype(np.float64)[::6]
        s.ink(c_[: int(len(c_) * 0.45)], 1.3, INKBLUE, 0.8, tremor=0.8, skip=0.15)
    # wet footprints from the door toward the step
    for j, (X_, Z_) in enumerate(((-0.05, 2.62), (0.12, 2.50), (-0.06, 2.40))):
        fp = s.ellipse(*P(X_, 0, Z_), 26, 9, 0)
        s.wash(fp, SLATE, 0.18, wob=2, soft=2, edge=0.8, ew=4, dry=0.3)
    # ---- near the step on the left wall: a paper calendar, this month's page, a small square of days
    cz0, cz1, ch0, ch1 = 1.62, 1.86, 1.08, 1.58
    cal = QP(s, [(WL, ch1, cz0), (WL, ch1, cz1), (WL, ch0, cz1), (WL, ch0, cz0)])
    cast(s, cal, 16, 10, PAYNE, 0.3, blur=10, clip=lw)
    base(s, cal, CREAM, 0.2, wob=0.6, soft=0.6, edge=0.6)
    pic = QP(s, [(WL, ch1 - 0.02, cz0 + 0.02), (WL, ch1 - 0.02, cz1 - 0.02), (WL, ch1 - 0.19, cz1 - 0.02), (WL, ch1 - 0.19, cz0 + 0.02)])
    s.wash(pic, CELADON, 0.6, wob=1, bleed=4, edge=0.5)
    s.wash(pic * s.vgrad(*sorted([P(WL, ch1 - 0.12, cz0)[1], P(WL, ch1 - 0.19, cz0)[1]]), 0, 1), MOSS, 0.4, wob=1, bleed=4)
    for i in range(8):
        zz = cz0 + 0.02 + (cz1 - cz0 - 0.04) * i / 7
        seg(s, P(WL, ch1 - 0.23, zz), P(WL, ch0 + 0.02, zz), 0.9, SEPIA, 0.5, bow=0, skip=0.2)
    for j in range(6):
        hh = ch1 - 0.23 - (ch1 - 0.25 - ch0) * j / 5
        seg(s, P(WL, hh, cz0 + 0.02), P(WL, hh, cz1 - 0.02), 0.9, SEPIA, 0.5, bow=0, skip=0.2)
    cell = QP(s, [(WL, ch1 - 0.23 - 0.054 * 2, cz0 + 0.02 + 0.031 * 3), (WL, ch1 - 0.23 - 0.054 * 2, cz0 + 0.02 + 0.031 * 4), (WL, ch1 - 0.23 - 0.054 * 3, cz0 + 0.02 + 0.031 * 4), (WL, ch1 - 0.23 - 0.054 * 3, cz0 + 0.02 + 0.031 * 3)])
    s.wash(cell, ROSE, 0.45, wob=0.4, soft=0.6, edge=0.7)          # today, circled in red
    path(s, [P(WL, ch1, cz0), P(WL, ch1, cz1), P(WL, ch0, cz1), P(WL, ch0, cz0)], 1.6, SEPIA, 1.1, closed=True, over=3, r=r)
    s.ink([P(WL, ch1 + 0.05, (cz0 + cz1) / 2), P(WL, ch1, cz0 + 0.03)], 1.0, SEPIA, 0.8)
    s.ink([P(WL, ch1 + 0.05, (cz0 + cz1) / 2), P(WL, ch1, cz1 - 0.03)], 1.0, SEPIA, 0.8)
    # ---- near the step on the right wall: a small mirror giving back the grey light
    mz0, mz1, mh0, mh1 = 1.70, 2.05, 1.12, 1.72
    mfr = QP(s, [(WR, mh1, mz0), (WR, mh1, mz1), (WR, mh0, mz1), (WR, mh0, mz0)])
    mgl = QP(s, [(WR, mh1 - 0.03, mz0 + 0.03), (WR, mh1 - 0.03, mz1 - 0.03), (WR, mh0 + 0.03, mz1 - 0.03), (WR, mh0 + 0.03, mz0 + 0.03)])
    cast(s, mfr, -14, 10, PAYNE, 0.3, blur=10, clip=rw)
    wood(s, mfr, SIENNA, SIENNA, 0.4, ang=math.pi / 2, n=6, seed=8, lines=False)
    s.lift(mgl, 1.0, soft=0.6, wob=0.4)
    s.wash(mgl, MIST, 0.45, wob=0.8, soft=1, edge=0.5, body=1)
    s.wash(mgl, SLATE, 0.3, wob=1, bleed=10, grad=s.hgrad(P(WR, 0, mz1)[0], P(WR, 0, mz0)[0], 0, 1))
    for q in (0.3, 0.42):
        a_, b_ = P(WR, mh1 - 0.05, mz0 + (mz1 - mz0) * q), P(WR, mh0 + 0.05, mz0 + (mz1 - mz0) * (q + 0.25))
        _lift(s, s.ribbon([a_, b_], 9, (0.2, 0.2)), 0.45, clip=mgl, soft=2)
    path(s, [P(WR, mh1, mz0), P(WR, mh1, mz1), P(WR, mh0, mz1), P(WR, mh0, mz0)], 1.8, INK, 1.2, closed=True, over=4, r=r)
    path(s, [P(WR, mh1 - 0.03, mz0 + 0.03), P(WR, mh1 - 0.03, mz1 - 0.03), P(WR, mh0 + 0.03, mz1 - 0.03), P(WR, mh0 + 0.03, mz0 + 0.03)], 1.2, SEPIA, 0.9, closed=True, over=2, r=r)
    # ---- the architecture's pen
    seg(s, (bx0, -10), (bx0, by1), 2.4, INK, 1.4, r=r)
    seg(s, (bx1, -10), P(WR, cH, ZB), 2.4, INK, 1.4, r=r)
    seg(s, (bx0, by1), (P(cX, 0, ZB)[0], by1), 2.0, INK, 1.2, r=r, over=6)
    seg(s, (bx0, by1), P(WL, 0, ZS + 0.18), 2.2, INK, 1.3, r=r, over=4)
    seg(s, P(WR, 0, cZ), P(WR, 0, ZS + 0.18), 2.2, INK, 1.3, r=r, over=4)
    seg(s, P(WL, 0.10, ZB), P(dX0 - jamb, 0.10, ZB), 1.4, SEPIA, 0.8, r=r)
    seg(s, P(WL, 0.10, ZB), P(WL, 0.10, ZS + 0.2), 1.4, SEPIA, 0.8, r=r)
    # the whole morning: a cool grey glaze, heavier away from the window
    s.glaze(RAIN, 0.10, grad=s.hgrad(SW, 0, 0.3, 1), flow=0.4, fs=500)
    ux, uy = P(UMB_TIP[0], 0, UMB_TIP[1])
    ux1, uy1 = P(*UMB_TOP)
    deg = math.degrees(math.atan2(ux1 - ux, -(uy1 - uy)))
    meta = {"umb": [round(ux, 1), round(uy, 1), round(deg, 1)], "umb_scale": round(math.hypot(ux1 - ux, uy1 - uy) / UMB_LEN, 3),
            "puddle": [round(tx + 6), round(ty + 8), 76, 16], "window": [round(min(gx0_, gx1_)), round(min(gy0_, gy1_)), round(max(gx0_, gx1_)), round(max(gy0_, gy1_))],
            "door": [round(P(dX0, 0, ZB)[0]), round(P(0, dH, ZB)[1]), round(P(dX1, 0, ZB)[0]), round(P(0, 0, ZB)[1])],
            "peephole": [round(px_), round(py_)], "keys": [round(kx), round(ky)]}
    s.save("en_bg", opaque=True, q=90, meta=meta)


# ============================================================================================== the umbrella, upright
UMB_LEN = 1000                 # tip to the top of the handle's crook


def _umbrella(name, fixed):
    s = Sheet(560, 1100, 311 if not fixed else 312)
    r = rs(311)
    ax, ay = 250, 1070          # the tip
    hem = ay - 700              # where the rib tips are
    # the canopy's outline: narrow at the crown (just above the ferrule), loose and widest below the hem when unkempt
    ys = np.linspace(ay - 58, hem, 24)
    u = (ay - 58 - ys) / (ay - 58 - hem)
    if fixed:
        wl = (14 + 46 * np.sin(np.clip(u, 0, 1) * math.pi * 0.62) ** 0.9) * (1 - 0.3 * sstep(0.84, 1.0, u))
        wr = wl * 0.95
        jl = jr = np.zeros_like(u)
    else:
        wl = 16 + 80 * np.sin(np.clip(u, 0, 1) * math.pi * 0.6) ** 0.8 + 18 * sstep(0.6, 1.0, u)
        wr = 14 + 66 * np.sin(np.clip(u, 0, 1) * math.pi * 0.6) ** 0.8 + 10 * sstep(0.7, 1.0, u)
        jl = 10 * np.sin(u * 17 + 1) * sstep(0.2, 0.5, u)
        jr = 8 * np.sin(u * 13 + 2) * sstep(0.2, 0.5, u)
    L = [(ax - wl[i] - jl[i], ys[i]) for i in range(len(ys))]
    R_ = [(ax + wr[i] + jr[i], ys[i]) for i in range(len(ys))]
    # the hem: rib tips sticking out, fabric scalloping between them
    tipsx = np.linspace(-wl[-1] * 0.9, wr[-1] * 0.9, 7) + ax + (r.random(7) - 0.5) * (4 if fixed else 18)
    tipsy = hem - 12 - (r.random(7) * (6 if fixed else 30)) - 8 * np.cos(np.linspace(-1.2, 1.2, 7))
    hem_pts = []
    for i, tx in enumerate(tipsx):
        hem_pts.append((tx, tipsy[i] + 4))
        if i < len(tipsx) - 1:
            hem_pts.append(((tx + tipsx[i + 1]) / 2 + (r.random() - 0.5) * 6, (tipsy[i] + tipsy[i + 1]) / 2 + (6 if fixed else 10 + 18 * r.random())))
    outline = L + hem_pts + R_[::-1]
    can = s.poly(outline, smooth=True)
    # shaft above the hem, the handle's crook
    shaft_top = ay - 870
    s.stroke([(ax, hem + 10), (ax + 1, shaft_top)], 13, PAYNE, 1.2, taper=(0, 0), wob=0.4, edge=0.5, body=1)
    _lift(s, s.ribbon([(ax + 3, hem - 10), (ax + 3, shaft_top + 10)], 3, (0.1, 0.1)), 0.5, soft=0.8, wob=0.3)
    s.ink([(ax - 6.5, hem), (ax - 6, shaft_top)], 1.4, INK, 1.1)
    s.ink([(ax + 6.5, hem), (ax + 7, shaft_top)], 1.4, INK, 1.1)
    hc = (ax + 52, shaft_top - 6)
    crook = [(ax + 1, shaft_top + 20)] + [tuple(p) for p in ell(hc[0], hc[1], 52, 56, math.pi, 2 * math.pi + 0.55, 24)]
    hm = s.ribbon(crook, 32, (0.0, 0.0))
    base(s, hm, SIENNA, 0.9, wob=0.6, soft=0.6, edge=0.6, ew=3, flow=0.5, fs=40, gran=0.5)
    s.wash(hm, BURNT, 0.55, wob=0.6, soft=1, edge=0.3, grad=s.rgrad(hc[0], hc[1], 30, 80, 1, 0.2))
    _lift(s, s.ribbon([tuple(p) for p in ell(hc[0], hc[1], 62, 66, math.pi * 1.15, math.pi * 1.75, 12)], 6, (0.3, 0.3)), 0.55, soft=1.2, wob=0.4)
    for k in range(3):
        s.ink([tuple(p) for p in ell(hc[0], hc[1], 46 + k * 6, 50 + k * 6, math.pi * (1.05 + 0.1 * k), math.pi * (1.5 + 0.2 * k), 12)], 1.0, UMBER, 0.6, skip=0.3)
    contour(s, hm, 2.0, SEPIA, 1.4)
    col = s.rect(ax - 10, shaft_top + 8, ax + 11, shaft_top + 30)
    base(s, col, MIST, 0.6, wob=0.4, soft=0.6, edge=0.6)
    path(s, [(ax - 10, shaft_top + 8), (ax + 11, shaft_top + 8), (ax + 11, shaft_top + 30), (ax - 10, shaft_top + 30)], 1.3, INK, 1.1, closed=True, over=2)
    # the ferrule
    fer = s.poly([(ax - 7, ay - 60), (ax + 7, ay - 60), (ax + 3, ay - 4), (ax - 3, ay - 4)], smooth=False)
    base(s, fer, MIST, 0.6, wob=0.4, soft=0.6, edge=0.6)
    s.wash(fer, SLATE, 0.5, wob=0.4, soft=1, grad=s.hgrad(ax - 7, ax + 7, 0, 1))
    contour(s, fer, 1.4, INK, 1.2)
    s.wash(s.ellipse(ax, ay - 3, 5, 4), SEPIA, 1.0, wob=0.3, soft=0.5, body=1)
    # ---- the cloth: navy, the folds twisting a little round the shaft, each fold its own value, wet sheen on the ridges
    base(s, can, NAVY, 1.05, wob=0.8, soft=0.8, edge=0.6, ew=5, flow=0.5, fs=80, gran=0.55)
    s.wash(can, INDIGO, 0.45, wob=0.8, soft=1, edge=0, grad=s.hgrad(ax - 100, ax + 90, 1, 0.2))
    nf = 6
    folds = []
    for i in range(nf + 1):
        q = i / nf
        pts = []
        for j, y in enumerate(ys):
            uu = u[j]
            span = (wl[j] + wr[j])
            tw = (0.10 if not fixed else 0.22) * uu
            x = ax - wl[j] + span * min(1, max(0, q + tw * (1 - q) * (1.0 if not fixed else 0.6)))
            pts.append((x + (jl[j] if q < 0.5 else jr[j]) * 0.5, y))
        folds.append(pts)
    for i in range(nf):
        if i % 2 == 0:
            band = s.poly(folds[i] + folds[i + 1][::-1], smooth=True)
            s.wash(band * can, INKBLUE, 0.35, wob=0.6, soft=1.2, edge=0.4, ew=3)
        else:
            band = s.poly(folds[i] + folds[i + 1][::-1], smooth=True)
            _lift(s, band * can * s.hgrad(ax - 60, ax + 90, 0.4, 1), 0.18, soft=2, wob=0.6)
    for i in range(1, nf):
        s.ink(folds[i][1:-1], 1.4, INKBLUE, 1.0, skip=0.25, tremor=0.6, clip=cv2.GaussianBlur(can, (0, 0), 1.2))
        hi = [(x + 4, y) for (x, y) in folds[i][3:-3]]
        _lift(s, s.ribbon(hi, 3.5, (0.3, 0.3)), 0.45 if i > nf / 2 else 0.25, clip=can, soft=1.2, wob=0.6, dry=0.15)
    # rib tips at the hem: little metal caps
    for i, tx in enumerate(tipsx):
        ty = tipsy[i]
        s.wash(s.ellipse(tx, ty, 4, 6), SLATE, 1.0, wob=0.3, soft=0.5, body=1)
        _lift(s, s.ellipse(tx + 1, ty - 2, 1.5, 2), 0.7, soft=0.4, wob=0)
    # drops beading on the cloth, a drip hanging at the crown
    # the wet sheen: broken streaks of window light down the ridges on the right, a cooler glint on the left
    for i in range(nf // 2, nf):
        pts = [(x + 6, y) for (x, y) in folds[i][2:-2]]
        n_ = len(pts)
        for (a_, b_) in ((0.05, 0.35), (0.45, 0.62), (0.7, 0.95)):
            seg_pts = pts[int(a_ * n_): max(int(a_ * n_) + 2, int(b_ * n_))]
            if len(seg_pts) >= 2:
                _lift(s, s.ribbon(seg_pts, 7 + 4 * (i - nf // 2), (0.4, 0.4)), 0.42, clip=can, soft=2.2, wob=0.8, dry=0.2)
    _lift(s, s.ribbon([(ax - wl[j] * 0.7, ys[j]) for j in range(3, 20, 4)], 5, (0.3, 0.3)), 0.2, clip=can, soft=2, wob=0.6)
    drops = []
    for _ in range(16 if not fixed else 8):
        y = ay - 80 - r.random() * 560
        jy = int(np.clip((ay - 58 - y) / (ay - 58 - hem) * 23, 0, 23))
        x = ax - wl[jy] * 0.7 + r.random() * (wl[jy] + wr[jy]) * 0.7
        drops.append((x, y, 2.4 + 2.6 * r.random()))
    dm = s.zeros()
    for x, y, rad in drops:
        dm = np.maximum(dm, s.ellipse(x, y, rad, rad * 1.15))
    s.wash(dm * can, INKBLUE, 0.55, wob=0.2, soft=0.5, edge=1.0, ew=1.0)
    hl = s.zeros()
    for x, y, rad in drops:
        hl = np.maximum(hl, s.ellipse(x - rad * 0.3, y - rad * 0.4, rad * 0.38, rad * 0.32))
    _lift(s, hl * can, 0.9, soft=0.4, wob=0.1)
    drip_y = ay - 52
    drop = s.ellipse(ax - 13, drip_y + 6, 5, 7)
    s.wash(drop, RAIN, 0.5, wob=0.2, soft=0.5, edge=0.9, ew=1.2, body=0.7)
    _lift(s, s.ellipse(ax - 14, drip_y + 4, 1.5, 2), 0.9, soft=0.3, wob=0)
    contour(s, can, 2.2, INKBLUE, 1.5)
    # ---- the strap: fastened round the middle, or hanging open with its snap
    sy = ay - 430
    jy = int(np.clip((ay - 58 - sy) / (ay - 58 - hem) * 23, 0, 23))
    if fixed:
        band = s.poly([(ax - wl[jy] - 3, sy - 12), (ax + wr[jy] + 3, sy - 16), (ax + wr[jy] + 3, sy + 10), (ax - wl[jy] - 3, sy + 14)])
        base(s, band, NAVY, 1.2, wob=0.5, soft=0.6, edge=0.6)
        s.wash(band, INKBLUE, 0.4, wob=0.5, soft=1, grad=s.hgrad(ax - 60, ax + 60, 1, 0))
        contour(s, band, 1.6, INK, 1.3)
        sn = s.ellipse(ax + wr[jy] * 0.4, sy - 1, 8, 8)
        base(s, sn, MIST, 0.6, wob=0.3, soft=0.5, edge=0.6)
        oval(s, ax + wr[jy] * 0.4, sy - 1, 8, 8, 1.2, INK, 1.1)
        strap_end = (ax + wr[jy] * 0.4, sy)
    else:
        a0 = (ax + wr[jy] - 4, sy)
        sp = [a0, (a0[0] + 26, sy + 50), (a0[0] + 30, sy + 110), (a0[0] + 14, sy + 158)]
        st = s.ribbon(sp, 22, (0.0, 0.0))
        base(s, st, NAVY, 1.15, wob=0.5, soft=0.6, edge=0.6)
        s.wash(st, INKBLUE, 0.35, wob=0.5, soft=1, grad=s.hgrad(a0[0], a0[0] + 40, 0, 1))
        s.ink(spline(sp)[::4] + np.array([0, 0]), 1.0, INKBLUE, 0.6, skip=0.3)
        contour(s, st, 1.6, INK, 1.3)
        e_ = sp[-1]
        sn = s.ellipse(e_[0], e_[1] - 4, 8, 8)
        base(s, sn, MIST, 0.6, wob=0.3, soft=0.5, edge=0.6)
        s.wash(s.ellipse(e_[0], e_[1] - 4, 3, 3), SLATE, 0.8, wob=0.2, soft=0.4)
        oval(s, e_[0], e_[1] - 4, 8, 8, 1.2, INK, 1.1)
        s.ink([(e_[0] - 2, e_[1] + 6), (e_[0] - 3, e_[1] + 14)], 2.2, RAIN, 0.8)                 # a drop on its way
        strap_end = e_
    # ---- the bent rib: out of the folds below the hem, kinked, a tent of cloth dragged with it
    if not fixed:
        k0 = (ax + wr[17] - 10, ys[17])
        kink = (k0[0] + 70, k0[1] - 58)
        rib_end = (kink[0] + 54, kink[1] + 22)
        tent = s.poly([(k0[0] - 4, k0[1] + 26), (k0[0] + 22, k0[1] - 14), (k0[0] + 40, k0[1] - 30), (k0[0] + 4, k0[1] - 46)])
        base(s, tent, NAVY, 1.0, wob=0.5, soft=0.6, edge=0.6)
        s.wash(tent, INKBLUE, 0.4, wob=0.5, soft=1)
        _lift(s, s.ribbon([(k0[0] + 4, k0[1] - 20), (kink[0] - 6, kink[1] + 4)], 3, (0.3, 0.3)), 0.4, soft=1)
        contour(s, tent, 1.6, INKBLUE, 1.3)
        s.ink([k0, kink, rib_end], 3.2, INK, 1.6, smooth=False, taper=(0, 0.1), tremor=0.3)
        _lift(s, s.ribbon([(k0[0] + 2, k0[1] - 3), (kink[0] + 1, kink[1] - 3), (rib_end[0] - 4, rib_end[1] - 4)], 1.4, (0.1, 0.1), smooth=False), 0.6, soft=0.4, wob=0)
        s.wash(s.ellipse(rib_end[0], rib_end[1], 4, 5), SLATE, 1.0, wob=0.3, soft=0.5, body=1)
    else:
        rib_end = (tipsx[-1], hem - 26)
    # the lowest fold of loose cloth, where drops fall from
    drip = (ax - wl[3] - jl[3], ys[3]) if not fixed else (ax - 13, drip_y + 12)
    s.save(name, crop=True, anchor=(ax, ay), meta={"tip_drip": [round(drip[0] - ax, 1), round(drip[1] - ay, 1)], "rib": [round(rib_end[0] - ax, 1), round(rib_end[1] - ay, 1)],
                                                   "strap": [round(strap_end[0] - ax, 1), round(strap_end[1] - ay, 1)], "len": UMB_LEN, "hem": round(hem - ay)})


# ============================================================================================== fix: the umbrella opened on the floor
# a pinhole camera looking down at the floor at 56 degrees; world X right, Y away, Z up (metres)
FXW, FXH = 2560, 1440
_al = math.radians(56)
_D = 1.75
CAM = np.array([0.0, -_D * math.cos(_al), _D * math.sin(_al)])
_fwd = -CAM / np.linalg.norm(CAM)
_rt = np.array([1.0, 0, 0])
_up = np.cross(_rt, _fwd)
FF, FCX, FCY = 2050.0, 1300.0, 1010.0


def C(p):
    """World point(s) -> screen."""
    p = np.asarray(p, float) - CAM
    d = p @ _fwd
    return np.stack([FCX + FF * (p @ _rt) / d, FCY - FF * (p @ _up) / d], -1)


# the canopy: a spherical cap (rim radius 0.52 m) lying tilted on the floor, inside up, the shaft out to the lower right
RS, TM = 0.57, math.radians(66)
_ax = np.array([0.40, -0.80, 0.44])
_ax /= np.linalg.norm(_ax)
_e1 = np.cross(_ax, [0, 0, 1.0])
_e1 /= np.linalg.norm(_e1)
_e2 = np.cross(_ax, _e1)
HUB = np.array([-0.06, 0.16, 0.0])
NR = 8
PHI = np.arange(NR) * 2 * math.pi / NR + 0.30
KB = 2                         # the bent rib
KT = 0.56                      # where along it (fraction of the cap angle) it is kinked


def _lift_floor():
    """Raise the hub so the rim's lowest point just touches the floor."""
    c = HUB + _ax * RS
    lows = []
    for ph in np.linspace(0, 2 * math.pi, 72):
        q = c + RS * (-_ax * math.cos(TM) + (_e1 * math.cos(ph) + _e2 * math.sin(ph)) * math.sin(TM))
        lows.append(q[2])
    return HUB + np.array([0, 0, -min(lows) + 0.004])


HUBW = _lift_floor()
SC = HUBW + _ax * RS           # sphere centre


def cap(th, ph, sag=1.0):
    """Point on the canopy at cap angle th (0 = hub) and azimuth ph; sag < 1 pulls the cloth in toward the axis."""
    radial = _e1 * math.cos(ph) + _e2 * math.sin(ph)
    return SC + RS * (-_ax * math.cos(th) + radial * math.sin(th) * sag)


def _rot(v, k, a):
    k = k / (np.linalg.norm(k) + 1e-12)
    return v * math.cos(a) + np.cross(k, v) * math.sin(a) + k * (k @ v) * (1 - math.cos(a))


def rib3(i, n=30):
    """The rib's points (world). The bent one runs true to the kink, then the outer part swings in and round."""
    pts = np.array([cap(t, PHI[i]) for t in np.linspace(0, TM, n)])
    if i == KB:
        k = int(KT * (n - 1))
        kink = pts[k].copy()
        out = []
        for p in pts[k:]:
            v = p - kink
            dv = (pts[-1] - kink)
            v = _rot(v, np.cross(dv, _ax), 0.22)        # in toward the axis
            v = _rot(v, _ax, 0.95)                      # and round toward the next rib
            out.append(kink + v * 0.95)
        pts = np.vstack([pts[:k], np.array(out)])
    return pts


def fx_geom():
    ribs = [rib3(i) for i in range(NR)]
    tips = [rb[-1] for rb in ribs]
    kink = ribs[KB][int(KT * 29)]
    runner = HUBW + _ax * 0.16
    shaft_end = HUBW + _ax * 1.6
    return ribs, tips, kink, runner, shaft_end


def fx_bg():
    s = Sheet(FXW, FXH, 321)
    r = rs(321)
    # floor tiles (8 cm), from the inverse of the camera on the floor plane
    ray = np.stack([(s.X - FCX) / FF, -(s.Y - FCY) / FF, np.ones_like(s.X)], -1)
    dirw = ray[..., 0:1] * _rt + ray[..., 1:2] * _up + ray[..., 2:3] * _fwd
    tt = -CAM[2] / np.minimum(dirw[..., 2], -1e-4)
    Xw = CAM[0] + tt * dirw[..., 0]
    Yw = CAM[1] + tt * dirw[..., 1]
    cell = 0.08
    u, v = Xw / cell, Yw / cell
    iu, iv = np.floor(u), np.floor(v)
    hh = np.sin(iu * 12.9898 + iv * 78.233) * 43758.5453
    rnd = (hh - np.floor(hh)).astype(np.float32)
    pxm = FF / np.maximum(tt, 0.1)                                         # px per metre (roughly) there
    gd = (np.minimum(np.minimum(u - iu, 1 - (u - iu)), np.minimum(v - iv, 1 - (v - iv))) * cell * pxm * 0.8).astype(np.float32)
    s.glaze(TILE, 0.55, flow=0.4, fs=300, gran=0.5)
    s.wash(s.full(), CREAM, 0.5, wob=0, soft=0, edge=0, grad=np.clip((rnd - 0.6) * 3, 0, 1), gran=0.3)
    s.wash(s.full(), SLATE, 0.45, wob=0, soft=0, edge=0, grad=np.clip((0.25 - rnd) * 4, 0, 1), gran=0.4)
    s.wash(s.full(), CELADON, 0.35, wob=0, soft=0, edge=0, grad=np.clip(1 - np.abs(rnd - 0.45) * 8, 0, 1), gran=0.3)
    s.wash(np.clip(2.2 - gd, 0, 1), UMBER, 0.45, wob=1.0, soft=0.8, edge=0, gran=0.3, flow=0.6, fs=60)
    # wet: dark soaked patches, the window's light lying on the water in long soft streaks
    for _ in range(9):
        m = s.blob(r.random() * FXW, r.random() * FXH, 120 + 200 * r.random(), 60 + 110 * r.random(), 0.35, 10, rot=r.random() * 3)
        s.wash(m, PAYNE, 0.22, wob=4, bleed=6, edge=0.9, ew=8, flow=0.4)
        _lift(s, m * s.noise(40, 2), 0.18, soft=2, wob=2)
    for _ in range(6):
        x, y = r.random() * FXW, r.random() * FXH
        _lift(s, s.ribbon([(x - 160, y + 40), (x, y), (x + 160, y - 40)], 18 + 18 * r.random(), (0.5, 0.5)), 0.3, soft=6, wob=3)
    # the doormat's edge, top left: coir, its bound edge
    mat = s.poly([(-20, -20), (980, -20), (900, 230), (-20, 330)], smooth=False)
    base(s, mat, SIENNA, 0.8, wob=1.5, soft=1, edge=0.6, flow=0.5, fs=40, gran=0.8)
    s.wash(mat, UMBER, 0.45, wob=0, soft=0, edge=0, grad=s.noise(4, 2), gran=0.7)
    s.wash(mat, OCHRE, 0.35, wob=1, soft=1, dry=0.5, gran=0.7)
    for k in range(60):
        x = -20 + k * 17
        s.ink([(x, -20), (x - 12, 330)], 1.0, UMBER, 0.6, skip=0.4, clip=mat, tremor=1.2)
    edge_ = s.poly([(-20, 300), (900, 205), (906, 228), (-20, 336)], smooth=False)
    base(s, edge_, NAVY, 0.7, wob=1, soft=0.8, edge=0.5)
    s.ink([(-20, 300), (900, 205)], 1.8, INK, 1.2)
    s.ink([(-20, 336), (906, 230), (900, 205)], 2.0, INK, 1.3)
    s.wash(s.poly([(-20, 336), (906, 230), (940, 270), (-20, 390)], smooth=False), PAYNE, 0.35, wob=2, bleed=10)
    # a towel, bottom right, spread out to take the drips: cream terry with a mint stripe, folded over itself
    tw = s.poly([(1640, 1140), (2600, 1010), (2600, 1460), (1580, 1460)], smooth=False)
    cast(s, tw, -14, -8, PAYNE, 0.3, blur=12)
    base(s, tw, CREAM, 0.42, wob=2, soft=1, edge=0.5, flow=0.5, fs=60, gran=0.7)
    s.wash(tw, SLATE, 0.22, wob=2, bleed=14, grad=s.vgrad(1000, 1460, 0, 1))
    s.wash(tw, OCHRE, 0.12, wob=0, soft=0, dry=0.55, gran=0.8)
    stripe = s.poly([(1610, 1290), (2600, 1170), (2600, 1215), (1605, 1340)], smooth=False) * tw
    s.wash(stripe, CELADON, 0.8, wob=1.5, soft=1, edge=0.5)
    s.wash(stripe, LEAF, 0.18, wob=1.5, soft=1)
    fold = s.poly([(1640, 1140), (2600, 1010), (2600, 1060), (1660, 1186)], smooth=False)
    s.wash(fold, SLATE, 0.3, wob=2, bleed=6)
    for k in range(40):                                                   # terry loops: a dry scumble of little marks
        x, y = 1640 + r.random() * 900, 1080 + r.random() * 360
        s.ink([(x, y), (x + 3, y + 4), (x + 1, y + 7)], 1.0, UMBER, 0.35, clip=tw)
    s.ink([(1640, 1140), (2600, 1010)], 2.2, SEPIA, 1.3)
    s.ink([(1640, 1140), (1600, 1300), (1580, 1460)], 2.0, SEPIA, 1.2)
    # drops scattered on the tiles
    dm = s.zeros()
    for _ in range(140):
        x, y, rad = r.random() * FXW, r.random() * FXH, 3 + 9 * r.random() ** 2
        dm = np.maximum(dm, s.ellipse(x, y, rad, rad * 0.8))
    s.wash(dm * (1 - mat), SLATE, 0.35, wob=0.6, soft=0.5, edge=1.0, ew=1.5)
    _lift(s, cv2.erode(dm, np.ones((3, 3), np.uint8)) * (1 - mat), 0.2, soft=0.5, wob=0.3)
    s.save("fx_bg", opaque=True, q=90)


def fx_umb():
    ribs, tips, kink, runner, shaft_end = fx_geom()
    s = Sheet(FXW + 600, FXH + 500, 322)
    OX, OY = 300, 250                                            # this sheet's offset from the fix sheet
    r = rs(322)

    def S(p):
        q = C(p)
        return q + np.array([OX, OY])
    hub = S(HUBW)
    R2 = [S(rb) for rb in ribs]
    T2 = [S(t) for t in tips]
    # the hem between rib tips: the rim sagging in toward the axis, deeper beside the bent rib
    hems = []
    for i in range(NR):
        j = (i + 1) % NR
        a0, a1 = PHI[i], PHI[i] + 2 * math.pi / NR
        ss = np.linspace(0, 1, 24)
        sag = 1 - (0.07 + (0.12 if KB in (i, j) else 0.0)) * np.sin(np.pi * ss)
        pts = np.array([cap(TM - 0.03 * math.sin(math.pi * q), a0 + (a1 - a0) * q, sg) for q, sg in zip(ss, sag)])
        # pin the ends to the actual tips (the bent tip has moved)
        pts += np.outer(1 - ss, tips[i] - pts[0]) + np.outer(ss, tips[j] - pts[-1])
        hems.append(S(pts))
    # cast shadow of the whole canopy on the floor (light from up and left)
    allp = np.vstack(hems)
    W = s.poly(allp, smooth=False)
    sh = cv2.GaussianBlur(cv2.warpAffine(W, np.float32([[1, 0, 40], [0, 1, 56]]), (s.w, s.h)), (0, 0), 30) * (1 - W)
    s.wash(sh, PAYNE, 0.45, wob=0, soft=0, edge=0, gran=0.4)
    # panels: the cloth seen from inside; each faces the light differently
    L = np.array([-0.35, 0.25, 0.90])
    L /= np.linalg.norm(L)
    order = sorted(range(NR), key=lambda i: -np.linalg.norm(((tips[i] + tips[(i + 1) % NR]) / 2) - CAM))
    pms = {}
    for i in order:
        j = (i + 1) % NR
        poly = np.vstack([hub[None], R2[i], hems[i], R2[j][::-1]])
        pm = s.poly(poly, smooth=False)
        pms[i] = pm
        mid3 = cap(TM * 0.6, PHI[i] + math.pi / NR)
        nrm = (SC - mid3) / np.linalg.norm(SC - mid3)
        lit = float(np.clip(nrm @ L, 0, 1))
        _opaque(s, pm, NAVY, 0.85 + 0.1 * (i % 2), wob=0.6, soft=0.7, edge=0.7, ew=6, flow=0.3, fs=420, gran=0.45)
        s.wash(pm, INDIGO, 0.45, wob=0.6, soft=1, edge=0, grad=s.rgrad(hub[0], hub[1], 80, 1000, 1, 0.1))      # deeper in the bowl, darker
        s.wash(pm, INKBLUE, 0.10 + 0.55 * (1 - lit), wob=0.6, soft=1, edge=0.35, flow=0.35, fs=300)
        _lift(s, pm * s.rgrad(*S(mid3), 60, 620, 1, 0), 0.16 + 0.10 * (i % 2) + 0.55 * max(0.0, lit - 0.2), soft=2, wob=0.6, flow=0.4, fs=200)
        rim_band = pm * (1 - s.rgrad(hub[0], hub[1], 450, 950, 1, 0))
        _lift(s, rim_band, 0.10 + 0.08 * (i % 2), soft=2, wob=0.6)                                        # grey light through the cloth near the edge
        # the seams along the ribs are darker, the cloth between them bellies
        for rr_ in (R2[i], R2[j]):
            s.wash(s.ribbon(rr_, 70, (0.3, 0.2)) * pm, INKBLUE, 0.35, wob=0.6, soft=14, edge=0)
            s.wash(s.ribbon(rr_, 22, (0.1, 0.2)) * pm, INKBLUE, 0.35, wob=0.6, soft=4, edge=0)
    # the bent rib drags its two panels: creases running from the kink
    k2 = S(kink)
    for i in (KB - 1, KB):
        jn = (i + 1) % NR
        other = R2[i] if i != KB else R2[jn]
        for q in (0.55, 0.75, 0.9):
            p_ = other[int(q * (len(other) - 1))]
            mid_ = (k2 + p_) / 2 + (r.random(2) - 0.5) * 30
            s.stroke([k2, mid_, p_], 30, INKBLUE, 0.45, wob=1, bleed=8, taper=(0.5, 0.6), clip=pms[i % NR], edge=0.2)
            _lift(s, s.ribbon([k2 + (6, -10), mid_ + (8, -12), p_ + (6, -10)], 8, (0.5, 0.6)), 0.25, clip=pms[i % NR], soft=3, wob=1)
            s.ink([k2, mid_, p_], 1.3, INKBLUE, 0.8, skip=0.4, tremor=1.0)
    # water pooled down at the hub, drops all over the cloth
    pool = s.blob(hub[0] + 30, hub[1] + 20, 150, 90, 0.3, 10, rot=-0.3)
    s.wash(pool * W, RAIN, 0.25, wob=3, soft=2, edge=1.0, ew=8)
    _lift(s, s.ribbon([(hub[0] - 60, hub[1] - 10), (hub[0] + 20, hub[1] - 30), (hub[0] + 100, hub[1] - 20)], 12, (0.4, 0.4)) * pool, 0.5, soft=3, wob=1)
    drops = []
    for _ in range(140):
        i = int(r.integers(NR))
        th = TM * (0.15 + 0.8 * r.random())
        ph = PHI[i] + 2 * math.pi / NR * (0.15 + 0.7 * r.random())
        drops.append((S(cap(th, ph)), 3 + 6 * r.random() ** 2))
    dm = s.zeros()
    hl = s.zeros()
    for (x, y), rad in drops:
        dm = np.maximum(dm, s.ellipse(x, y, rad, rad * 0.85))
        hl = np.maximum(hl, s.ellipse(x - rad * 0.3, y - rad * 0.35, rad * 0.35, rad * 0.3))
    s.wash(dm * W, INKBLUE, 0.5, wob=0.3, soft=0.5, edge=1.0, ew=1.2)
    _lift(s, hl * W, 0.85, soft=0.4, wob=0.1)
    # stretchers: from the runner out to each rib a third of the way along
    run2 = S(runner)
    for i in range(NR):
        p_ = S(cap(TM * 0.36, PHI[i]))
        s.stroke([run2, p_], 7, PAYNE, 1.1, taper=(0, 0), wob=0.3, edge=0.4, body=1)
        s.ink([run2, p_], 1.4, INK, 1.2, tremor=0.3)
        _lift(s, s.ribbon([run2 + (1, -3), p_ + (1, -3)], 2, (0.2, 0.2)), 0.45, soft=0.6, wob=0)
    # ribs: thin dark metal, a hair of light, the tip caps
    for i in range(NR):
        rr_ = R2[i]
        s.stroke(rr_[::3], 7, INK, 1.2, taper=(0, 0), wob=0.3, edge=0.3, body=1)
        _lift(s, s.ribbon(rr_[2::3] + np.array([1.5, -2.5]), 2.2, (0.1, 0.1)), 0.5, soft=0.6, wob=0)
        tm = s.ellipse(T2[i][0], T2[i][1], 9, 9)
        base(s, tm, SLATE, 0.9, wob=0.3, soft=0.5, edge=0.6)
        _lift(s, s.ellipse(T2[i][0] - 2, T2[i][1] - 3, 3, 3), 0.7, soft=0.5, wob=0)
        oval(s, T2[i][0], T2[i][1], 9, 9, 1.4, INK, 1.2)
        # a drop hanging at the tip
        s.wash(s.ellipse(T2[i][0], T2[i][1] + 14, 6, 8), RAIN, 0.45, wob=0.2, soft=0.5, edge=0.9, ew=1.2, body=0.6)
        _lift(s, s.ellipse(T2[i][0] - 2, T2[i][1] + 10, 1.8, 2.2), 0.9, soft=0.4, wob=0)
    # the hem's pen line, edge by edge
    for i in range(NR):
        s.ink(hems[i][::2], 2.4, INKBLUE, 1.5, tremor=0.8, taper=(0.06, 0.12))
    # the shaft, the runner, the hub's cap
    sh2 = S(shaft_end)
    s.stroke([hub, sh2], 15, PAYNE, 1.2, taper=(0, 0), wob=0.4, edge=0.5, body=1)
    _lift(s, s.ribbon([hub + (2, -5), sh2 + (2, -5)], 3.5, (0.1, 0.1)), 0.55, soft=0.8, wob=0.3)
    s.ink([hub + (0, -7), sh2 + (0, -7)], 1.6, INK, 1.2)
    s.ink([hub + (0, 7), sh2 + (0, 7)], 1.6, INK, 1.2)
    rn = s.ribbon([S(runner - _ax * 0.03), S(runner + _ax * 0.03)], 30, (0, 0))
    base(s, rn, SLATE, 1.0, wob=0.4, soft=0.6, edge=0.6)
    _lift(s, rn * s.hgrad(run2[0] - 15, run2[0] + 15, 1, 0), 0.3, soft=1)
    contour(s, rn, 1.6, INK, 1.2)
    hc = s.ellipse(hub[0], hub[1], 26, 22)
    base(s, hc, INK, 0.9, wob=0.4, soft=0.6, edge=0.5)
    _lift(s, s.ellipse(hub[0] - 6, hub[1] - 6, 7, 5), 0.5, soft=1, wob=0.2)
    oval(s, hub[0], hub[1], 26, 22, 1.6, INK, 1.3)
    k_ = k2
    # where the shaft leaves the fix frame (2560x1440)
    dd = sh2 - hub
    tx_ = min([q for q in ((OX + FXW - hub[0]) / dd[0] if dd[0] > 0 else 9e9, (OY + FXH - hub[1]) / dd[1] if dd[1] > 0 else 9e9,
                           (OX - hub[0]) / dd[0] if dd[0] < 0 else 9e9, (OY - hub[1]) / dd[1] if dd[1] < 0 else 9e9)])
    ex_ = hub + dd * min(1.0, tx_)
    meta = {"ribtips": [[round(float(t[0] - hub[0]), 1), round(float(t[1] - hub[1]), 1)] for t in T2],
            "bent": [round(float(k_[0] - hub[0]), 1), round(float(k_[1] - hub[1]), 1)],
            "shaft": [round(float(ex_[0] - hub[0]), 1), round(float(ex_[1] - hub[1]), 1)],
            "at": [round(float(hub[0] - OX), 1), round(float(hub[1] - OY), 1)]}
    s.save("fx_umb", crop=True, anchor=(float(hub[0]), float(hub[1])), meta=meta)


def fx_hands():
    """Both my hands round the bent rib, from below the frame, overhand: the backs of the hands and the knuckles toward
    us, the fingers curling over the rib and away under it. A piece of the rib is drawn with them so they hold it."""
    ribs, tips, kink, runner, shaft_end = fx_geom()
    k2 = C(kink)
    s = Sheet(1800, 1600, 323)
    gx, gy = 900, 560
    rb = C(ribs[KB]) - k2 + np.array([gx, gy])
    rb = resample(spline(rb, step=3.0), 3.0)
    sarc = arclen(rb)
    sk = sarc[np.argmin(np.hypot(rb[:, 0] - gx, rb[:, 1] - gy))]

    def at(d):
        """Point on the rib d px along it from the kink (toward the tip if positive), and its direction."""
        q = np.clip(sk + d, 4, sarc[-1] - 10)
        x, y = np.interp(q, sarc, rb[:, 0]), np.interp(q, sarc, rb[:, 1])
        x2, y2 = np.interp(q + 6, sarc, rb[:, 0]), np.interp(q + 6, sarc, rb[:, 1])
        return np.array([x, y]), math.atan2(y2 - y, x2 - x)
    # the rib under the hands
    seg_ = rb[(sarc > sk - 330) & (sarc < sk + 330)]
    s.stroke(seg_[::3], 7, INK, 1.2, taper=(0, 0), wob=0.3, edge=0.3, body=1)
    _lift(s, s.ribbon(seg_[2::3] + np.array([1.5, -2.5]), 2.2, (0.1, 0.1)), 0.5, soft=0.6, wob=0)
    drips = []

    def one(d, seed_, thumb_side):
        rr = rs(seed_)
        c0, ang = at(d)
        cu, su = math.cos(ang), math.sin(ang)
        vx, vy = su, -cu                       # across the rib, away from us (up the screen)
        if vy > 0:
            vx, vy = -vx, -vy

        def L(u, v):
            return (c0[0] + u * cu + v * vx, c0[1] + u * su + v * vy)

        def LP(ps):
            return [L(*p) for p in ps]
        w0 = np.array(L(0, -150))
        far = w0 + np.array([0.0, 900.0])      # the forearm runs off the bottom of the frame whatever the rib does

        def Fp(u, v):
            a = np.array(L(u, -150))
            b = far + np.array([u * 1.25, 0.0])
            return tuple(a * (1 - v) + b * v)
        sleeve = s.poly([Fp(-92, 0.06), Fp(92, 0.06), Fp(126, 1), Fp(-126, 1)], smooth=False)
        cuff = s.poly(LP([(-88, -120), (88, -120), (98, -215), (-100, -215)]))
        # the back of the hand ends in a row of knuckles; the thumb lies along the rib, pointing at the other hand
        back = s.poly(LP([(-72, -132), (-84, -60), (-74, -10), (-56, -2), (-36, -8), (-18, 0), (2, -6), (20, 0), (38, -6), (54, -1), (72, -10), (82, -60), (68, -132)]))
        fx_ = (-52, -18, 17, 50)
        fw_ = (32, 34, 33, 29)
        fing = [s.ribbon(LP([(x, -14), (x + 1, 12), (x + 2, 30)]), w, profile=lambda t: 1 - 0.18 * t) for x, w in zip(fx_, fw_)]
        tsg = thumb_side
        thumb = s.ribbon(LP([(tsg * 60, -96), (tsg * 92, -50), (tsg * 112, -8), (tsg * 128, 10)]), 36, profile=lambda t: 1 - 0.3 * t)
        hm = np.maximum(back, thumb)
        for f in fing:
            hm = np.maximum(hm, f)
        arm = np.maximum(np.maximum(hm, cuff), sleeve)
        shd = cv2.GaussianBlur(cv2.warpAffine(arm, np.float32([[1, 0, 34], [0, 1, 44]]), (s.w, s.h)), (0, 0), 18) * (1 - arm)
        s.wash(shd, INKBLUE, 0.35, wob=0, soft=0, edge=0, gran=0.4)
        base(s, sleeve, CREAM, 0.6, wob=1.5, edge=0.45, flow=0.5, fs=140, gran=0.5)
        s.wash(sleeve, OCHRE, 0.30, wob=2, bleed=16, grad=s.hgrad(c0[0] - 160, c0[0] + 160, 0.2, 1))
        s.wash(sleeve, SLATE, 0.16, wob=2, bleed=14, edge=0.1)
        for j in range(5):
            a = Fp(-120, 0.18 + j * 0.16)
            b = Fp(120, 0.22 + j * 0.16)
            m_ = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 14)
            s.stroke([a, m_, b], 22, OCHRE, 0.22, wob=2, bleed=6, clip=sleeve, taper=(0.4, 0.4), edge=0.2)
            s.ink([a, m_, b], 1.2, SEPIA, 0.45, skip=0.4, clip=sleeve)
        contour(s, sleeve, 2.2, SEPIA, 1.2)
        base(s, cuff, CREAM, 0.6, wob=1.2, edge=0.55, flow=0.4)
        s.wash(cuff, OCHRE, 0.30, wob=1.5, bleed=8)
        for i in range(13):
            q = -86 + i * 14
            s.ink(LP([(q, -124), (q * 1.1, -211)]), 1.3, SEPIA, 0.6, clip=cuff, tremor=0.5)
        contour(s, cuff, 2.0, SEPIA, 1.3)
        # the fingers lie over the rib and curl away under it: lit on top at the knuckles, darker where they turn
        for f, x in zip(fing, fx_):
            _opaque(s, f, SKIN, 0.62, wob=1.0, soft=0.8, edge=0.45, flow=0.3)
            s.wash(f, ROSE, 0.16, wob=1, bleed=5)
            s.wash(f, BURNT, 0.55, wob=1, bleed=4, grad=s.rgrad(*L(x, 34), 4, 30, 1, 0))      # curling away under the rib
            s.wash(f, UMBER, 0.25, wob=1, soft=2, grad=s.rgrad(*L(x, 36), 2, 16, 1, 0))
            contour(s, f, 1.5, SEPIA, 1.1)
        _opaque(s, back, SKIN, 0.62, wob=1.2, soft=0.8, edge=0.45, flow=0.3)
        s.wash(back, BURNT, 0.25, wob=2, bleed=10, grad=s.rgrad(*L(0, -150), 30, 200, 1, 0))
        _opaque(s, thumb, SKIN, 0.62, wob=1.0, soft=0.8, edge=0.45, flow=0.3)
        s.wash(thumb, BURNT, 0.36, wob=1.5, bleed=8)
        contour(s, np.maximum(back, thumb), 2.0, SEPIA, 1.3)
        for x in fx_:
            s.ink(LP([(x - 12, -10), (x, -15), (x + 12, -10)]), 1.2, BURNT, 0.7)
        for (a, b) in (((-36, -120), (-46, -24)), ((0, -120), (-4, -24)), ((34, -120), (38, -24))):
            s.ink(LP([a, b]), 1.0, BURNT, 0.35, skip=0.4)
        # wet: light on the knuckles and along the thumb, beads, water gathering at the lowest edge to drip
        for x in fx_:
            p_ = L(x - 4, -16)
            _lift(s, s.ellipse(p_[0], p_[1], 4, 3, math.degrees(ang)), 0.75, soft=0.6, wob=0.2)
        _lift(s, s.ribbon(LP([(tsg * 66, -90), (tsg * 96, -48), (tsg * 114, -12)]), 5, (0.3, 0.3)), 0.5, soft=1, wob=0.3)
        for _ in range(8):
            p_ = L(-70 + 140 * rr.random(), -130 + 120 * rr.random())
            rad = 2.5 + 2.5 * rr.random()
            s.wash(s.ellipse(p_[0], p_[1], rad, rad * 0.9), BURNT, 0.35, wob=0.2, soft=0.5, edge=1.0, ew=1)
            _lift(s, s.ellipse(p_[0] - rad * 0.3, p_[1] - rad * 0.35, rad * 0.35, rad * 0.3), 0.9, soft=0.3, wob=0)
        for p_ in (L(tsg * 128, 22), L(-tsg * 82, -40)):
            s.wash(s.ellipse(p_[0], p_[1] + 6, 5, 8), RAIN, 0.5, wob=0.2, soft=0.5, edge=1.0, ew=1.2, body=0.6)
            _lift(s, s.ellipse(p_[0] - 1.5, p_[1] + 3, 1.6, 2.2), 0.9, soft=0.3, wob=0)
            drips.append([round(p_[0] - gx, 1), round(p_[1] + 12 - gy, 1)])
    one(-185, 1, 1)
    one(170, 2, -1)
    s.save("fx_hands", crop=True, anchor=(gx, gy), meta={"drips": drips, "at": [round(float(k2[0]), 1), round(float(k2[1]), 1)]})


def umb():
    _umbrella("en_umb", False)


def umb_fixed():
    _umbrella("en_umb_fixed", True)


ALL = {"en_bg": bg, "en_umb": umb, "en_umb_fixed": umb_fixed, "fx_bg": fx_bg, "fx_umb": fx_umb, "fx_hands": fx_hands}
