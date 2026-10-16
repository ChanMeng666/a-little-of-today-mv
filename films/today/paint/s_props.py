"""Small things the page moves about: blank sticky notes that float up into the sky, and the strip of washi tape the
lyric is lettered on.

note_small, note_small_b: a blank yellow sticky note, ~170 px square, opaque paper, sheet 260x260, anchor its centre
  (130, 130). The adhesive strip is along the top edge. note_small: light from the upper left, the bottom-right corner
  curling up; note_small_b: light from the right, the note bent over a little at the bottom, the bottom-left corner lifting.
ly_tape, ly_tape_mint: a strip of masking (washi) tape torn off at both ends, sheet 1600x220, saved cropped (about
  1560x150); translucent warm cream / pale mint paper with a faint printed grid of small squares, a few wrinkles and
  pooled edges. The page stretches the whole crop to the line's width. meta band [y0, y1]: the tape's long edges in
  the cropped image, so the lettering can be centred between them.
"""
import json
import math
import os

import cv2
import numpy as np

from kit import *

NOTE_BACK = (0.96, 0.91, 0.62)       # the back of the sticky paper: paler, a touch greener


def _note(name, seed, light, curl, bend):
    s = Sheet(260, 260, seed)
    r = rs(seed)
    c = 130
    h = 85
    # corners, a little out of square (it is cut by a hand-held guillotine and then handled)
    TL, TR = (c - h + r.random() * 2, c - h + r.random() * 2), (c + h + r.random() * 2, c - h + 1 + r.random() * 2)
    BR, BL = (c + h - 1, c + h + r.random() * 2), (c - h + 1, c + h - 1)
    if bend:        # the bottom bends away from us: the lower edge narrows a little and rides up
        BR, BL = (BR[0] - 7, BR[1] - 10), (BL[0] + 7, BL[1] - 10)
    # the curl: a corner lifted off along a curved line
    if curl == "br":
        A, B, K = (TR[0] - 1, TR[1] + 120), (BL[0] + 116, BL[1] - 1), BR
        flap_tip = (BR[0] - 34, BR[1] - 30)
    else:
        A, B, K = (BL[0] + 1, BL[1] - 58), (BL[0] + 60, BL[1]), BL
        flap_tip = (BL[0] + 28, BL[1] - 26)
    mid = ((A[0] + B[0]) / 2 + (K[0] - (A[0] + B[0]) / 2) * 0.18, (A[1] + B[1]) / 2 + (K[1] - (A[1] + B[1]) / 2) * 0.18)
    if curl == "br":
        body = [TL, TR, A, mid, B, BL]
    else:
        body = [TL, TR, BR, B, mid, A]
    bm = s.poly(body, smooth=False)
    # the lifted flap: from the curl line, bowing out toward where the corner was, its tip folded back over the note
    flap = s.poly([A, (mid[0] + (K[0] - mid[0]) * 0.55, mid[1] + (K[1] - mid[1]) * 0.55), B, ((B[0] + flap_tip[0]) / 2, (B[1] + flap_tip[1]) / 2),
                   flap_tip, ((A[0] + flap_tip[0]) / 2, (A[1] + flap_tip[1]) / 2)], smooth=True)
    lx = s.hgrad(c - h, c + h, 1, 0) if light == "left" else s.hgrad(c - h, c + h, 0, 1)
    base(s, bm, NOTE, 0.78, wob=0.8, edge=0.5, flow=0.25, fs=90, gran=0.3)
    s.wash(bm, OCHRE, 0.12, wob=1, bleed=12, grad=1 - lx, clip=cv2.GaussianBlur(bm, (0, 0), 2))
    s.lift(bm * lx, 0.16, soft=12, wob=2)
    # the adhesive strip along the top: a band a shade deeper with a soft lower edge
    strip = bm * (1 - s.vgrad(TL[1] + 26, TL[1] + 34))
    s.wash(strip, OCHRE, 0.12, wob=1.5, bleed=3, edge=0.15)
    if bend:
        s.wash(bm, SIENNA, 0.22, wob=1.5, bleed=14, grad=s.vgrad(c + 10, c + h, 0, 1), clip=cv2.GaussianBlur(bm, (0, 0), 2))
    # the curl throws a soft shadow onto the note under it
    sh = cv2.warpAffine(flap, np.float32([[1, 0, -6 if curl == "br" else 6], [0, 1, -4]]), (s.w, s.h))
    s.wash(cv2.GaussianBlur(sh, (0, 0), 5) * bm * (1 - flap), SIENNA, 0.45, wob=0, soft=0, edge=0, gran=0.3)
    base(s, flap, NOTE_BACK, 0.9, wob=0.6, edge=0.5, gran=0.4)
    s.wash(flap, OCHRE, 0.28, wob=0.6, soft=2, grad=s.rgrad(*flap_tip, 4, 50, 0, 1), clip=flap)
    s.lift(flap * s.rgrad(*flap_tip, 2, 30, 1, 0), 0.3, soft=2, wob=0.5)
    # pen: each edge its own stroke, a little past the corners; the curl's edge drawn twice
    pen_c = UMBER
    if curl == "br":
        edges = [(TL, TR), (TR, A), (B, BL), (BL, TL)]
    else:
        edges = [(TL, TR), (TR, BR), (BR, B), (A, TL)]
    for a, b in edges:
        seg(s, a, b, 1.5, pen_c, 0.9, over=4 + 4 * r.random(), r=r)
    s.ink([A, (mid[0] + (K[0] - mid[0]) * 0.55, mid[1] + (K[1] - mid[1]) * 0.55), B], 1.5, pen_c, 1.0)
    s.ink([A, ((A[0] + flap_tip[0]) / 2, (A[1] + flap_tip[1]) / 2), flap_tip, ((B[0] + flap_tip[0]) / 2, (B[1] + flap_tip[1]) / 2), B], 1.4, pen_c, 0.95)
    s.save(name, crop=True, anchor=(c, c), meta={"side": 2 * h})


def note_small():
    _note("note_small", 501, "left", "br", False)


def note_small_b():
    _note("note_small_b", 502, "right", "bl", True)


def _tape(name, col, deep, grid_c, seed):
    s = Sheet(1600, 220, seed)
    r = rs(seed)
    y0, y1 = 48, 172
    xl, xr = 30, 1570

    def torn(x, sgn):
        """A torn end: a ragged zig-zag across the tape with a few fibres pulled out."""
        ys = np.linspace(y0 - 2, y1 + 2, 26)
        xs = x + sgn * (np.abs(noise1(26, 4, seed + int(x))) * 22 + r.random(26) * 9 * np.where(np.arange(26) % 2, 1, 0.3))
        return list(zip(xs, ys))
    left = torn(xl, -1)
    right = torn(xr, 1)
    top = [(x, y0 + 1.2 * math.sin(x * 0.011 + 1)) for x in np.linspace(xl + 10, xr - 10, 40)]
    bot = [(x, y1 + 1.2 * math.sin(x * 0.013 + 4)) for x in np.linspace(xr - 10, xl + 10, 40)]
    P = top + right + bot + left[::-1]
    m = s.poly(P, smooth=False)
    m = cv2.GaussianBlur(m, (0, 0), 0.7)
    # the paper: a pale wash that pools along the cut edges, with slow uneven take-up; mostly opaque, a little see-through
    s.wash(m, col, 0.62, wob=0.5, soft=0.6, edge=0.7, ew=3, gran=0.45, flow=0.35, fs=120, body=0.84)
    s.wash(m, deep, 0.05, wob=1, bleed=10, flow=0.8, fs=160, clip=m)
    # fibres along the torn ends
    for x, sgn in ((xl, -1), (xr, 1)):
        for i in range(26):
            yy = y0 + r.random() * (y1 - y0)
            L = 6 + r.random() * 16
            ang = (r.random() - 0.5) * 1.2
            xx = x + sgn * (r.random() * 10)
            s.ink([(xx, yy), (xx + sgn * L * math.cos(ang) * 0.5, yy + L * math.sin(ang) * 0.5), (xx + sgn * L * math.cos(ang), yy + L * math.sin(ang))], 0.8, col, 1.6, taper=(0.1, 0.8), tremor=0.6)
        band = m * s.hgrad(x - sgn * 30, x, 0, 1) if sgn < 0 else m * s.hgrad(x - 30, x, 0, 1)
        s.wash(band, deep, 0.06, wob=1, soft=3)
    # the printed grid of small squares, slightly off register with the tape
    g = 15.0
    ox, oy = 7 + r.random() * 4, y0 + 6 + r.random() * 3
    mm = cv2.GaussianBlur(m, (0, 0), 1.5)
    gm = np.zeros((s.h, s.w), np.float32)
    xs = np.arange(xl - 20 + ox, xr + 20, g)
    ys = np.arange(oy, y1 - 2, g)
    for x in xs:
        cv2.line(gm, (int(x * 16), int((y0 - 4) * 16)), (int((x + 0.6) * 16), int((y1 + 4) * 16)), 1.0, 1, cv2.LINE_AA, 4)
    for y in ys:
        cv2.line(gm, (int((xl - 20) * 16), int(y * 16)), (int((xr + 20) * 16), int((y + 0.8) * 16)), 1.0, 1, cv2.LINE_AA, 4)
    gm = cv2.GaussianBlur(gm, (0, 0), 0.45) * mm * (0.55 + 0.45 * s.noise(40, 2, seed + 3))
    s.wash(gm, grid_c, 0.13, wob=0, soft=0, edge=0, gran=0.6, flow=0)
    # wrinkles: where it was pressed down unevenly, a pale ridge with a soft shadow beside it
    for _ in range(7):
        x = xl + 60 + r.random() * (xr - xl - 120)
        if r.random() < 0.5:
            x = (xl + 40 + r.random() * 160) if r.random() < 0.5 else (xr - 40 - r.random() * 160)
        dx = (r.random() - 0.5) * 60
        ya = y0 + r.random() * 20
        yb = y1 - r.random() * 30
        pts = [(x, ya), (x + dx * 0.5 + (r.random() - 0.5) * 8, (ya + yb) / 2), (x + dx, yb)]
        s.stroke([(p[0] + 3, p[1]) for p in pts], 6, deep, 0.10, wob=1, bleed=3, clip=m, taper=(0.4, 0.4), edge=0.2)
        s.lift(s.ribbon(pts, 2.5, taper=(0.4, 0.4)) * m, 0.35, soft=1, wob=0.5)
    out = s.save(name, crop=True)
    out["band"] = [y0 - out["oy"], y1 - out["oy"]]
    json.dump(out, open(os.path.join(META, name + ".json"), "w", encoding="utf-8"), ensure_ascii=False)


def tape():
    _tape("ly_tape", (0.98, 0.93, 0.72), OCHRE, ROSE, 601)


def tape_mint():
    _tape("ly_tape_mint", (0.84, 0.93, 0.86), LEAF, TEAL, 602)


ALL = {"note_small": note_small, "note_small_b": note_small_b, "ly_tape": tape, "ly_tape_mint": tape_mint}
