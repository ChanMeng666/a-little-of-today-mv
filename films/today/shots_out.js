// Outside: the two blocks across the lane and their windows, one window close, the old lane and the two of us in it.
import { W, H, PAL, rgba, L, S, inSet, camT, camK, toScreen, glow, lamp, rain, ripples, bokeh, pen, chalk, squarePts, notesUp, along, MAN, IM, clamp, lerp, sstep, hash, noise1, makeCanvas, TAU, E } from './stage.js';
import { TM, B, T0, CT, beatNear, beatIdx, kickN, since, ramp } from './core.js';
import { entryFrame } from './shots_sill.js';

const NIGHT = [64, 72, 120];
const darken = (c, col, a) => { if (a <= 0.003) return; c.save(); c.setTransform(1, 0, 0, 1, 0, 0); c.globalCompositeOperation = 'multiply'; c.globalAlpha = a; c.fillStyle = rgba(col); c.fillRect(0, 0, W, H); c.restore(); };

// ================================================================ the block
const BK = 'block';
const winRect = (side, i) => (MAN['bk_' + side].windows || [])[i];
const MINE = () => winRect('left', MAN.bk_left.mine ?? 0), YOURS = () => winRect('right', MAN.bk_right.yours ?? 0);
// lights: fn(side, i) → 0..1 level for each window; o: dawn 0..1, night 0..1, rain, sil (draw the silhouettes), inWin
// Back to front: sky, the lane beyond and the dark glass (darkened for night), the lit rooms clipped to the windows that
// are on, the facades (darkened; their glass is clear), the wires.
const quad = (side, i) => (MAN['bk_' + side].quads || [])[i];
function nightBuf(c, cam, names, night, dawn) {
  const [cv, cx] = buf(); cx.setTransform(1, 0, 0, 1, 0, 0); cx.globalAlpha = 1; cx.globalCompositeOperation = 'source-over'; cx.clearRect(0, 0, W, H);
  for (const [n, p] of names) {
    L(cx, BK, cam, n, p);
    // two stray dark spikes stand above the far building in bk_far; the facades hide them in most framings, not all
    if (n === 'bk_far') inSet(cx, BK, cam, p, () => cx.clearRect(1380, 0, 330, 985));
  }
  if (night > 0.003) { cx.save(); cx.globalCompositeOperation = 'source-atop'; cx.globalAlpha = 0.6 * night; cx.fillStyle = rgba(NIGHT); cx.fillRect(0, 0, W, H); cx.restore(); }
  if (dawn > 0.003) { cx.save(); cx.globalCompositeOperation = 'source-atop'; cx.globalAlpha = 0.16 * dawn; cx.fillStyle = rgba([255, 190, 160]); cx.fillRect(0, 0, W, H); cx.restore(); }
  c.save(); c.setTransform(1, 0, 0, 1, 0, 0); c.drawImage(cv, 0, 0); c.restore();
}
function block(c, cam, t, lights, o = {}) {
  const dawn = o.dawn ?? 0, night = o.night ?? 1;
  L(c, BK, cam, 'bk_sky', 0.35);
  if (dawn > 0.003) L(c, BK, cam, 'bk_sky_dawn', 0.35, { a: dawn });
  nightBuf(c, cam, [['bk_far', 0.9], ['bk_glass', 1]], night, dawn);
  inSet(c, BK, cam, 1, () => {
    for (const side of ['left', 'right']) (MAN['bk_' + side].windows || []).forEach((r, i) => {
      const lv = lights(side, i); if (lv <= 0.003) return;
      const q = quad(side, i);
      c.save(); c.beginPath();
      if (q) { c.moveTo(...q[0]); for (const pt of q.slice(1)) c.lineTo(...pt); c.closePath(); } else c.rect(r[0], r[1], r[2] - r[0], r[3] - r[1]);
      c.clip(); c.globalAlpha = lv;
      if (o.inWin && isMine(side, i)) {           // cover-fit: crop the picture to the window's shape
        const iw = o.inWin.width, ih = o.inWin.height, ar = (r[2] - r[0]) / (r[3] - r[1]), cw = Math.min(iw, ih * ar), ch = cw / ar;
        c.drawImage(o.inWin, (iw - cw) / 2, (ih - ch) / 2, cw, ch, r[0], r[1], r[2] - r[0], r[3] - r[1]);
      }
      else { const m = RAWM('bk_glass_lit'); if (m) c.drawImage(IM.bk_glass_lit, m.ox, m.oy); }
      c.restore();
    });
    if (o.sil) {
      const ym = YOURS(), mm = MINE();
      if (ym) S(c, BK, cam, 'bk_sil_you', (ym[0] + ym[2]) / 2, ym[3], 1, { a: lights('right', MAN.bk_right.yours ?? 0) });
      if (mm) S(c, BK, cam, 'bk_sil_me', (mm[0] + mm[2]) / 2, mm[3], 1, { a: lights('left', MAN.bk_left.mine ?? 0) });
    }
  });
  nightBuf(c, cam, [['bk_left', 1], ['bk_right', 1]], night, dawn);
  // light spilling out of the lit windows onto the wall and the grilles
  inSet(c, BK, cam, 1, () => {
    for (const side of ['left', 'right']) (MAN['bk_' + side].windows || []).forEach((r, i) => {
      const lv = lights(side, i); if (lv <= 0.02) return;
      glow(c, (r[0] + r[2]) / 2, (r[1] + r[3]) / 2, (r[2] - r[0]) * 1.5, PAL.amber, 0.30 * lv, { blend: 'screen' });
    });
  });
  if (o.rain ?? 1) rain(c, t, { n: 160, a: 0.2 * (o.rain ?? 1), len: 70, seed: 12, v: 1600, slant: -0.08 });
  L(c, BK, cam, 'bk_wires', 1.3);
  if (MAN.bk_wires.lamp) inSet(c, BK, cam, 1.3, () => lamp(c, MAN.bk_wires.lamp[0], MAN.bk_wires.lamp[1], 420, t, 0.7 * night, { seed: 4 }));
}
const RAWM = (n) => (IM[n] ? MAN[n] : null);
let BUF;
const buf = () => (BUF ||= makeCanvas(W, H));
const isMine = (side, i) => side === 'left' && i === (MAN.bk_left.mine ?? 0), isYours = (side, i) => side === 'right' && i === (MAN.bk_right.yours ?? 0);
// a window clicks on: a small flicker, then steady
const on = (t, at) => (t < at ? 0 : clamp((t - at) / 0.08) * (1 - 0.35 * Math.exp(-(t - at) * 6) * (Math.sin((t - at) * 60) > 0 ? 1 : 0)));

// Ch1 「如果你刚好也没睡 / 就把今天留一小块给我」
export function windows(c, t, lt, sh) {
  // push in toward the two windows as they light, so they read
  const mm = MINE() || [864, 763, 968, 903], yy = YOURS() || [2097, 786, 2193, 915], mx = (mm[0] + yy[2]) / 2, my = (mm[1] + yy[3]) / 2;
  const cam = camK(t, [[sh.t0, 1536, 900, 2700], [CT(9, 0), lerp(1536, mx, 0.6), lerp(900, my, 0.6), 2100], [sh.t1, mx, my, 1750]], { drift: 3 });
  const tm = CT(8, 7), ty = CT(9, 1);
  block(c, cam, t, (side, i) => (isMine(side, i) ? on(t, tm) : isYours(side, i) ? on(t, ty) : 0));
  // the pen goes once round your window on 「一小块」
  const y = YOURS();
  if (y) inSet(c, BK, cam, 1, () => { const u = ramp(t, CT(9, 4) - 0.05, CT(9, 7) - CT(9, 4) + 0.3, 'io'), r = Math.max(y[2] - y[0], y[3] - y[1]) * 0.7; pen(c, squarePts((y[0] + y[2]) / 2, (y[1] + y[3]) / 2, r, 0.04, 3), u, { color: 'rgba(250,214,110,0.95)', w: 6, seed: 5, tremor: 2 }); });
  return { grade: { warm: 0.1, bloom: 1.1, sat: 1.0, vig: 0.55 } };
}
// Ch2 「如果你刚好也没睡」: higher, later; the windows go out one by one until only ours are left
export function windows2(c, t, lt, sh) {
  const cam = camK(t, [[sh.t0, 1536, 520, 2300], [sh.t1, 1536, 600, 2150]], { drift: 3 });
  const n0 = sh.t0, n1 = CT(24, 7) + 0.2;
  block(c, cam, t, (side, i) => {
    if (isMine(side, i) || isYours(side, i)) return 1;
    if (hash(i * 7.1 + (side === 'left' ? 2 : 5)) > 0.55) return 0;
    const off = lerp(n0, n1, hash(i * 3.9 + (side === 'left' ? 4 : 8)));
    return 1 - clamp((t - off) / 0.08);
  });
  return { grade: { warm: 0.1, bloom: 1.1, sat: 1.0, vig: 0.6 } };
}
// Ch3 「如果你刚好也没睡」: first light, our two windows still on
export function dawn(c, t, lt, sh) {
  const cam = camK(t, [[sh.t0, 1536, 900, 2500], [sh.t1, 1536, 860, 2300]], { drift: 2 });
  const d = ramp(t, sh.t0, sh.len + 0.5, 'io');
  block(c, cam, t, (side, i) => (isMine(side, i) || isYours(side, i) ? 1 - 0.3 * d : 0), { dawn: 0.7 + 0.3 * d, night: 0.8 - 0.6 * d, rain: 0 });
  inSet(c, BK, cam, 0.35, () => glow(c, 1536, 1500, 1900, [255, 196, 150], 0.30 + 0.35 * d, { blend: 'screen', sy: 0.45 }));
  return { grade: { warm: lerp(0.15, 0.45, d), bloom: 1.1, sat: 1.08, lift: 0.01 } };
}
// V2 「像我们总把小事想得很远」: the entry, small, inside one lit window of the block; the camera keeps going back
export function far(c, t, lt, sh) {
  const m = MINE() || [1000, 600, 1200, 800];
  const cx = (m[0] + m[2]) / 2, cy = (m[1] + m[3]) / 2;
  // the entry has just closed into my window; the camera goes back and back
  const cam = camK(t, [[sh.t0, cx, cy, 1300], [sh.t1 + 0.3, 1536, 860, 3000, 0, 'io']]);
  block(c, cam, t, (side, i) => (isMine(side, i) ? 1 : hash(i * 5.3) > 0.82 ? 0.7 : 0), { night: 0.8, dawn: 0.35, rain: 0.5, inWin: entryFrame(t) });
  return { grade: { warm: 0.1, bloom: 1.0, sat: 1.0, vig: 0.55 } };
}

// ================================================================ one window, close
const WN = 'win';
function win(c, cam, t, o = {}) {
  L(c, WN, cam, 'wn_bg');
  const m = MAN.wn_bg, lv = o.lamp ?? 1;
  if (m.lamp) inSet(c, WN, cam, 1, () => lamp(c, m.lamp[0], m.lamp[1], 500, t, 0.7 * lv, { seed: 6 }));
  const d = m.desk || [1280, 900];
  if (o.who === 'you') S(c, WN, cam, 'wn_you', d[0], d[1], 1, { sy: 1 + 0.008 * Math.sin(t * 1.2) });
  if (o.who === 'up') S(c, WN, cam, 'wn_you_up', d[0], d[1]);
  if (o.note) { const g = m.glass || [800, 300, 1800, 1100]; S(c, WN, cam, 'wn_note', g[0] + (g[2] - g[0]) * 0.32, g[1] + (g[3] - g[1]) * 0.36, 1, { r: -0.05, a: o.note, s: 0.42 }); }
  if (lv < 1) darken(c, [60, 56, 96], 0.85 * (1 - lv));
  L(c, WN, cam, 'wn_grille');
  rain(c, t, { n: 120, a: 0.2, len: 64, seed: 21, v: 1500, slant: -0.07 });
}
// Ch1 「不用讲清楚为什么疲惫」
export function tired(c, t, lt, sh) {
  const cam = camK(t, [[sh.t0, 1280, 740, 2000], [sh.t1, 1290, 760, 1700]], { drift: 3 });
  win(c, cam, t, { who: t < CT(10, 6) ? 'up' : 'you' });
  return { grade: { warm: 0.2, bloom: 1.0, sat: 1.0, vig: 0.6 } };
}
// Ch2: and the lamp goes down
export function tired2(c, t, lt, sh) {
  const cam = camK(t, [[sh.t0, 1290, 760, 1800], [sh.t1, 1280, 740, 2100]], { drift: 3 });
  win(c, cam, t, { who: 'you', lamp: 1 - 0.8 * ramp(t, CT(26, 6), 1.6) });
  return { grade: { warm: 0.2, bloom: 1.0, sat: 1.0, vig: 0.65 } };
}
// Ch2 「就把今天留一小块给我」: a note on my glass, facing yours; the pen goes round it
export function noteWin(c, t, lt, sh) {
  const cam = camK(t, [[sh.t0, 1200, 700, 1900], [sh.t1, 1180, 690, 1650]], { drift: 3 });
  c.save(); c.translate(W, 0); c.scale(-1, 1);          // my window: the same kind of window across the lane, seen the other way round
  win(c, cam, t, { note: ramp(t, sh.t0, 0.3) });
  const g = MAN.wn_bg.glass || [800, 300, 1800, 1100], nx = g[0] + (g[2] - g[0]) * 0.32, ny = g[1] + (g[3] - g[1]) * 0.36;
  inSet(c, WN, cam, 1, () => { const u = ramp(t, CT(25, 4) - 0.05, CT(25, 7) - CT(25, 4) + 0.3, 'io'); pen(c, squarePts(nx, ny, 115, -0.05, 9), u, { color: 'rgba(250,214,110,0.95)', w: 6, seed: 7, tremor: 2 }); });
  c.restore();
  return { grade: { warm: 0.2, bloom: 1.1, sat: 1.0, vig: 0.6 } };
}

// ================================================================ the lane
const LN = 'lane';
const LP = (k) => (MAN.ln_ground.p || {})[k === 'sky' ? 'ln_sky_dusk' : 'ln_' + k] ?? { sky: 0.35, far: 0.7, left: 1, right: 1, ground: 1, fg: 1.4 }[k];
// the lane was painted through a camera at eye height 1.5 m with its horizon at the vanishing point: a person whose feet
// are at sheet y stands (y - horizon) * 1.1 px tall, and the lane's middle at that y is on the path
const HZ = () => (MAN.ln_ground.vanish || [1420, 900])[1];
const hAt = (y) => (y - HZ()) * 1.1;
function xAt(y) { const P = PATH(); for (let i = 1; i < P.length; i++) if ((P[i - 1][1] - y) * (P[i][1] - y) <= 0) { const u = (y - P[i - 1][1]) / (P[i][1] - P[i - 1][1] || 1); return lerp(P[i - 1][0], P[i][0], u); } return P[P.length - 1][0]; }
const PATH = () => MAN.ln_ground.path || [[1536, 1700], [1536, 1100], [1700, 900]];
// o: time of day ('dusk' | 'night' | 'day'), rain, figs: fn() drawn between the ground and the side walls' front
function laneSet(c, cam, t, o = {}) {
  const tod = o.tod || 'dusk', night = tod === 'night' ? 1 : tod === 'dusk' ? 0.45 : 0;
  L(c, LN, cam, tod === 'day' ? 'ln_sky_day' : 'ln_sky_dusk', LP('sky'));
  if (tod === 'night') darken(c, [70, 76, 120], 0.6);
  for (const k of ['far', 'ground', 'left', 'right']) {
    L(c, LN, cam, 'ln_' + k, LP(k));
    if (k === 'ground' && o.path) o.path();
  }
  if (night > 0) {
    darken(c, [92, 96, 150], 0.5 * night + 0.15);
    inSet(c, LN, cam, 1, () => (MAN.ln_ground.windows || MAN.ln_left.windows || []).forEach((r, i) => {
      if (hash(i * 2.7) > 0.7) return;
      const lv = 0.75 + 0.25 * noise1(t * 0.3, i);
      glow(c, (r[0] + r[2]) / 2, (r[1] + r[3]) / 2, (r[2] - r[0]) * 1.3, PAL.amber, 0.55 * lv * night, { blend: 'screen' });
    }));
  }
  if (tod === 'day') inSet(c, LN, cam, 1, () => glow(c, 1536, 400, 1800, PAL.sun, 0.25, { blend: 'screen' }));
  const lp = MAN.ln_ground.lamp || MAN.ln_right.lamp;
  if (lp && tod !== 'day') inSet(c, LN, cam, 1, () => { lamp(c, lp[0], lp[1], 520, t, (o.lampLv ?? 1) * (0.6 + 0.4 * night), { seed: 8 }); glow(c, lp[0] - 80, lp[1] + 700, 520, PAL.amber, 0.25 * (o.lampLv ?? 1), { sy: 0.35 }); });
  if (o.figs) o.figs();
  if (o.rain) { rain(c, t, { n: 200, a: 0.24 * o.rain, len: 70, seed: 31, v: 1600, slant: -0.07 }); inSet(c, LN, cam, 1, () => ripples(c, t, [900, 1250, 2200, 1700], { n: 30, per: 1.1, R: 40, sy: 0.25, a: 0.45 * o.rain, seed: 4 })); }
  const fgDark = tod === 'night' ? 0.62 : tod === 'dusk' ? 0.45 : 0.12;
  const [cv, cx] = buf(); cx.setTransform(1, 0, 0, 1, 0, 0); cx.globalAlpha = 1; cx.globalCompositeOperation = 'source-over'; cx.clearRect(0, 0, W, H);
  L(cx, LN, cam, 'ln_fg', LP('fg'));
  inSet(cx, LN, cam, LP('fg'), () => cx.clearRect(0, 0, 1400, 1728));
  cx.save(); cx.globalCompositeOperation = 'source-atop'; cx.globalAlpha = fgDark; cx.fillStyle = rgba([48, 50, 84]); cx.fillRect(0, 0, W, H); cx.restore();
  c.save(); c.setTransform(1, 0, 0, 1, 0, 0); c.drawImage(cv, 0, 0); c.restore();
}
// the dotted pen line along the lane: drawn to `u` of the path, dashes walking
function dotted(c, cam, t, u, o = {}) {
  inSet(c, LN, cam, 1, () => {
    const pts = PATH(), n = 70;
    c.save(); c.lineCap = 'round';
    for (let i = 0; i < n; i++) {
      const a = i / n; if (a > u) break;
      if (i % 2) continue;
      const [x0, y0] = along(pts, a), [x1, y1] = along(pts, Math.min(u, a + 0.6 / n));
      const near = 1 - a; c.strokeStyle = rgba(o.col || PAL.note, 0.9); c.lineWidth = 3 + 9 * near * near;
      c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1, y1); c.stroke();
    }
    c.restore();
  });
}
// Ch1 「如果你刚好也想起 / 那条走到一半的旧巷口」: down the lane toward the bend; the line stops halfway
function laneShot(c, t, sh, tod, rainA, upto, o = {}) {
  const bend = MAN.ln_ground.bend || [1475, 1004];
  const cam = camK(t, [[sh.t0, 1536, 870, 3050], [sh.t1, 1600, 1060, 2300]], { ease: 'io', drift: 3 });
  laneSet(c, cam, t, { tod, rain: rainA, path: () => dotted(c, cam, t, upto * ramp(t, sh.t0 + 0.3, sh.len * 0.7, 'io')), lampLv: o.lampLv });
}
export function lane(c, t, lt, sh) { laneShot(c, t, sh, 'dusk', 0, 0.5, { lampLv: on(t, CT(12, 7)) }); return { grade: { warm: 0.05, bloom: 1.0, sat: 1.0, vig: 0.5 } }; }
export function laneNight(c, t, lt, sh) { laneShot(c, t, sh, 'night', 1, 0.65); return { grade: { warm: 0.05, bloom: 1.1, sat: 1.0, vig: 0.6 } }; }
// Ch3 「如果你刚好也想起 / 那条走到一半的旧巷口」: daylight after the rain; the camera goes round the corner at last
export function laneDay(c, t, lt, sh) {
  const bend = MAN.ln_ground.bend || [1475, 1004];
  const cam = camK(t, [[sh.t0, 1536, 870, 3050], [CT(41, 6), bend[0] + 20, bend[1] - 60, 1700, 0, 'io'], [sh.t1, bend[0] + 120, bend[1] + 20, 1400]], { drift: 3 });
  laneSet(c, cam, t, { tod: 'day', path: () => dotted(c, cam, t, 0.5 + 0.5 * ramp(t, sh.t0 + 0.4, sh.len - 0.6, 'io'), { col: [240, 196, 70] }) });
  return { grade: { warm: 0.28, bloom: 1.0, sat: 1.12, lift: 0.015 } };
}
// the two of us at feet height y in the lane (anchor = ground between our feet)
function pairAt(c, cam, t, name, y, walking) {
  const h = hAt(y), s = h / (MAN[name + '_w0'].h_me || 880), x = xAt(y) - 20;
  if (!walking) return S(c, LN, cam, name + '_w0', x, y, 1, { s });
  const ph = (t * TM.tempo / 60) % 1, fr = ph < 0.5 ? 0 : 1, bob = Math.abs(Math.sin(ph * TAU)) * 0.014 * h, sway = Math.sin(ph * TAU) * 0.006 * h;
  S(c, LN, cam, `${name}_w${fr}`, x + sway, y - bob, 1, { s, r: sway / h * 0.6 });
}
// walking away from us down the lane; the camera keeps a little behind
function walkShot(c, t, sh, name, tod, rainA, gr) {
  const k = ramp(t, sh.t0, sh.len + 0.6, 'lin'), y = lerp(1500, 1330, k), h = hAt(y);
  const w = Math.max(1900, h / 0.27), cy = y - h * 0.3;
  const cam = camK(t, [[sh.t0, xAt(1500) - 40, cy, w], [sh.t1, xAt(1330) - 30, cy - 10, w * 0.97]], { drift: 3 });
  laneSet(c, cam, t, { tod, rain: rainA, figs: () => pairAt(c, cam, t, name, y, true), path: () => dotted(c, cam, t, tod === 'day' ? 1 : 0.5) });
  return { grade: gr };
}
// Ch1 「下次我们别急着回家」
export function walk(c, t, lt, sh) { return walkShot(c, t, sh, 'fig_pair', 'dusk', 0, { warm: 0.08, bloom: 1.0, sat: 1.0, vig: 0.5 }); }
// Ch2: under your one bent umbrella
export function umbrella(c, t, lt, sh) { return walkShot(c, t, sh, 'fig_umb', 'night', 1, { warm: 0.08, bloom: 1.1, sat: 1.0, vig: 0.6 }); }
// Ch3: daylight, umbrella furled
export function walkDay(c, t, lt, sh) {
  const bend = MAN.ln_ground.bend || [1475, 1003], k = ramp(t, sh.t0, sh.len + 0.6, 'lin'), y = lerp(1240, 1120, k);
  const cam = camK(t, [[sh.t0, bend[0] + 120, bend[1] + 20, 1400], [sh.t1, bend[0] + 160, bend[1] - 10, 1250]], { drift: 3 });
  laneSet(c, cam, t, { tod: 'day', figs: () => pairAt(c, cam, t, 'fig_furl', y, true), path: () => dotted(c, cam, t, 1, { col: [240, 196, 70] }) });
  return { grade: { warm: 0.28, bloom: 1.0, sat: 1.1, lift: 0.015 } };
}

// 「把没说的话，慢慢说」: blank notes rise from us into the sky
function riseShot(c, t, sh, name, tod, rainA, n, gr) {
  const y = 1320, h = hAt(y), x = xAt(y);
  const cam = camK(t, [[sh.t0, x, y - h * 0.5, 1900], [sh.t1, x, y - h * 1.6, 2400]], { ease: 'io', drift: 3 });
  laneSet(c, cam, t, { tod, rain: rainA, figs: () => {
    pairAt(c, cam, t, name, y, false);
    inSet(c, LN, cam, 1, () => notesUp(c, t, n, (i) => [x + (hash(i * 9.1) - 0.5) * h * 0.6, y - h * 0.95, sh.t0 + 0.3 + i * (sh.len * 0.75 / n)], { speed: 110, life: 7, s: 0.32 }));
  }, path: () => dotted(c, cam, t, tod === 'day' ? 1 : 0.5) });
  return { grade: gr };
}
export function rise(c, t, lt, sh) { return riseShot(c, t, sh, 'fig_pair', 'dusk', 0, 9, { warm: 0.1, bloom: 1.1, sat: 1.02, vig: 0.5 }); }
export function rise2(c, t, lt, sh) { return riseShot(c, t, sh, 'fig_umb', 'night', 1, 12, { warm: 0.1, bloom: 1.15, sat: 1.0, vig: 0.6 }); }
// Ch3 「把没说的话，慢慢说」: a whole flock of them, into a bright sky
export function notesSky(c, t, lt, sh) {
  const bend = MAN.ln_ground.bend || [1800, 950];
  const cam = camK(t, [[sh.t0, bend[0] + 100, bend[1] - 200, 1700], [sh.t1, bend[0] + 60, bend[1] - 700, 2400, 0.03]], { ease: 'io', drift: 3 });
  laneSet(c, cam, t, { tod: 'day', path: () => dotted(c, cam, t, 1, { col: [240, 196, 70] }), figs: () => {
    inSet(c, LN, cam, 1, () => notesUp(c, t, 46, (i) => [bend[0] - 700 + 1400 * hash(i * 4.3), bend[1] + 300 * hash(i * 2.2), sh.t0 + (i / 46) * 3.6 - 0.5], { speed: 260, life: 6, s: 0.6, wind: 70 }));
  } });
  return { grade: { warm: 0.3, bloom: 1.2, sat: 1.12, lift: 0.02 } };
}
