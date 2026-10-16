// Around the neighbourhood: the breakfast stall and the soy milk, the river steps at night, the osmanthus shop.
import { W, H, PAL, rgba, L, S, inSet, camT, camK, toScreen, glow, lamp, steam, bokeh, handwrite, pen, MAN, IM, clamp, lerp, sstep, hash, noise1, makeCanvas, TAU, E } from './stage.js';
import { TM, B, T0, CT, beatNear, beatIdx, kickN, since, ramp } from './core.js';

const darken = (c, col, a) => { if (a <= 0.003) return; c.save(); c.setTransform(1, 0, 0, 1, 0, 0); c.globalCompositeOperation = 'multiply'; c.globalAlpha = a; c.fillStyle = rgba(col); c.fillRect(0, 0, W, H); c.restore(); };
// osmanthus blossoms: tiny four-petalled orange stars drifting down and sideways, closed form, current transform
function blossoms(c, t, box, o = {}) {
  const n = o.n ?? 60, a = o.a ?? 0.9, sz = o.size ?? 7, fall = o.fall ?? 60;
  c.save();
  for (let i = 0; i < n; i++) {
    const h1 = hash(i * 1.71 + 5), h2 = hash(i * 3.13 + 2), h3 = hash(i * 5.9 + 8), per = (box[3] - box[1]) / (fall * (0.6 + 0.8 * h3));
    const u = ((t / per + h1) % 1), x = lerp(box[0], box[2], h2) + Math.sin(t * (0.7 + h3) + i) * 40 + (o.wind ?? 25) * u * per * 0.4, y = lerp(box[1], box[3], u);
    const al = a * sstep(0, 0.1, u) * (1 - sstep(0.85, 1, u)), r = sz * (0.6 + 0.8 * h3), rot = t * (1 + 2 * h1) + i;
    c.save(); c.translate(x, y); c.rotate(rot); c.fillStyle = rgba(h1 > 0.3 ? PAL.osman : [255, 214, 120], al);
    for (let k = 0; k < 4; k++) { c.rotate(TAU / 4); c.beginPath(); c.ellipse(r * 0.55, 0, r * 0.55, r * 0.34, 0, 0, TAU); c.fill(); }
    c.restore();
  }
  c.restore();
}

// ================================================================ the breakfast stall
const ST = 'stall';
function stallSet(c, cam, t, o = {}) {
  L(c, ST, cam, 'st_bg');
  const m = MAN.st_bg;
  inSet(c, ST, cam, 1, () => {
    glow(c, 1536, 900, 1500, PAL.warm, 0.18, { blend: 'screen' });
    if (m.pot) steam(c, t, m.pot[0], m.pot[1], { h: 700, w: 90, a: 0.42, sway: 140, n: 34, per: 0.22, seed: 41 });
    (m.steamers || []).forEach(([x, y], i) => steam(c, t, x, y, { h: 640, w: 70, a: 0.36, sway: 120, n: 30, per: 0.24, seed: 50 + i }));
    if (m.griddle) steam(c, t, m.griddle[0], m.griddle[1], { h: 300, w: 40, a: 0.2, sway: 60, n: 20, per: 0.3, seed: 60 });
  });
  // the boss ladles soy milk: up on the beat, pour, down
  if (MAN.st_hands && m.pot) { const k = kickN(t, 2, 0, 3); S(c, ST, cam, 'st_hands', m.pot[0], m.pot[1] - 40 * k + 8 * Math.sin(t * 2.2)); }
  if (o.fg) L(c, ST, cam, 'st_fg', 1.35);
}
// Pre2 「楼下的老板记得你的口味」
export function stall(c, t, lt, sh) {
  const m = MAN.st_bg.pot || [1536, 900];
  const cam = camK(t, [[sh.t0, 1536, 860, 2600], [sh.t1, m[0], 880, 2200]], { drift: 3 });
  stallSet(c, cam, t, { fg: true });
  return { grade: { warm: 0.2, bloom: 0.8, sat: 1.04, lift: 0.01 } };
}
// Pre2 「你说这些不算什么特别」: across the lane, a little wider; the foreground slides past
export function stall2(c, t, lt, sh) {
  const cam = camK(t, [[sh.t0, 1300, 880, 2900], [sh.t1, 1800, 880, 2900]], { ease: 'lin', drift: 2 });
  stallSet(c, cam, t, { fg: true });
  return { grade: { warm: 0.2, bloom: 0.8, sat: 1.0, lift: 0.01 } };
}
// Pre2 「热豆浆少糖，吸管放旁边」
export function soy(c, t, lt, sh) {
  const SY = 'soy', cup = [1280, 1130];
  const cam = camK(t, [[sh.t0, 1280, 760, 2000], [sh.t1, 1300, 780, 1750]], { drift: 3 });
  L(c, SY, cam, 'sy_bg');
  S(c, SY, cam, 'sy_cup', cup[0], cup[1]);
  const lid = MAN.sy_cup.lid || [0, -520, 220, 70], st = MAN.sy_cup.steam || [0, -540];
  // 少糖 in marker on the lid, foreshortened like the lid
  inSet(c, SY, cam, 1, () => {
    c.save(); c.translate(cup[0] + lid[0], cup[1] + lid[1]); c.scale(1, lid[3] / lid[2] * 1.6);
    handwrite(c, '少糖', 0, 0, [CT(21, 3) - 0.05, CT(21, 4) - 0.02], t, { size: 120, col: 'rgba(30,40,120,0.85)', align: 'center', seed: 4, dur: 0.25, weight: 400 });
    c.restore();
    steam(c, t, cup[0] + st[0], cup[1] + st[1], { h: 520, w: 60, a: 0.35, sway: 90, n: 28, per: 0.24, seed: 70 });
  });
  // the straw, set down beside the cup by the boss's hand
  const dn = CT(21, 9), into = ramp(t, CT(21, 6) - 0.2, dn - CT(21, 6) + 0.2, 'out'), away = ramp(t, dn + 0.35, 0.8, 'in');
  const tx = lerp(cup[0] + 1100, cup[0] + 470, into), ty = lerp(cup[1] - 700, cup[1] - 30, into);
  if (t > dn) S(c, SY, cam, 'sy_straw', cup[0] + 470, cup[1] - 30, 1, { r: -0.08, s: 0.8 });
  if (into > 0 && away < 1) {
    if (t <= dn) S(c, SY, cam, 'sy_straw', tx, ty, 1, { r: -0.08 - 0.3 * (1 - into), s: 0.8 });
    S(c, SY, cam, 'sy_hand', (t <= dn ? tx : cup[0] + 470) + 700 * away, (t <= dn ? ty : cup[1] - 30) - 600 * away, 1, { r: -0.3 * (1 - into), s: 0.72 });
  }
  return { grade: { warm: 0.25, bloom: 0.9, sat: 1.04, lift: 0.01 } };
}

// ================================================================ the river steps
const SP = 'step';
function steps(c, cam, t, o = {}) {
  L(c, SP, cam, 'sp_bg');
  const m = MAN.sp_bg, rf = m.reflect || [0, 700, 2880, 1000];
  // the reflections shiver: thin bright dashes along the water
  inSet(c, SP, cam, 1, () => {
    c.save(); c.globalCompositeOperation = 'screen';
    for (let i = 0; i < 90; i++) {
      const h1 = hash(i * 1.9), h2 = hash(i * 3.7), x = lerp(rf[0], rf[2], h1), y = lerp(rf[1], rf[3], h2 * h2);
      const w = 20 + 60 * hash(i * 5.1), a = 0.25 + 0.35 * Math.max(0, Math.sin(t * (1.5 + 2 * h2) + i));
      c.fillStyle = rgba(h1 > 0.6 ? PAL.amber : h1 > 0.3 ? PAL.warm : [190, 210, 255], a); c.fillRect(x + Math.sin(t * 0.8 + i) * 10, y, w * (0.6 + 0.4 * Math.sin(t * 1.3 + i * 2)), 3);
    }
    c.restore();
    (m.bridge || []).forEach(([x, y], i) => glow(c, x, y, 40, PAL.warm, 0.5 + 0.2 * Math.sin(t * 2 + i), { blend: 'screen' }));
  });
  const me = m.seat_me || [1250, 1452], you = [me[0] + 430, me[1]], br = Math.sin(t * 1.1) * 0.004;     // 430 apart: elbows nearly touch
  // he has taken his jacket off; it lies folded on the step between them, and slides over to her side (0 → 1)
  const coat = o.coat ?? 0, e = E.inOutCubic(clamp(coat));
  S(c, SP, cam, 'sp_you_tee', you[0], you[1], 1, { sy: 1 + br });
  S(c, SP, cam, 'sp_me', me[0], me[1], 1, { sy: 1 - br });
  const jx = lerp(you[0] - 200, me[0] + 175, e), lift = Math.sin(e * Math.PI) * 14;
  S(c, SP, cam, 'sp_coat_fold', jx, me[1] - 70 - lift, 1, { s: 0.5, r: 0.04 - 0.06 * Math.sin(e * Math.PI) });
}
// Bridge 「也许喜欢不是很响的事」
export function step(c, t, lt, sh) {
  const m = MAN.sp_bg, me = m.seat_me || [1260, 1350], you = [me[0] + 430, me[1]];
  const cam = camK(t, [[sh.t0, (me[0] + you[0]) / 2, me[1] - 500, 2700], [sh.t1, (me[0] + you[0]) / 2, me[1] - 420, 2300]], { drift: 2 });
  steps(c, cam, t, { coat: 0 });
  return { grade: { warm: 0.12, bloom: 1.1, sat: 1.0, vig: 0.6 } };
}
// Bridge 「是你把外套往我这边挪」
export function coat(c, t, lt, sh) {
  const m = MAN.sp_bg, me = m.seat_me || [1260, 1350], you = [me[0] + 430, me[1]];
  const cam = camK(t, [[sh.t0, (me[0] + you[0]) / 2, me[1] - 360, 1700], [sh.t1, (me[0] * 0.6 + you[0] * 0.4), me[1] - 380, 1500]], { drift: 2 });
  steps(c, cam, t, { coat: ramp(t, CT(33, 5), CT(33, 9) - CT(33, 5) + 0.4, 'io') });
  return { grade: { warm: 0.2, bloom: 1.1, sat: 1.02, vig: 0.6 } };
}

// ================================================================ the osmanthus shop
const OS = 'shop';
function shopSet(c, cam, t, o = {}) {
  L(c, OS, cam, 'os_bg');
  const m = MAN.os_bg;
  inSet(c, OS, cam, 1, () => {
    if (m.window) { const [x0, y0, x1, y1] = m.window; glow(c, (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * 0.9, PAL.amber, 0.45 + 0.05 * noise1(t, 3), { blend: 'screen' }); glow(c, (x0 + x1) / 2, y1 + 200, (x1 - x0) * 0.8, PAL.warm, 0.2, { sy: 0.3 }); }
  });
  L(c, OS, cam, 'os_tree', 1.15);
  inSet(c, OS, cam, 1.15, () => blossoms(c, t, [1500, 300, 3072, 1728], { n: o.n ?? 50, size: 9, fall: 55 }));
  L(c, OS, cam, 'os_fg', 1.4);
}
// Bridge 「是我经过那家小店的时候」: walking past; the camera trucks along the shop front
export function shop(c, t, lt, sh) {
  const cam = camK(t, [[sh.t0, 1150, 880, 2400], [sh.t1, 1950, 880, 2400]], { ease: 'lin', drift: 2 });
  shopSet(c, cam, t);
  return { grade: { warm: 0.2, bloom: 1.0, sat: 1.04, vig: 0.55 } };
}
// Bridge 「想起你说，桂花味的不错」
export function osmanthus(c, t, lt, sh) {
  const sg = MAN.os_bg.sign || [1500, 600];
  const cam = camK(t, [[sh.t0, sg[0] + 200, sg[1] + 150, 1700], [sh.t1, sg[0] + 260, sg[1] + 120, 1450]], { drift: 3 });
  shopSet(c, cam, t, { n: 90 });
  inSet(c, OS, cam, 1, () => blossoms(c, t + 3, [sg[0] - 900, sg[1] - 600, sg[0] + 1300, sg[1] + 900], { n: 40, size: 8, fall: 45, a: 0.8 }));
  return { grade: { warm: 0.3, bloom: 1.1, sat: 1.08, vig: 0.5 } };
}
