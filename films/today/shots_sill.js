// The window sill and its mint; the entry and your umbrella.
import { W, H, PAL, rgba, L, S, inSet, camT, camK, toScreen, glow, rain, ring, pen, MAN, IM, clamp, lerp, sstep, hash, noise1, makeCanvas, TAU, E } from './stage.js';
import { TM, B, T0, CT, beatNear, beatIdx, kickN, since, ramp } from './core.js';

const [maskCv, maskCx] = makeCanvas(960, 540);
const half = (cx) => { cx.setTransform(0.5, 0, 0, 0.5, 0, 0); cx.globalAlpha = 1; cx.globalCompositeOperation = 'source-over'; cx.clearRect(0, 0, W, H); return cx; };
// a damped spring kick at each of the given times: a small shiver
const shiver = (t, times, f = 22, d = 6) => { let a = 0; for (const x of times) if (t >= x) a += Math.exp(-(t - x) * d) * Math.sin((t - x) * f); return a; };
// a drop that gathers at (x, y), lets go every `per` seconds and falls `fall` px; returns the moments it lands
function drips(c, t, x, y, per, fall, o = {}) {
  const ph = o.ph ?? 0, k = Math.floor((t + ph) / per), u = (t + ph) / per - k, grow = sstep(0, 0.75, u), g = 2600 * (o.g ?? 1);
  const r = (o.r ?? 7) * (0.4 + 0.6 * grow), col = o.col || [230, 238, 246];
  c.save();
  // hanging drop
  // a drop is mostly its dark rim and one highlight: on pale tiles that is all that shows
  c.strokeStyle = rgba([52, 66, 96], 0.55); c.lineWidth = Math.max(1.2, r * 0.22);
  if (u < 0.8) { c.fillStyle = rgba(col, 0.55 * grow); c.beginPath(); c.ellipse(x, y + r * 0.9 * grow, r * 0.8, r * (1 + 0.5 * grow), 0, 0, TAU); c.fill(); c.globalAlpha = grow; c.stroke(); c.globalAlpha = 1; c.fillStyle = rgba([255, 255, 255], 0.7 * grow); c.beginPath(); c.arc(x - r * 0.3, y + r * 0.6 * grow, r * 0.25, 0, TAU); c.fill(); }
  // falling drop
  const tf = (u - 0.8) * per, d = 0.5 * g * tf * tf;
  if (u >= 0.8 && d < fall) { c.fillStyle = rgba(col, 0.6); c.beginPath(); c.ellipse(x, y + r + d, r * 0.55, r * 1.5, 0, 0, TAU); c.fill(); c.stroke(); }
  c.restore();
  // its rings where it lands (this and the one before)
  if (o.ring) for (const kk of [k, k - 1]) {
    const tl = (kk + 0.8) * per - ph + Math.sqrt(2 * fall / g);
    ring(c, x + (o.ring.dx ?? 0), y + fall + (o.ring.dy ?? 0), t - tl, { R: o.ring.R ?? 60, sy: o.ring.sy ?? 0.3, life: 1.2, col: o.ring.col || [210, 220, 230], a: 0.6, w: 2.2, rings: 2, seed: kk });
  }
}

// ================================================================ the sill
const SL = 'sill';
const potAt = () => MAN.sl_frame.pot || [1440, MAN.sl_frame.sill_y ?? 1180];
// o: big (the grown plant), leaf (0..1 new leaf), tw (warped time for the rain outside), rain (0..1), sun, hand (0..1)
function sill(c, cam, t, o = {}) {
  L(c, SL, cam, 'sl_out', 0.72);
  if ((o.rain ?? 1) > 0.01) { const tw = o.tw ?? t; c.save(); c.globalAlpha = o.rain ?? 1; rain(c, tw, { n: 140, a: 0.22, len: 70, seed: 5, v: 1500, slant: -0.06 }); c.restore(); }
  L(c, SL, cam, 'sl_grille', 0.86);
  L(c, SL, cam, 'sl_frame');
  const [px, py] = potAt();
  if (o.sun) inSet(c, SL, cam, 1, () => { glow(c, px - 300, py - 700, 1500, PAL.sun, 0.35 * o.sun, { blend: 'screen' }); });
  // the plant breathes a little; each leaf layer is one piece, so it sways about its base
  const sw = 0.012 * Math.sin(t * 0.9) + 0.006 * noise1(t * 0.5, 3), name = o.big ? 'sl_mint_big' : 'sl_mint';
  S(c, SL, cam, 'sl_pot', px, py);
  const soil = MAN.sl_pot.soil || [0, -180], bx = px + soil[0], by = py + soil[1];
  S(c, SL, cam, name, bx, by, 1, { r: sw, s: o.grow ?? 1 });
  // the new leaf: up from nothing at a stem tip, turning into place
  if (o.leaf > 0 && MAN.sl_leaf) {
    const tips = MAN[name].tips || [[0, -400, 0]], tp = tips[o.tipI ?? 0], e = E.outBack(clamp(o.leaf));
    const ca = Math.cos(sw), sa = Math.sin(sw), lx = bx + tp[0] * ca - tp[1] * sa, ly = by + tp[0] * sa + tp[1] * ca;
    S(c, SL, cam, 'sl_leaf', lx, ly, 1, { s: 0.05 + 0.95 * e, r: (tp[2] || 0) * Math.PI / 180 + sw + 0.5 * (1 - e) });     // tips give their angle in degrees
  }
  if (o.hand > 0 && MAN.sl_hand) S(c, SL, cam, 'sl_hand', px + 420 + 500 * (1 - o.hand), py + 30 + 200 * (1 - o.hand));
}
function sillGlass(cam) {
  const m = half(maskCx), panes = MAN.sl_frame.panes || [];
  inSet(m, SL, cam, 1, () => { m.fillStyle = '#fff'; for (const p of panes) m.fillRect(p[0], p[1], p[2] - p[0], p[3] - p[1]); });
  // the plant stands in front of the glass: it is not wet
  const [px, py] = potAt(), soil = MAN.sl_pot.soil || [0, -180];
  S(m, SL, cam, 'sl_pot', px, py, 1, { blend: 'destination-out' });
  S(m, SL, cam, 'sl_mint', px + soil[0], py + soil[1], 1, { blend: 'destination-out' });
  return maskCv;
}
// Pre1 「窗边那盆薄荷长出新叶 / 我却忘了它什么时候变绿」: a leaf; and the green comes out of the mint into a grey afternoon
export function mint(c, t, lt, sh) {
  const [px, py] = potAt();
  const cam = camK(t, [[sh.t0, px - 60, py - 420, 2500], [CT(4, 9) + 0.4, px - 20, py - 380, 1900, 0, 'io'], [sh.t1, px, py - 360, 1650]], { drift: 3 });
  sill(c, cam, t, { leaf: ramp(t, CT(4, 6), CT(4, 9) - CT(4, 6) + 0.4, 'lin') });
  const [sx, sy] = toScreen(SL, cam, px, py - 330), r = lerp(0, 0.22, ramp(t, CT(4, 6), 1.5)) + lerp(0, 2.6, ramp(t, CT(5, 9) - 0.1, 1.6, 'in'));
  return { glass: { rain: 0.75, beads: 0.6, fog: 0.35, mask: sillGlass(cam), scale: 1.2, seed: 30 }, grade: { sat: 0.28, cold: 0.45, satIn: 1.12, wave: [sx / H, sy / H, r], warm: 0.1, bloom: 0.6 } };
}
// Pre1 「我说那就慢一点，等雨停」: the rain slows almost to stopping; my hand comes to rest by the pot
export function slow(c, t, lt, sh) {
  const [px, py] = potAt(), len = sh.len + 1, tw = sh.t0 + lt - 0.88 * lt * lt / (2 * len);
  const cam = camK(t, [[sh.t0, px + 120, py - 330, 2100], [sh.t1, px + 160, py - 320, 1950]], { drift: 1.5 });
  sill(c, cam, t, { tw, rain: 1 - 0.6 * ramp(t, sh.t0 + 1, sh.len), hand: ramp(t, CT(7, 7), 1.2, 'out'), leaf: 1 });
  return { glass: { rain: 0.75, beads: 0.6, fog: 0.3, mask: sillGlass(cam), scale: 1.2, seed: 30, time: tw }, grade: { sat: 1.0, warm: 0.12, bloom: 0.6 } };
}
// Ch3 「不用保证以后会怎样」: the mint in morning sun, grown
export function mint2(c, t, lt, sh) {
  const [px, py] = potAt();
  const cam = camK(t, [[sh.t0, px - 80, py - 460, 2000], [sh.t1, px - 40, py - 430, 1750]], { drift: 2 });
  sill(c, cam, t, { big: true, rain: 0, sun: 1 });
  return { glass: { rain: 0.0, beads: 0.45, fog: 0.08, mask: sillGlass(cam), scale: 1.2, seed: 33 }, grade: { warm: 0.3, bloom: 0.9, sat: 1.1, lift: 0.015 } };
}
// Outro 「薄荷又长高」
export function mintEnd(c, t, lt, sh) {
  const [px, py] = potAt();
  const cam = camK(t, [[sh.t0, px, py - 470, 2050], [sh.t1, px, py - 500, 1850]], { drift: 2 });
  sill(c, cam, t, { big: true, rain: 0, sun: 1.2, grow: 1 + 0.06 * ramp(t, sh.t0, sh.len, 'out') });
  return { glass: { rain: 0.0, beads: 0.3, fog: 0.05, mask: sillGlass(cam), scale: 1.2, seed: 34 }, grade: { warm: 0.36, bloom: 1.0, sat: 1.12, lift: 0.02 } };
}

// ================================================================ the entry
const EN = 'entry';
function entryRoom(c, cam, t, o = {}) {
  L(c, EN, cam, 'en_bg');
  // the painter gives the lean in degrees and the umbrella's scale on this sheet
  const [ux, uy, ud] = MAN.en_bg.umb || [930, 1086, -6.8], us = MAN.en_bg.umb_scale ?? 0.426, name = o.fixed ? 'en_umb_fixed' : 'en_umb';
  const ua = ud * Math.PI / 180, wob = o.wob ?? 0;
  S(c, EN, cam, name, ux, uy, 1, { r: ua + wob, s: us });
  if (!o.fixed) inSet(c, EN, cam, 1, () => {
    const m = MAN.en_umb, td0 = m.tip_drip || [0, -10], pd = MAN.en_bg.puddle || [ux, uy, 120, 30];
    const td = [(td0[0] * Math.cos(ua) - td0[1] * Math.sin(ua)) * us, (td0[0] * Math.sin(ua) + td0[1] * Math.cos(ua)) * us];
    // the pool, and drops that let go about once a bar
    c.save(); c.globalCompositeOperation = 'multiply'; c.fillStyle = 'rgba(150,165,185,0.35)'; c.beginPath(); c.ellipse(pd[0], pd[1], pd[2], pd[3], 0, 0, TAU); c.fill(); c.restore();
    drips(c, t, ux + td[0], uy + td[1], 60 / TM.tempo * 2, Math.max(8, pd[1] - uy - td[1]), { r: 4, ring: { R: pd[2] * 0.9, sy: pd[3] / pd[2] } });
  });
  if (o.sun) inSet(c, EN, cam, 1, () => glow(c, 900, 300, 1500, PAL.sun, 0.3 * o.sun, { blend: 'screen' }));
}
// V2 「你借我的伞还放在玄关」
export function entry(c, t, lt, sh) {
  const [ux, uy] = MAN.en_bg.umb || [930, 1086];
  const cam = camK(t, [[sh.t0, 1240, 720, 2400], [sh.t1, ux + 160, uy - 330, 1500]], { drift: 2 });
  entryRoom(c, cam, t);
  return { grade: { warm: -0.05, bloom: 0.5, sat: 0.92, cold: 0.1 } };
}
// V2 「伞骨有一点歪，收不太严」: close on it; it shivers where the rib will not lie down
export function bent(c, t, lt, sh) {
  const [ux, uy] = MAN.en_bg.umb || [930, 1086], us = MAN.en_bg.umb_scale ?? 0.426, rib = MAN.en_umb.rib || [190, -570];
  const cam = camK(t, [[sh.t0, ux + rib[0] * us * 0.5, uy + rib[1] * us * 0.85, 760], [sh.t1, ux + rib[0] * us * 0.6, uy + rib[1] * us * 0.95, 640]], { drift: 3 });
  entryRoom(c, cam, t, { wob: 0.02 * shiver(t, [CT(17, 5), CT(17, 9), CT(17, 10)]) });
  return { grade: { warm: -0.05, bloom: 0.5, sat: 0.95 } };
}
// V2 「我试着修好，弄得满手水」
export function fix(c, t, lt, sh) {
  const FX = 'fix', cam = camK(t, [[sh.t0, 1280, 720, 2300], [sh.t1, 1300, 700, 2000]], { drift: 3 });
  L(c, FX, cam, 'fx_bg');
  const hub = [1280, 720], rot = 0.05 * Math.sin(t * 0.7) + 0.02 * shiver(t, [CT(18, 2), CT(18, 4)], 18, 4);
  S(c, FX, cam, 'fx_umb', hub[0], hub[1], 1, { r: rot });
  const fu = MAN.fx_umb, bentP = fu.bent || [300, 100], ca = Math.cos(rot), sa = Math.sin(rot);
  const gx = hub[0] + bentP[0] * ca - bentP[1] * sa, gy = hub[1] + bentP[0] * sa + bentP[1] * ca;
  // the hands work at it: small tugs
  const tug = 6 * Math.sin(t * 9) * sstep(sh.t0, sh.t0 + 0.4, t) + 10 * shiver(t, [CT(18, 2), CT(18, 4)], 20, 5);
  S(c, FX, cam, 'fx_hands', gx + tug, gy + tug * 0.4, 1, { r: rot * 0.5 });
  inSet(c, FX, cam, 1, () => {
    const more = sstep(CT(18, 7), CT(18, 9), t);
    (fu.ribtips || []).forEach(([dx, dy], i) => { const x = hub[0] + dx * ca - dy * sa, y = hub[1] + dx * sa + dy * ca; drips(c, t, x, y, 0.9 + 0.5 * hash(i), 140, { ph: hash(i * 3) * 2, r: 7, ring: { R: 50, sy: 0.45, col: [70, 90, 120] } }); });
    (MAN.fx_hands.drips || []).forEach(([dx, dy], i) => { if (i < 2 + 4 * more) drips(c, t, gx + dx, gy + dy, 0.55 + 0.3 * hash(i + 7), 120, { ph: hash(i * 5 + 1), r: 6, ring: { R: 40, sy: 0.45, col: [70, 90, 120] } }); });
  });
  return { grade: { warm: 0.0, bloom: 0.6, sat: 0.98 } };
}
// Outro 「伞还在玄关」: the rib is straight now; morning sun
export function home(c, t, lt, sh) {
  const [ux, uy] = MAN.en_bg.umb || [930, 1086];
  const cam = camK(t, [[sh.t0, ux + 250, uy - 400, 2000], [sh.t1, ux + 230, uy - 420, 1750]], { drift: 2 });
  entryRoom(c, cam, t, { fixed: true, sun: 1 });
  return { grade: { warm: 0.3, bloom: 0.8, sat: 1.05, lift: 0.015 } };
}

// the entry as a small picture, for the shot that leaves it behind in one window of the block
let EF;
export function entryFrame(t) {
  if (!EF) EF = makeCanvas(1440, 810);
  const [cv, cx] = EF; cx.setTransform(0.75, 0, 0, 0.75, 0, 0); cx.globalAlpha = 1; cx.globalCompositeOperation = 'source-over'; cx.filter = 'none';
  const [ux, uy] = MAN.en_bg.umb || [930, 1086];
  entryRoom(cx, { x: ux + 200, y: uy - 380, w: 1700, r: 0 }, t);
  cx.setTransform(1, 0, 0, 1, 0, 0); cx.globalCompositeOperation = 'multiply'; cx.fillStyle = 'rgb(255,232,190)'; cx.fillRect(0, 0, 1440, 810);
  return cv;
}
