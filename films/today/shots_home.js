// At home: the fridge door and its notes, the kitchen in the morning, the table from above.
import { W, H, PAL, rgba, L, S, inSet, camT, camK, toScreen, glow, lamp, steam, pen, chalk, squarePts, handwrite, lightPatch, MAN, IM, clamp, lerp, sstep, hash, noise1, makeCanvas, TAU, E } from './stage.js';
import { TM, B, T0, CT, beatNear, beatIdx, kickN, since, ramp } from './core.js';

const INK = 'rgba(40,34,46,0.92)';
// motes of dust turning in a sunbeam: frame px, closed form
function motes(c, t, box, o = {}) {
  const n = o.n ?? 40, a = o.a ?? 0.55;
  c.save(); c.setTransform(1, 0, 0, 1, 0, 0); c.globalCompositeOperation = 'screen';
  for (let i = 0; i < n; i++) {
    const h1 = hash(i * 1.3 + 7), h2 = hash(i * 2.9 + 1), h3 = hash(i * 4.1 + 3);
    const x = lerp(box[0], box[2], (h1 + 0.012 * t * (h3 - 0.4) + 0.02 * Math.sin(t * 0.3 + i)) % 1), y = lerp(box[1], box[3], (h2 + 0.006 * t + 0.015 * Math.sin(t * 0.21 + i * 2)) % 1);
    const r = 1.2 + 2.6 * h3, tw = 0.5 + 0.5 * Math.sin(t * (0.8 + h1) + i);
    c.fillStyle = rgba(PAL.sun, a * tw * (0.4 + 0.6 * h2)); c.beginPath(); c.arc(x, y, r, 0, TAU); c.fill();
  }
  c.restore();
}
// soft leaf shadows from a tree outside the window, swaying: multiply blobs in the current transform
function leafShade(c, t, box, o = {}) {
  const n = o.n ?? 14, a = o.a ?? 0.10;
  c.save(); c.globalCompositeOperation = 'multiply'; c.filter = `blur(${o.blur ?? 18}px)`;
  for (let i = 0; i < n; i++) {
    const h1 = hash(i * 3.3 + 2), h2 = hash(i * 5.7 + 4), sw = Math.sin(t * 0.7 + h1 * 6) * 18 + noise1(t * 0.4, i) * 14;
    const x = lerp(box[0], box[2], h1) + sw, y = lerp(box[1], box[3], h2) + sw * 0.3;
    c.fillStyle = rgba([150, 160, 190], a * (0.6 + 0.4 * h2));
    c.beginPath(); c.ellipse(x, y, 40 + 70 * h2, 22 + 30 * h1, h1 * 3, 0, TAU); c.fill();
  }
  c.filter = 'none'; c.restore();
}

// ================================================================ the fridge door
const FR = 'fridge';
const NOTE = () => MAN.fr_bg.note_main;             // [x, y, rot]: top centre of the first note
// a note on the door: its shadow, the note, and what is written on it (fn(c) in note-local px, origin at the anchor)
function noteOn(c, cam, name, x, y, r, o = {}) {
  const lift = o.lift ?? 0, s = o.s ?? 1, a = o.a ?? 1;
  if (a <= 0.003) return;
  if (MAN.fr_note_sh) S(c, FR, cam, 'fr_note_sh', x + 10 * lift, y + 16 * lift, 1, { r, s: s * (1 + 0.03 * lift), a: a * (1 - 0.4 * lift) });
  S(c, FR, cam, name, x, y, 1, { r, s: s * (1 + 0.035 * lift), a });
  if (o.write) inSet(c, FR, cam, 1, () => { c.save(); c.translate(x, y); c.rotate(r); c.scale(s, s); c.globalAlpha *= a; o.write(c); c.restore(); });
}
// the area inside the note where writing goes, in note-local px: [x0, y0, x1, y1]
const WB = () => MAN.fr_note.write || [-170, 90, 170, 380];
// the first note's words. t1: times "牛奶" and "别忘了问" are written (sung)
const firstWords = (t) => (c) => {
  const [x0, y0, x1] = WB(), cx = (x0 + x1) / 2;
  handwrite(c, '牛奶', cx - 60, y0 + 70, [CT(1, 2) - 0.08, CT(1, 3) - 0.08], t, { size: 78, col: INK, seed: 1, align: 'center' });
  handwrite(c, '别忘了问', cx + 4, y0 + 190, [CT(1, 5) - 0.06, CT(1, 6) - 0.06, CT(1, 7) - 0.06, CT(1, 8) - 0.06], t, { size: 66, col: INK, seed: 2, align: 'center' });
  // a small underline under 问, the pen's afterthought
  pen(c, [[cx + 40, y0 + 236], [cx + 90, y0 + 240], [cx + 128, y0 + 234]], ramp(t, CT(1, 8) + 0.3, 0.35), { color: INK, w: 3.2, seed: 3, tremor: 1.2 });
};
function fridgeLight(c, cam, t, o = {}) {
  inSet(c, FR, cam, 1, () => {
    leafShade(c, t, [300, 100, 2300, 1300], { a: o.shade ?? 0.12 });
    glow(c, 700, 300, 1400, PAL.sun, (o.sun ?? 0.18), { blend: 'soft-light' });
  });
  motes(c, t, [0, 0, W, H], { n: 36, a: 0.4 * (o.motes ?? 1) });
}

// V1 「你把便利贴贴在冰箱门 / 写着牛奶，还有别忘了问」
export function note(c, t, lt, sh) {
  const [nx, ny, nr] = NOTE(), press = CT(0, 5) - 0.04;      // the second 贴
  const cam = camK(t, [[0, nx + 80, ny + 300, 2250], [press + 0.4, nx + 40, ny + 260, 2050, 0, 'out'], [sh.t1, nx, ny + 250, 1500]], { drift: 3 });
  L(c, FR, cam, 'fr_bg');
  // the note travels in with the hand, is pressed on with a small bounce
  const into = ramp(t, 0.15, press - 0.15, 'out'), settle = t > press ? Math.exp(-(t - press) * 9) * Math.sin((t - press) * 30) : 0;
  const hx = lerp(nx + 900, nx, into), hy = lerp(ny + 760, ny, into), lift = 1 - sstep(press - 0.1, press + 0.08, t);
  noteOn(c, cam, 'fr_note', hx, hy + 6 * settle, nr + 0.12 * (1 - into) + 0.01 * settle, { lift, write: firstWords(t) });
  // your hand: in with the note, presses, leaves
  const out = ramp(t, press + 0.55, 1.1, 'in'), hu = MAN.fr_hand_you;
  if (out < 1 && hu) { const wb = WB(); S(c, FR, cam, 'fr_hand_you', hx + 20 + 900 * out, hy + (wb[1] + wb[3]) / 2 + 40 + 560 * out, 1, { r: -0.15 * out, s: 0.74 }); }
  fridgeLight(c, cam, t);
  return { grade: { warm: 0.22, bloom: 0.6, sat: 1.02, lift: 0.01 } };
}

// Ch3 「就把明天留一小块给我」: I write "明天" on a mint note and put it beside yours; the pen goes round both
export function tomorrow(c, t, lt, sh) {
  const [nx, ny, nr] = NOTE(), sp = (MAN.fr_bg.note_spots || [[nx + 480, ny + 40, 0.05]])[0];
  const mx = nx + 470, my = ny + 30, mr = 0.06;
  const cam = camK(t, [[sh.t0, nx + 200, ny + 260, 1900], [sh.t1, nx + 240, ny + 240, 1650]], { drift: 3 });
  L(c, FR, cam, 'fr_bg');
  oldNotes(c, cam, [[nx, ny], [mx, my]]);
  noteOn(c, cam, 'fr_note', nx, ny, nr, { write: firstWords(1e9) });
  const wr = [CT(37, 2) - 0.1, CT(37, 3) - 0.05];
  const press = sh.t0 + 0.25, into = ramp(t, sh.t0 - 0.2, 0.45, 'out');
  noteOn(c, cam, 'fr_note_mint', lerp(mx + 300, mx, into), lerp(my + 200, my, into), mr + 0.1 * (1 - into), { lift: 1 - sstep(press - 0.1, press + 0.1, t), write: (cc) => {
    const [x0, y0, x1] = WB(); handwrite(cc, '明天', (x0 + x1) / 2, y0 + 120, wr, t, { size: 96, col: INK, seed: 5, align: 'center', dur: 0.3 });
  } });
  // my hand writing: the pen tip follows the characters being written
  const hm = MAN.fr_hand_me;
  if (hm) {
    const [x0, y0, x1] = WB(), u = clamp((t - wr[0]) / (wr[1] + 0.3 - wr[0])), wob = Math.sin(t * 26) * 6 * (u > 0 && u < 1 ? 1 : 0);
    const away = ramp(t, wr[1] + 0.5, 0.8, 'in') + (1 - ramp(t, sh.t0, 0.6, 'out'));
    const px = mx + (x0 + x1) / 2 - 90 + 190 * u + wob, py = my + y0 + 130 + 10 * Math.sin(u * 9);
    S(c, FR, cam, 'fr_hand_me', px + 700 * away, py + 500 * away, 1, { r: 0.04 });
  }
  // the pen square round the two notes, on 「一小块」
  inSet(c, FR, cam, 1, () => {
    const u = ramp(t, CT(37, 4) - 0.05, CT(37, 7) - CT(37, 4) + 0.2, 'io');
    pen(c, squarePts(nx + 235, ny + 230, 420, -0.02, 7).map(([x, y]) => [x, y * 0.9 + (ny + 230) * 0.1]), u, { color: 'rgba(214,150,30,0.9)', w: 7, seed: 9, tremor: 2.5 });
  });
  fridgeLight(c, cam, t, { sun: 0.22 });
  return { grade: { warm: 0.28, bloom: 0.7, sat: 1.06 } };
}

// Pre2 「可我记住了，记了好多天」: one note more on every beat, each with the small thing it remembers
const NOTE_COLS = ['fr_note', 'fr_note_pink', 'fr_note_mint', 'fr_note_blue'];
// the notes of 「记了好多天」 stay up afterwards; avoid: [[x, y], ...] spots taken by the big notes of a later shot
function oldNotes(c, cam, avoid = [], maxY = 1e9) {
  (MAN.fr_bg.note_spots || []).forEach(([x, y, r], i) => {
    if (y > maxY || avoid.some(([ax, ay]) => Math.hypot(x - ax, y - ay) < 400)) return;
    noteOn(c, cam, NOTE_COLS[(i + 1) % NOTE_COLS.length], x, y, r, { s: 0.62, write: (cc) => { const [x0, y0, x1] = WB(); handwrite(cc, REMEMBER[i % REMEMBER.length], (x0 + x1) / 2, y0 + 90, [0, 0, 0, 0, 0], 1e9, { size: 70, col: INK, seed: 20 + i, align: 'center' }); } });
  });
}
const REMEMBER = ['少糖', '吸管\n放旁边', '伞骨\n歪了', '薄荷\n浇水', '桂花', '雨停了', '热豆浆', '巷口', '早点睡', '外套', '慢慢说', '留一小块', '晚安', '记得'];
export function notes(c, t, lt, sh) {
  const [nx, ny, nr] = NOTE(), spots = MAN.fr_bg.note_spots || [];
  const cam = camK(t, [[sh.t0, nx, ny + 250, 1500], [sh.t1, 1280, 720, 2560]], { ease: 'io', drift: 2 });
  L(c, FR, cam, 'fr_bg');
  noteOn(c, cam, 'fr_note', nx, ny, nr, { write: firstWords(1e9) });
  // beats from just after the cut to the end of the line
  const b0 = beatIdx(sh.t0 + 0.15) + 1;
  spots.forEach(([x, y, r], i) => {
    const tb = B[b0 + Math.floor(i * 0.75)] ?? sh.t1; if (t < tb - 0.25) return;
    const u = ramp(t, tb - 0.25, 0.3, 'out'), lift = 1 - sstep(tb - 0.05, tb + 0.06, t);
    noteOn(c, cam, NOTE_COLS[(i + 1) % NOTE_COLS.length], x + 60 * (1 - u), y + 90 * (1 - u), r + 0.15 * (1 - u), { a: sstep(0, 0.3, u), lift, s: 0.82, write: (cc) => {
      const [x0, y0, x1] = WB(); handwrite(cc, REMEMBER[i % REMEMBER.length], (x0 + x1) / 2, y0 + 90, [tb, tb + 0.05, tb + 0.1, tb + 0.15, tb + 0.2], t, { size: 70, col: INK, seed: 20 + i, align: 'center', dur: 0.15 });
    } });
  });
  fridgeLight(c, cam, t, { sun: 0.2 });
  return { grade: { warm: 0.26, bloom: 0.6, sat: 1.05 } };
}

// Outro 「你若有空，我们再走一遭」: a third note to the left of yours asks; then the camera steps back from all three
export function end(c, t, lt, sh) {
  const [nx, ny, nr] = NOTE(), mx = nx + 470, my = ny + 30, x3 = nx - 470, y3 = ny + 40;
  const wr = [CT(45, 0), CT(45, 1), CT(45, 2) + 0.1, CT(45, 3) + 0.2], back = TM.lines[45].t1 - 0.6;
  const cam = camK(t, [[sh.t0, x3 + 100, y3 + 260, 1500], [back, x3 + 60, y3 + 240, 1250], [back + 3.2, nx + 20, ny + 330, 1750]], { drift: 2 });
  L(c, FR, cam, 'fr_bg');
  oldNotes(c, cam, [[nx, ny], [mx, my], [x3, y3]], 760);      // the lower door is left clear for the end card
  noteOn(c, cam, 'fr_note', nx, ny, nr, { write: firstWords(1e9) });
  noteOn(c, cam, 'fr_note_mint', mx, my, 0.06, { write: (cc) => {
    const [x0, y0, x1] = WB(); handwrite(cc, '明天', (x0 + x1) / 2, y0 + 120, [0, 0], 1e9, { size: 96, col: INK, seed: 5, align: 'center' });
  } });
  noteOn(c, cam, 'fr_note_pink', x3, y3, -0.05, { write: (cc) => {
    const [x0, y0, x1] = WB(); handwrite(cc, '有空吗？', (x0 + x1) / 2, y0 + 120, wr, t, { size: 80, col: INK, seed: 8, align: 'center', dur: 0.3 });
  } });
  {
    const [x0, y0, x1] = WB(), tip = [0, 0], u = clamp((t - wr[0]) / (wr[3] + 0.3 - wr[0])), wob = Math.sin(t * 24) * 6 * (u > 0 && u < 1 ? 1 : 0), away = ramp(t, wr[3] + 0.5, 0.9, 'in') + (1 - ramp(t, sh.t0 - 0.2, 0.5, 'out'));
    S(c, FR, cam, 'fr_hand_me', x3 + (x0 + x1) / 2 - 140 + 280 * u + wob + 700 * away, y3 + y0 + 120 + 8 * Math.sin(u * 11) + 700 * away, 1, { r: 0.04 });
  }
  fridgeLight(c, cam, t, { sun: 0.24 });
  return { grade: { warm: 0.3, bloom: 0.6, sat: 1.05 }, card: sstep(TM.lines[45].t1 + 0.5, TM.lines[45].t1 + 1.5, t) };     // after the last lyric has gone
}

// ================================================================ the kitchen, wide
const KT = 'kitchen';
function kitchenRoom(c, cam, t, o = {}) {
  const m = MAN.kt_bg, sun = o.sun ?? 0.55;
  L(c, KT, cam, 'kt_bg');
  const day = o.day ?? 0.35;                                  // 0 morning (patch near the window) → 1 evening (far across, up the wall)
  // the window's light on the floor and the wall goes behind the furniture: kt_occ is the room's furniture cut out
  inSet(c, KT, cam, 1, () => {
    c.save(); c.globalCompositeOperation = 'multiply'; c.fillStyle = 'rgba(176,170,196,0.55)'; c.fillRect(-500, -500, 4200, 2800); c.restore();
    const fq = m.floor, rise = Math.sin(Math.PI * clamp(day * 1.05 + 0.04));
    // on the floor: from under the window across toward the fridge as the day goes; up the wall in the afternoon
    if (fq) { const u = lerp(0.04, 0.66, day); lightPatch(c, fq, u, 0.22, u + 0.24, 0.92, 1.25 * sun * rise, { bars: 2, soft: 7, gap: 0.07 }); }
    if (m.wall) { const u = lerp(0.98, 0.40, day); lightPatch(c, m.wall, u - 0.14, 0.60 - 0.22 * day, u, 0.86 - 0.22 * day, 0.8 * sun * sstep(0.3, 0.7, day) * rise, { bars: 2, soft: 9, gap: 0.07 }); }
  });
  L(c, KT, cam, 'kt_occ');
  inSet(c, KT, cam, 1, () => {
    if (m.window) { const [x0, y0, x1, y1] = m.window; glow(c, (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * 1.1, PAL.sun, 0.3 * sun / 0.55, { blend: 'screen' }); }
    if (m.kettle_spout) steam(c, t, m.kettle_spout[0], m.kettle_spout[1], { h: 330, w: 26, a: 0.30 * (o.steam ?? 1), sway: 60, n: 24, per: 0.25, seed: 3 });
    if (m.burner) glow(c, m.burner[0], m.burner[1], 50, [120, 160, 255], 0.5 * (o.steam ?? 1) * (0.85 + 0.15 * noise1(t * 9, 2)), { blend: 'screen' });
    if (m.fridge_note) glow(c, m.fridge_note[0], m.fridge_note[1], 70, PAL.note, 0.22 + 0.18 * kickN(t, 4), { blend: 'screen' });
  });
}
// V1 「我昨晚说到一半的事情」
export function kitchen(c, t, lt, sh) {
  const [sw, shh] = [3072, 1728];
  const cam = camK(t, [[sh.t0, sw * 0.56, shh * 0.52, 2700], [sh.t1, sw * 0.48, shh * 0.52, 2450]], { drift: 3 });
  kitchenRoom(c, cam, t, { day: 0.18 + 0.03 * (t - sh.t0) / sh.len });
  motes(c, t, [W * 0.05, H * 0.1, W * 0.5, H * 0.8], { n: 50, a: 0.5 });
  return { grade: { warm: 0.25, bloom: 0.7, sat: 1.0, lift: 0.01 } };
}
// Pre1 「你说最近总觉得日子很快」: the day goes round and round
export function rush(c, t, lt, sh) {
  const cyc = 2.5, u = (t - sh.t0) / sh.len * cyc, day = u % 1, dark = sstep(0.8, 1, day) * (1 - sstep(0.97, 1, day));
  const cam = camK(t, [[sh.t0, 1500, 880, 2900], [sh.t1, 1560, 870, 2600]], { drift: 2 });
  kitchenRoom(c, cam, t * 6, { day, sun: 0.75, steam: 1.5 });
  c.save(); c.globalCompositeOperation = 'multiply'; c.fillStyle = rgba([110, 120, 170], 0.45 * sstep(0.62, 0.92, day) * (1 - sstep(0.96, 1, day)));                 // dusk falls, and is morning again c.fillRect(0, 0, W, H); c.restore();
  return { grade: { warm: lerp(0.4, -0.2, sstep(0.55, 0.9, day)), bloom: 0.8, sat: 1.0 } };
}

// ================================================================ the table from above
const TB = 'table';
function tableTop(c, cam, t, o = {}) {
  L(c, TB, cam, 'tb_bg');
  const [cx, cy] = MAN.tb_bg.centre || [1536, 864];
  const night = o.night ?? 0;
  if (o.sunSquare) inSet(c, TB, cam, 1, () => lightPatch(c, [[cx - 900, cy - 520], [cx + 900, cy - 520], [cx + 900, cy + 520], [cx - 900, cy + 520]], o.sunSquare[0], o.sunSquare[1], o.sunSquare[2], o.sunSquare[3], o.sunA ?? 0.5, { bars: 1, soft: 6, gap: 0.05 }));
  if (!night) inSet(c, TB, cam, 1, () => leafShade(c, t, [cx - 1200, cy - 700, cx + 1200, cy + 700], { a: 0.08, n: 18 }));
  return [cx, cy];
}
function nightLamp(c, cam, t, cx0, cy0, lv = 1) {
  // the lamp hangs on a cord and is never quite still: its pool sways a little
  const cx = cx0 + 40 * Math.sin(t * 0.55) + 12 * noise1(t * 0.4, 4), cy = cy0 + 25 * Math.sin(t * 0.55 + 1.2);
  c.save(); c.globalCompositeOperation = 'multiply'; c.fillStyle = rgba([80, 84, 130]); c.fillRect(0, 0, W, H); c.restore();
  // one lamp hangs low over the middle of the table: a warm pool, the edges left in the dark
  inSet(c, TB, cam, 1, () => { glow(c, cx, cy, 1100, PAL.amber, 0.42 * lv, { blend: 'soft-light', mid: 0.5 }); glow(c, cx, cy, 700, PAL.warm, 0.22 * lv, { blend: 'screen', mid: 0.5 }); });
}
// V1 「今天醒来，还是没有下文」: I start to write about last night and stop
export function tableAM(c, t, lt, sh) {
  const cam = camK(t, [[sh.t0, 1536, 900, 2300], [sh.t1, 1560, 930, 2000]], { drift: 2 });
  const [cx, cy] = tableTop(c, cam, t, { sunSquare: [0.05, 0.1, 0.45, 0.6], sunA: 0.45 });
  const you = [cx - 520, cy - 280], me = [cx + 560, cy - 120], pad = [cx + 40, cy + 120];
  S(c, TB, cam, 'tb_mug_you', you[0], you[1]);
  S(c, TB, cam, 'tb_mug_me', me[0], me[1]);
  inSet(c, TB, cam, 1, () => steam(c, t, me[0], me[1], { h: 260, w: 30, a: 0.25, sway: 50, n: 22, per: 0.27, seed: 8 }));
  S(c, TB, cam, 'tb_pad', pad[0], pad[1], 1, { r: -0.04 });
  const wb = MAN.tb_pad.write || [-150, -150, 150, 150];
  const wr = [CT(3, 0), CT(3, 1), CT(3, 2), CT(3, 3)].map((x) => x + 0.05);
  inSet(c, TB, cam, 1, () => { c.save(); c.translate(pad[0], pad[1]); c.rotate(-0.04); handwrite(c, '关于昨晚', wb[0] + 30, (wb[1] + wb[3]) / 2 - 20, wr, t, { size: 58, col: INK, seed: 11, dur: 0.25 }); c.restore(); });
  // my hand writes, stops; on 「没有下文」 the pen taps twice and lifts
  const hm = MAN.tb_hand_me_pen;
  if (hm) {
    const tip = hm.tip || [0, -400], u = clamp((t - wr[0]) / (wr[3] + 0.3 - wr[0])), wob = Math.sin(t * 25) * 5 * (u > 0 && u < 1 ? 1 : 0);
    const tap = Math.max(Math.exp(-since(t, [CT(3, 7), CT(3, 9)]) * 10), 0) * 14, hover = sstep(wr[3] + 0.4, wr[3] + 0.9, t) * 30;
    const into = 1 - ramp(t, sh.t0 - 0.1, 0.6, 'out');
    const px = pad[0] + wb[0] + 50 + 250 * u + wob, py = pad[1] + (wb[1] + wb[3]) / 2 - 10 - hover + tap + 600 * into;
    S(c, TB, cam, 'tb_hand_me_pen', px - tip[0], py - tip[1], 1, { ax: undefined });
  }
  return { grade: { warm: 0.22, bloom: 0.6, sat: 1.0 } };
}
// Ch1 「我听你把沉默说完就够」: night, two mugs, two steams that find each other, two hands that almost touch
function nightTable(c, t, sh, o = {}) {
  const [cx, cy] = MAN.tb_bg.centre || [1300, 864];
  const cam = camK(t, [[sh.t0, cx - 60, cy + 40, 2700, -0.02], [sh.t1, cx + 30, cy - 20, 2350, 0.015]], { drift: 3 });
  tableTop(c, cam, t, { night: 1 });
  const you = [cx - 700, cy - 240], me = [cx + 700, cy + 230];
  S(c, TB, cam, 'tb_mug_you', you[0], you[1]);
  S(c, TB, cam, 'tb_mug_me', me[0], me[1]);
  if (o.phone) o.phone(cam, cx, cy);
  // place each hand by its fingertips: they come to rest a hand's breadth apart
  const br = Math.sin(t * 1.3) * 6, ty = MAN.tb_hand_you.tip || [0, 1150], tm = MAN.tb_hand_me.tip || [0, -1150];
  const iy = lerp(-420, 0, ramp(t, sh.t0, 0.9, 'out')), im = lerp(420, 0, ramp(t, sh.t0 + 0.3, 0.9, 'out'));
  // through the line the two hands drift closer, and stop a fingertip apart
  const near = ramp(t, sh.t0 + 1.0, sh.len - 0.6, 'io') * 70;
  const reach = o.reach ? o.reach(t) : null;
  if (reach) { const [k, px, py] = reach; S(c, TB, cam, 'tb_hand_you', lerp(cx - 90 + near * 0.4, px, k) - ty[0] + br, lerp(cy - 160 + near, py, k) - ty[1] + iy); }
  else S(c, TB, cam, 'tb_hand_you', cx - 90 - ty[0] + br + near * 0.4, cy - 160 - ty[1] + iy + near + 4 * Math.sin(t * 0.9));
  S(c, TB, cam, 'tb_hand_me', cx + 90 - tm[0] - br - near * 0.4, cy + 160 - tm[1] + im - near - 4 * Math.sin(t * 0.8 + 1));
  nightLamp(c, cam, t, cx, cy);
  // two steams lean toward each other and wind together above the table
  inSet(c, TB, cam, 1, () => {
    const bend = (dir) => (u) => [dir * 260 * sstep(0.1, 0.8, u) + Math.sin(u * 7 + t) * 40 * u, 0];
    steam(c, t, me[0], me[1], { h: 520, w: 34, a: 0.28, sway: 30, n: 28, per: 0.26, seed: 22, bend: bend(-1) });
  });
  return { grade: { warm: 0.35, bloom: 1.0, sat: 1.05, vig: 0.6 } };
}
export function tableNight(c, t, lt, sh) { return nightTable(c, t, sh); }
// Ch2: and your phone goes face down
export function tableNight2(c, t, lt, sh) {
  const f0 = CT(27, 2), flip = ramp(t, f0, 0.55, 'io');
  return nightTable(c, t, sh, { reach: (tt) => { const [ox, oy] = MAN.tb_bg.centre || [1300, 864]; return [ramp(tt, f0 - 0.7, 0.6, 'io') * (1 - ramp(tt, f0 + 0.7, 0.9, 'io')), ox - 600, oy + 300]; }, phone: (cam, cx, cy) => {
    const name = flip < 0.5 ? 'tb_phone' : 'tb_phone_back', sx = Math.abs(Math.cos(flip * Math.PI));
    S(c, TB, cam, name, cx - 620, cy + 330 - 40 * Math.sin(flip * Math.PI), 1, { r: 0.12, sx: Math.max(0.02, sx) });
    if (flip < 0.5) inSet(c, TB, cam, 1, () => glow(c, cx - 620, cy + 330, 300, [170, 200, 255], 0.35 * (1 - flip * 2), { blend: 'screen' }));
  } });
}
// Ch3 「先把这一刻好好收着」: a square of morning sun lies between two mugs; our hands come into it; the pen closes it
export function keep(c, t, lt, sh) {
  const [ox, oy] = MAN.tb_bg.centre || [1300, 864];
  const cam = camK(t, [[sh.t0, ox, oy, 2100], [sh.t1, ox, oy, 1800]], { drift: 2 });
  const [cx, cy] = tableTop(c, cam, t, { sunSquare: [0.32, 0.22, 0.68, 0.78], sunA: 0.42 });
  S(c, TB, cam, 'tb_mug_you', cx - 600, cy - 300);
  S(c, TB, cam, 'tb_mug_me', cx + 620, cy + 260);
  const ty = MAN.tb_hand_you.tip || [0, 1150], tm = MAN.tb_hand_me.tip || [0, -1150];
  S(c, TB, cam, 'tb_hand_you', cx - 100 - ty[0], cy - 120 - ty[1] + lerp(-700, 0, ramp(t, CT(39, 1), 1.0, 'out')));
  S(c, TB, cam, 'tb_hand_me', cx + 100 - tm[0], cy + 120 - tm[1] + lerp(700, 0, ramp(t, CT(39, 3), 1.0, 'out')));
  inSet(c, TB, cam, 1, () => {
    const u = ramp(t, CT(39, 6), CT(39, 8) - CT(39, 6) + 0.5, 'io');
    pen(c, squarePts(cx, cy, 340, 0.03, 13), u, { color: 'rgba(214,150,30,0.9)', w: 8, seed: 14, tremor: 2.6 });
  });
  return { grade: { warm: 0.32, bloom: 0.8, sat: 1.06 } };
}
