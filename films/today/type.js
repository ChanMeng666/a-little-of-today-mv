// Lettering: the lyric written in ink on a strip of washi tape stuck across the bottom of the frame, and the title.
import { W, H, MAN, IM, PAL, rgba, clamp, lerp, sstep, E, hash, noise1 } from './stage.js';
import { TM, DUR, T0 } from './core.js';

const LEAD = 0.42, TAIL = 0.55;
// the tape in three slices: its torn ends keep their shape, only the middle stretches, so the printed squares stay square
function tape3(c, im, x, y, w, h) {
  const iw = im.width, ih = im.height, e = Math.min(220, iw / 3), ed = e * h / ih, mid = Math.max(1, w - 2 * ed);
  c.drawImage(im, 0, 0, e, ih, x, y, ed, h);
  // tile the middle at its natural aspect instead of stretching it
  const tw = (iw - 2 * e) * h / ih; let xx = x + ed;
  while (xx < x + ed + mid - 0.5) { const ww = Math.min(tw, x + ed + mid - xx); c.drawImage(im, e, 0, ww * ih / h, ih, xx, y, ww, h); xx += ww; }
  c.drawImage(im, iw - e, 0, e, ih, x + ed + mid, y, ed, h);
}
const CHORUS = new Set(['ch1', 'ch2', 'ch3']);
// o (from the shot): { y } lifts the line off the bottom; { hide } leaves it out (a shot that letters the line itself)
export function drawLyric(c, t, o = {}) {
  if (o.hide) return;
  let l = null;
  for (const x of TM.lines) if (t >= x.t0 - LEAD && t < x.t1 + TAIL) l = x;
  if (!l) return;
  const nxt = TM.lines[l.i + 1], end = Math.min(l.t1 + TAIL, nxt ? Math.max(nxt.t0 - LEAD, l.t1 - 0.05) : 1e9);
  const a = Math.min(sstep(l.t0 - LEAD, l.t0 - LEAD + 0.3, t), 1 - sstep(end - 0.28, end, t));
  if (a <= 0.003) return;
  const size = 50, y = o.y ?? H - 98, chars = [...l.text], tape = CHORUS.has(l.sec) ? 'ly_tape_mint' : 'ly_tape';
  c.save();
  c.font = `400 ${size}px "LXGW WenKai"`; c.textBaseline = 'middle'; c.textAlign = 'left';
  const adv = chars.map((ch) => (ch === ' ' ? size * 0.55 : c.measureText(ch).width + size * 0.07));
  const tw = adv.reduce((s, v) => s + v, 0), bw = tw + 150, bh = 104;
  // each strip is torn off a little differently and stuck down a little crooked; it is pressed on, not faded on
  const rot = (hash(l.i * 3.7) - 0.5) * 0.035, dy = (hash(l.i * 5.1) - 0.5) * 8;
  const press = sstep(l.t0 - LEAD, l.t0 - LEAD + 0.22, t), lift = 1 - press;
  c.translate(W / 2, y + dy); c.rotate(rot + lift * 0.04);
  c.scale(1 + 0.04 * lift, 1 + 0.04 * lift);
  // the tape's own shadow, then the tape
  if (IM[tape]) {
    c.globalAlpha = a * 0.22; c.filter = 'blur(6px)'; tape3(c, IM[tape], -bw / 2 + 4, -bh / 2 + 7 + 6 * lift, bw, bh); c.filter = 'none';
    c.globalAlpha = a * 0.94; tape3(c, IM[tape], -bw / 2, -bh / 2, bw, bh);
  } else { c.globalAlpha = a * 0.85; c.fillStyle = 'rgb(246,236,200)'; c.fillRect(-bw / 2, -bh / 2 + 14, bw, bh - 28); }
  let x = -tw / 2;
  for (let k = 0; k < chars.length; k++) {
    if (chars[k] !== ' ') {
      const u = sstep(l.ct[k] - 0.1, l.ct[k] + 0.16, t);
      c.globalAlpha = a * lerp(0.30, 0.95, u);
      c.fillStyle = rgba([46, 38, 50]);
      c.save(); c.translate(x + adv[k] / 2, 3 + (hash(l.i * 13 + k) - 0.5) * 3 - 3 * Math.sin(clamp((t - l.ct[k] + 0.1) / 0.4) * Math.PI)); c.rotate((hash(l.i * 7 + k * 1.3) - 0.5) * 0.05);
      c.fillText(chars[k], -adv[k] / 2 + size * 0.035, 0); c.restore();
    }
    x += adv[k];
  }
  c.restore();
}

export function drawTitles(c, t) {
  // opening: the name of the song, written on the fridge door above the first note
  const a = Math.min(sstep(0.6, 1.6, t), 1 - sstep(5.4, 6.4, t));
  if (a <= 0.003) return;
  c.save(); c.textBaseline = 'middle'; c.textAlign = 'center';
  const chars = [...'把今天留一小块给我'], size = 70, gap = 12;
  c.font = `600 ${size}px "Noto Serif SC"`;
  let x = W / 2 - (chars.length * (size + gap) - gap) / 2 + size / 2;
  for (let k = 0; k < chars.length; k++) {
    const u = sstep(0.6 + k * 0.11, 1.3 + k * 0.11, t);
    c.globalAlpha = a * u;
    c.fillStyle = k >= 4 && k <= 6 ? rgba([214, 150, 30]) : rgba([52, 44, 54]);
    c.fillText(chars[k], x, 92 + 10 * (1 - u)); x += size + gap;
  }
  c.globalAlpha = a * sstep(1.9, 2.8, t);
  c.font = 'italic 500 38px "Cormorant Garamond"'; c.fillStyle = rgba([70, 62, 72]);
  c.fillText('Save Me a Little of Today', W / 2, 158);
  c.font = '400 26px "Noto Serif SC"'; c.fillStyle = rgba([90, 80, 88]);
  c.fillText('词 曲   Chan Meng', W / 2, 202);
  c.restore();
}

// the end card, lettered on the door below the last note once the camera has come to rest on it
export function drawCard(c, t, a) {
  c.save(); c.globalAlpha = a; c.textBaseline = 'middle'; c.textAlign = 'center';
  c.font = '600 64px "Noto Serif SC"'; c.fillStyle = rgba([52, 44, 54]);
  const chars = [...'把今天留一小块给我'], size = 64, gap = 12; let x = W / 2 - (chars.length * (size + gap) - gap) / 2 + size / 2;
  for (let k = 0; k < chars.length; k++) { c.fillStyle = k >= 4 && k <= 6 ? rgba([214, 150, 30]) : rgba([52, 44, 54]); c.fillText(chars[k], x, H - 210); x += size + gap; }
  c.font = 'italic 500 38px "Cormorant Garamond"'; c.fillStyle = rgba([80, 70, 80]); c.fillText('Save Me a Little of Today', W / 2, H - 142);
  c.font = '400 26px "Noto Serif SC"'; c.fillStyle = rgba([100, 90, 96]); c.fillText('词 曲   Chan Meng        every frame drawn in code', W / 2, H - 92);
  c.restore();
}
