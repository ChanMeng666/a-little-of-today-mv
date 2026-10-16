// Shared kit for the code-drawn films (first written for 记忆琥珀2022). Everything here is a pure function of its arguments:
// no wall clock, no Math.random. A film page imports what it needs and calls boot().

// ---------------------------------------------------------------- math
export const clamp = (x, a = 0, b = 1) => (x < a ? a : x > b ? b : x);
export const lerp = (a, b, u) => a + (b - a) * u;
export const sstep = (a, b, x) => { const u = clamp((x - a) / (b - a)); return u * u * (3 - 2 * u); };
export const mix = (a, b, x) => clamp((x - a) / (b - a));           // linear 0..1 ramp between a and b
export const pulse = (x, a, b, fade = 0.5) => Math.min(sstep(a, a + fade, x), 1 - sstep(b - fade, b, x));   // in at a, out at b
export const mod = (a, n) => ((a % n) + n) % n;
export const TAU = Math.PI * 2;
export const E = {
  inQuad: (u) => u * u, outQuad: (u) => 1 - (1 - u) * (1 - u), inOutQuad: (u) => (u < 0.5 ? 2 * u * u : 1 - 2 * (1 - u) * (1 - u)),
  inCubic: (u) => u * u * u, outCubic: (u) => 1 - (1 - u) ** 3, inOutCubic: (u) => (u < 0.5 ? 4 * u ** 3 : 1 - 4 * (1 - u) ** 3),
  outExpo: (u) => (u >= 1 ? 1 : 1 - 2 ** (-10 * u)), outBack: (u) => 1 + 2.70158 * (u - 1) ** 3 + 1.70158 * (u - 1) ** 2,
  outElastic: (u) => (u <= 0 ? 0 : u >= 1 ? 1 : 2 ** (-10 * u) * Math.sin((u * 10 - 0.75) * (TAU / 3)) + 1),
};

// ---------------------------------------------------------------- seeded randomness
export const hash = (n) => { const s = Math.sin(n * 127.1 + 311.7) * 43758.5453123; return s - Math.floor(s); };
export const hash2 = (x, y) => { const s = Math.sin(x * 127.1 + y * 311.7 + 74.7) * 43758.5453123; return s - Math.floor(s); };
export function rng(seed) {        // mulberry32
  let a = (seed * 2654435761) >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export const noise1 = (x, seed = 0) => { const i = Math.floor(x), f = x - i, u = f * f * (3 - 2 * f); return lerp(hash(i + seed * 57.3), hash(i + 1 + seed * 57.3), u) * 2 - 1; };
export function noise2(x, y, seed = 0) {
  const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi, u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf), s = seed * 13.7;
  return lerp(lerp(hash2(xi + s, yi), hash2(xi + 1 + s, yi), u), lerp(hash2(xi + s, yi + 1), hash2(xi + 1 + s, yi + 1), u), v) * 2 - 1;
}
export const fbm1 = (x, seed = 0, oct = 3) => { let a = 0, w = 0.5, f = 1; for (let i = 0; i < oct; i++) { a += noise1(x * f, seed + i) * w; w *= 0.5; f *= 2; } return a; };

// ---------------------------------------------------------------- timing
// timing.json (tools/build_timing.py): { duration, tempo, beats[], onsets[[t,v]], env_dt, env{all,low,mid,high}, lines[{i,sec,who,text,t0,t1,ct[]}] }
export async function loadTiming(url = './timing.json') {
  const TM = await (await fetch(url)).json();
  const { beats, lines } = TM;
  // sections: first/last sung moment of each "# name" block in lyrics.txt, in song order
  TM.sec = {}; TM.secs = [];
  for (const l of lines) {
    let s = TM.sec[l.sec];
    if (!s) { s = TM.sec[l.sec] = { name: l.sec, t0: l.t0, t1: l.t1, lines: [] }; TM.secs.push(s); }
    s.t1 = Math.max(s.t1, l.t1); s.lines.push(l);
  }
  TM.line = (i) => lines[i];
  TM.lineAt = (t, tail = 0) => { for (const l of lines) if (t >= l.t0 && t < l.t1 + tail) return l; return null; };
  TM.lastLine = (t) => { let r = null; for (const l of lines) if (l.t0 <= t) r = l; return r; };
  TM.secAt = (t) => { let r = null; for (const s of TM.secs) if (s.t0 <= t) r = s; return r; };
  // beat index (fractional): integer part = beats passed, fraction = phase inside the current beat
  TM.beat = (t) => {
    if (!beats.length) return 0;
    if (t <= beats[0]) return (t - beats[0]) / (beats[1] - beats[0]);
    let lo = 0, hi = beats.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (beats[m] <= t) lo = m; else hi = m; }
    const d = (beats[lo + 1] ?? beats[lo] + 60 / TM.tempo) - beats[lo];
    return lo + (t - beats[lo]) / d;
  };
  TM.kick = (t, decay = 6) => { const b = TM.beat(t); return b < 0 ? 0 : Math.exp(-(b - Math.floor(b)) * decay); };   // 1 on each beat, decaying
  TM.env_ = TM.env; TM.env = (k, t) => { const e = TM.env_[k]; const x = clamp(t / TM.env_dt, 0, e.length - 1.001), i = Math.floor(x); return lerp(e[i], e[i + 1], x - i); };
  // strongest onset within [t-win, t], as a decaying accent
  TM.hit = (t, decay = 8, min = 0.5) => { let a = 0; for (const [ot, v] of TM.onsets) { if (ot > t) break; if (t - ot < 1 && v >= min) a = Math.max(a, Math.min(v, 1.5) * Math.exp(-(t - ot) * decay)); } return a; };
  // how many characters of line l have been sung at t (fractional)
  TM.sung = (l, t) => { const c = l.ct; let n = 0; while (n < c.length && c[n] <= t) n++; return n; };
  return TM;
}

// ---------------------------------------------------------------- images
export const loadImage = (url) => new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = () => rej(new Error('image ' + url)); im.src = url; });
export async function loadImages(map) { const out = {}; await Promise.all(Object.entries(map).map(async ([k, u]) => { out[k] = await loadImage(u); })); return out; }
export function makeCanvas(w, h) { const c = document.createElement('canvas'); c.width = w; c.height = h; return [c, c.getContext('2d')]; }

// Tileable monochrome grain, built once from a seed. Draw with grain(ctx, tile, frameIndex, alpha).
export function grainTile(size = 256, seed = 7, contrast = 1) {
  const [c, x] = makeCanvas(size, size), im = x.createImageData(size, size), r = rng(seed);
  for (let i = 0; i < size * size; i++) { const v = 128 + (r() - 0.5) * 255 * contrast; im.data[i * 4] = im.data[i * 4 + 1] = im.data[i * 4 + 2] = v; im.data[i * 4 + 3] = 255; }
  x.putImageData(im, 0, 0); return c;
}
export function grain(ctx, tile, frame, alpha = 0.08, mode = 'overlay') {
  const W = ctx.canvas.width, H = ctx.canvas.height, s = tile.width, ox = Math.floor(hash(frame) * s), oy = Math.floor(hash(frame + 91) * s);
  ctx.save(); ctx.globalAlpha = alpha; ctx.globalCompositeOperation = mode;
  for (let y = -oy; y < H; y += s) for (let x = -ox; x < W; x += s) ctx.drawImage(tile, x, y);
  ctx.restore();
}
export function vignette(ctx, strength = 0.5, color = '0,0,0') {
  const W = ctx.canvas.width, H = ctx.canvas.height, g = ctx.createRadialGradient(W / 2, H / 2, H * 0.35, W / 2, H / 2, H * 1.05);
  g.addColorStop(0, `rgba(${color},0)`); g.addColorStop(1, `rgba(${color},${strength})`);
  ctx.save(); ctx.fillStyle = g; ctx.fillRect(0, 0, W, H); ctx.restore();
}

// ---------------------------------------------------------------- lyrics
// One subtitle line near the bottom, revealed character by character as it is sung, fading out after the line ends.
// opts: font, size, y, color, dim (unsung colour or null to hide unsung), stroke, strokeW, shadow, tracking, tail, lead, spoken(l)→style overrides
export function subtitle(ctx, TM, t, o = {}) {
  const lead = o.lead ?? 0.35, tail = o.tail ?? 0.6;
  let l = null;
  for (const x of TM.lines) if (t >= x.t0 - lead && t < x.t1 + tail) l = x;      // the latest line wins while two overlap
  if (!l) return null;
  const st = { font: '"Noto Serif SC"', weight: 500, size: 46, y: ctx.canvas.height - 92, color: '#fff', dim: null, stroke: 'rgba(0,0,0,0.55)', strokeW: 5, shadow: null, tracking: 0.06, ...o, ...(o.style ? o.style(l) : {}) };
  const a = Math.min(sstep(l.t0 - lead, l.t0 - lead + 0.3, t), 1 - sstep(l.t1 + tail - 0.45, l.t1 + tail, t));
  const chars = [...l.text], W = ctx.canvas.width;
  ctx.save();
  ctx.font = `${st.weight} ${st.size}px ${st.font}`; ctx.textBaseline = 'middle'; ctx.textAlign = 'left'; ctx.lineJoin = 'round';
  const adv = chars.map((c) => (c === ' ' ? st.size * 0.5 : ctx.measureText(c).width + st.size * st.tracking));
  let x = (st.x ?? W / 2) - adv.reduce((s, v) => s + v, 0) / 2;
  for (let k = 0; k < chars.length; k++) {
    const c = chars[k], u = sstep(l.ct[k] - 0.12, l.ct[k] + 0.18, t);
    if (c !== ' ') {
      const al = a * (st.dim ? 1 : u);
      if (al > 0.003) {
        ctx.globalAlpha = al;
        if (st.shadow) { ctx.shadowColor = st.shadow; ctx.shadowBlur = st.size * 0.35; }
        if (st.stroke && st.strokeW) { ctx.strokeStyle = st.stroke; ctx.lineWidth = st.strokeW; ctx.strokeText(c, x, st.y + (st.rise ?? 6) * (1 - u)); }
        ctx.fillStyle = st.dim && u < 0.5 ? st.dim : st.color;
        ctx.fillText(c, x, st.y + (st.dim ? 0 : (st.rise ?? 6) * (1 - u)));
        ctx.shadowBlur = 0;
      }
    }
    x += adv[k];
  }
  ctx.restore();
  return l;
}

// SRT text for a timing object (tools read this through the page: window.srt()).
export function srt(TM) {
  const f = (t) => { const ms = Math.round(t * 1000), h = Math.floor(ms / 3600000), m = Math.floor(ms / 60000) % 60, s = Math.floor(ms / 1000) % 60; return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')},${String(ms % 1000).padStart(3, '0')}`; };
  return TM.lines.map((l, i) => `${i + 1}\n${f(l.t0)} --> ${f(l.t1)}\n${l.text}\n`).join('\n');
}

// ---------------------------------------------------------------- scenes
// scenes: [{ t0, t1, draw(ctx, t, u, k) }] — u is 0..1 across the scene, k its index. Scenes may overlap; later ones draw on top.
// A scene with `fade` (seconds) is drawn through an offscreen buffer and cross-faded in over that time.
export function runScenes(ctx, scenes, t, buf) {
  for (let k = 0; k < scenes.length; k++) {
    const s = scenes[k];
    if (t < s.t0 || t >= s.t1) continue;
    const u = (t - s.t0) / (s.t1 - s.t0), a = s.fade ? sstep(s.t0, s.t0 + s.fade, t) : 1;
    if (a >= 0.999 || !buf) { s.draw(ctx, t, u, k); continue; }
    const bx = buf.getContext('2d'); bx.setTransform(1, 0, 0, 1, 0, 0); bx.clearRect(0, 0, buf.width, buf.height);
    s.draw(bx, t, u, k);
    ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalAlpha = a; ctx.drawImage(buf, 0, 0); ctx.restore();
  }
}

// ---------------------------------------------------------------- boot
// boot({ width, height, fps, duration, render(t) }) wires the page to harness/render.mjs and harness/preview.html.
export async function boot({ width = 1920, height = 1080, fps = 24, duration, render }) {
  window.DEMO = { width, height, fps, duration };
  window.renderAt = async (t) => { await render(clamp(t, 0, duration)); };
  if (document.fonts) { await Promise.all([...document.fonts].map((f) => f.load().catch(() => null))); await document.fonts.ready; }
  await window.renderAt(+(new URLSearchParams(location.search).get('t') || 0));
  window.__ready = true;
}
