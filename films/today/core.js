// What every shot needs: the song's timing, the art, and a few time helpers. Loaded once, before the first frame.
import { loadTiming } from '/lib/mv.js';
import { loadArt, MAN, IM, clamp, lerp, sstep, E } from './stage.js';

export const TM = await loadTiming(new URLSearchParams(location.search).get('timing') || './timing.json');
await loadArt();
export const DUR = TM.duration, FPS = 24;
export const B = TM.beats;
// lyric line i: start, end, the moment character k is sung
export const LN = (i) => TM.lines[i];
export const T0 = (i) => TM.lines[i].t0;
export const CT = (i, k) => TM.lines[i].ct[k];
// the beat nearest to t, and the n-th beat after/before it
export function beatNear(t, n = 0) {
  let lo = 0, hi = B.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (B[m] <= t) lo = m; else hi = m; }
  const i = Math.abs(B[hi] - t) < Math.abs(t - B[lo]) ? hi : lo;
  return B[clamp(i + n, 0, B.length - 1)];
}
export const beatIdx = (t) => Math.floor(TM.beat(t));
// decaying accent since the last beat that satisfies (index % every === phase)
export function kickN(t, every = 1, phase = 0, decay = 5) {
  const bi = beatIdx(t); if (bi < 0) return 0;
  const k = bi - ((((bi - phase) % every) + every) % every);
  return k < 0 ? 0 : Math.exp(-(t - B[k]) * decay);
}
// seconds since the latest of the given event times (Infinity before the first)
export const since = (t, times) => { let r = Infinity; for (const x of times) if (x <= t) r = t - x; return r; };
// 0→1 ramp starting at a lasting d, eased
export const ramp = (t, a, d, ez = 'io') => { const u = clamp((t - a) / d); return ez === 'out' ? E.outCubic(u) : ez === 'in' ? E.inCubic(u) : ez === 'back' ? E.outBack(u) : ez === 'lin' ? u : u * u * (3 - 2 * u); };
export { MAN, IM };
