// The edit decision list: when each shot starts and how the previous one turns into it.
// Transition types (post.js): 0 cut, 1 dissolve, 2 bloom from a point, 3 brush across, 4 iris open, 5 run,
// 6 iris close, 7 peel off like a sticky note (ctr = the corner it is pulled from), 8 a small square opens out,
// 9 the old picture closes into a small square (a lit window).
// Every line of the song is a shot boundary or sits inside one; the cuts come from timing.json, never typed seconds,
// except where there is no lyric to hang them on.
import { T0, CT, beatNear as b } from './core.js';
import * as HM from './shots_home.js';
import * as SL from './shots_sill.js';
import * as OUT from './shots_out.js';
import * as TW from './shots_town.js';

export const EDIT = [
  // verse 1: morning, the note, the half-finished sentence
  { id: 'note', t0: 0, shot: HM.note },
  { id: 'kitchen', t0: T0(2) - 0.35, shot: HM.kitchen, tr: { type: 7, dur: 1.1, ctr: [1, 1] } },
  { id: 'tableAM', t0: T0(3) - 0.1, shot: HM.tableAM, tr: { type: 1, dur: 0.5 } },
  // pre-chorus 1: the mint, the days going by, slow down
  { id: 'mint', t0: T0(4) - 0.35, shot: SL.mint, tr: { type: 2, dur: 1.0, ctr: [0.5, 0.55] } },
  { id: 'rush', t0: T0(6) - 0.1, shot: HM.rush, tr: { type: 2, dur: 0.8, ctr: [0.3, 0.7] } },
  { id: 'slow', t0: T0(7) - 0.12, shot: SL.slow, tr: { type: 1, dur: 0.9 } },
  // chorus 1: night, two windows; the lane
  { id: 'windows', t0: T0(8) - 0.5, shot: OUT.windows, tr: { type: 5, dur: 1.3 } },
  { id: 'tired', t0: T0(10) - 0.12, shot: OUT.tired, tr: { type: 8, dur: 0.75, ctr: [0.5, 0.5] } },
  { id: 'tableNight', t0: T0(11) - 0.1, shot: HM.tableNight, tr: { type: 1, dur: 0.6 } },
  { id: 'lane', t0: T0(12) - 0.15, shot: OUT.lane, tr: { type: 2, dur: 0.9, ctr: [0.5, 0.6] } },
  { id: 'walk', t0: T0(14) - 0.1, shot: OUT.walk },
  { id: 'rise', t0: T0(15) - 0.1, shot: OUT.rise, tr: { type: 1, dur: 0.5 } },
  // verse 2: the umbrella in the entry
  { id: 'entry', t0: T0(16) - 0.3, shot: SL.entry, tr: { type: 7, dur: 1.0, ctr: [0, 1] } },
  { id: 'bent', t0: T0(17) - 0.1, shot: SL.bent },
  { id: 'fix', t0: T0(18) - 0.1, shot: SL.fix, tr: { type: 1, dur: 0.35 } },
  { id: 'far', t0: T0(19) - 0.1, shot: OUT.far, tr: { type: 1, dur: 0.9 } },
  // pre-chorus 2: the breakfast stall downstairs; remembering
  { id: 'stall', t0: T0(20) - 0.15, shot: TW.stall, tr: { type: 2, dur: 0.9, ctr: [0.5, 0.6] } },
  { id: 'soy', t0: T0(21) - 0.1, shot: TW.soy },
  { id: 'stall2', t0: T0(22) - 0.1, shot: TW.stall2, tr: { type: 1, dur: 0.4 } },
  { id: 'notes', t0: T0(23) - 0.1, shot: HM.notes, tr: { type: 8, dur: 0.7, ctr: [0.5, 0.45] } },
  // chorus 2: wider, later at night, rain
  { id: 'windows2', t0: T0(24) - 0.9, shot: OUT.windows2, tr: { type: 5, dur: 1.2 } },
  { id: 'noteWin', t0: T0(25) - 0.1, shot: OUT.noteWin, tr: { type: 8, dur: 0.7, ctr: [0.5, 0.5] } },
  { id: 'tired2', t0: T0(26) - 0.1, shot: OUT.tired2, tr: { type: 1, dur: 0.5 } },
  { id: 'tableNight2', t0: T0(27) - 0.1, shot: HM.tableNight2, tr: { type: 1, dur: 0.6 } },
  { id: 'laneNight', t0: T0(28) - 0.15, shot: OUT.laneNight, tr: { type: 2, dur: 0.9, ctr: [0.5, 0.6] } },
  { id: 'umbrella', t0: T0(30) - 0.1, shot: OUT.umbrella },
  { id: 'rise2', t0: T0(31) - 0.1, shot: OUT.rise2, tr: { type: 1, dur: 0.5 } },
  // bridge: the river, the jacket, the osmanthus shop
  { id: 'step', t0: T0(32) - 0.7, shot: TW.step, tr: { type: 1, dur: 1.2 } },
  { id: 'coat', t0: T0(33) - 0.1, shot: TW.coat },
  { id: 'shop', t0: T0(34) - 0.1, shot: TW.shop, tr: { type: 2, dur: 0.9, ctr: [0.5, 0.6] } },
  { id: 'osmanthus', t0: T0(35) - 0.3, shot: TW.osmanthus, tr: { type: 1, dur: 0.5 } },
  // final chorus: tomorrow; the lane in daylight, past the bend
  { id: 'dawn', t0: T0(36) - 0.25, shot: OUT.dawn, tr: { type: 1, dur: 1.1 } },
  { id: 'tomorrow', t0: T0(37) - 0.15, shot: HM.tomorrow, tr: { type: 7, dur: 0.9, ctr: [1, 0] } },
  { id: 'mint2', t0: T0(38) - 0.1, shot: SL.mint2, tr: { type: 2, dur: 0.8, ctr: [0.5, 0.55] } },
  { id: 'keep', t0: T0(39) - 0.1, shot: HM.keep, tr: { type: 1, dur: 0.5 } },
  { id: 'laneDay', t0: T0(40) - 0.2, shot: OUT.laneDay, tr: { type: 8, dur: 0.6, ctr: [0.5, 0.5] } },
  { id: 'walkDay', t0: T0(42) - 0.1, shot: OUT.walkDay },
  { id: 'notesSky', t0: T0(43) - 0.1, shot: OUT.notesSky, tr: { type: 1, dur: 0.4 } },
  // outro
  { id: 'home', t0: T0(44) - 0.15, shot: SL.home, tr: { type: 1, dur: 0.9 } },
  { id: 'mintEnd', t0: CT(44, 6) - 0.15, shot: SL.mintEnd, tr: { type: 1, dur: 0.5 } },
  { id: 'end', t0: T0(45) - 0.15, shot: HM.end, tr: { type: 7, dur: 1.0, ctr: [1, 1] } },
];
