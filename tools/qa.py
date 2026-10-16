"""Motion QA of a rendered film: frozen stretches and jumps that are not cuts.

usage: python tools/qa.py <video.mp4> <film-dir>
Decodes the video small and grey, measures the mean absolute difference between neighbouring frames, and reports
  - frozen: runs longer than 0.75 s where nothing visibly moves (diff below 0.25 grey levels),
  - jumps: single-frame changes far above the local level that do not sit on a shot boundary (window.EDIT).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
video, film = sys.argv[1], sys.argv[2]
W, H, FPS = 192, 108, 24
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vf", f"scale={W}:{H},format=gray", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
fr = np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.float32)
d = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))
edit = json.loads(subprocess.run(["node", str(ROOT / "tools" / "edit_list.mjs"), film], capture_output=True, text=True, encoding="utf-8", check=True).stdout.strip().splitlines()[-1])
cuts = [s["t0"] for s in edit]
near_cut = lambda t: any(abs(t - c) < 1.6 for c in cuts)
print(f"{len(fr)} frames, mean diff {d.mean():.2f}, median {np.median(d):.2f}")
run = 0
for i, v in enumerate(d):
    if v < 0.25:
        run += 1
    else:
        if run > 0.75 * FPS:
            print(f"frozen {(i - run) / FPS:7.2f}-{i / FPS:7.2f}s ({run} frames)")
        run = 0
for i in range(5, len(d) - 5):
    loc = np.median(np.r_[d[i - 5:i], d[i + 1:i + 6]])
    if d[i] > max(6.0, 5 * loc) and not near_cut((i + 1) / FPS):
        print(f"jump at {(i + 1) / FPS:7.2f}s: diff {d[i]:.1f} vs local {loc:.1f}")
# per shot: how much moves
for s in edit:
    a, b = int(s["t0"] * FPS), max(int(s["t0"] * FPS) + 1, int(s["t1"] * FPS) - 1)
    print(f"  {s['id']:12s} {s['t0']:7.2f}  motion {d[a:b].mean():5.2f}  min {d[a:b].min():5.2f}")
