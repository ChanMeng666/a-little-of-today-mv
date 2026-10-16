"""Music analysis of a song → work/<name>.music.json : tempo, beats, downbeat guess, RMS envelope (10 Hz),
low/mid/high band envelopes, onset strength peaks, integrated loudness.

usage: python tools/analyze.py <name>
"""
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
name = sys.argv[1]
y, sr = librosa.load(ROOT / "songs" / f"{name}.mp3", sr=22050, mono=True)
dur = len(y) / sr
hop = 512
oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
tempo, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=hop, start_bpm=72, tightness=120)
bt = librosa.frames_to_time(beats, sr=sr, hop_length=hop)

# envelopes at 20 Hz
S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop)) ** 2
f = librosa.fft_frequencies(sr=sr, n_fft=2048)
ft = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)
grid = np.arange(0, dur, 0.05)


def band(lo, hi):
    e = np.sqrt(S[(f >= lo) & (f < hi)].sum(axis=0))
    e = np.interp(grid, ft, e)
    k = np.hanning(9); k /= k.sum()
    e = np.convolve(e, k, mode="same")
    return (e / (np.percentile(e, 99) + 1e-9)).clip(0, 1.5)


env = {"all": band(20, 11000), "low": band(20, 200), "mid": band(200, 2000), "high": band(4000, 11000)}
# strong onsets (piano / guitar hits) for accents
on = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=hop, backtrack=False, delta=0.25, wait=4)
ont = librosa.frames_to_time(on, sr=sr, hop_length=hop)
onv = oenv[on] / (np.percentile(oenv, 99) + 1e-9)

yy, sr2 = sf.read(ROOT / "work" / f"{name}.48k.wav") if (ROOT / "work" / f"{name}.48k.wav").exists() else (None, None)
lufs = float(pyln.Meter(sr2).integrated_loudness(yy)) if yy is not None else None
out = {
    "duration": round(dur, 3), "tempo": float(np.atleast_1d(tempo)[0]), "lufs": lufs,
    "beats": [round(float(b), 3) for b in bt],
    "onsets": [[round(float(t), 3), round(float(v), 2)] for t, v in zip(ont, onv)],
    "env_dt": 0.05,
    "env": {k: [round(float(v), 3) for v in e] for k, e in env.items()},
}
(ROOT / "work" / f"{name}.music.json").write_text(json.dumps(out), encoding="utf-8")
print(name, "dur", out["duration"], "tempo", out["tempo"], "beats", len(bt), "lufs", lufs)
# coarse energy map, 5 s per cell
a = env["all"]
for i in range(0, len(a), 100):
    seg = a[i:i + 100]
    print(f"{i * 0.05:6.1f}s  all {seg.mean():.2f}  low {env['low'][i:i + 100].mean():.2f}  high {env['high'][i:i + 100].mean():.2f}  " + "#" * int(seg.mean() * 40))
