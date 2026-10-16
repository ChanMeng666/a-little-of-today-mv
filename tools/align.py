"""Align the known lyric sheet to the ASR word timestamps (pinyin-aware global alignment).

usage: python tools/align.py <name> [model]
reads  films/<name>/lyrics.txt + work/<name>.asr.<model>.json
writes work/<name>.aligned.json  (per line: start/end + per-character times; unmatched characters interpolated)
"""
import json
import re
import sys
from pathlib import Path

from opencc import OpenCC
from pypinyin import lazy_pinyin

ROOT = Path(__file__).resolve().parent.parent
name = sys.argv[1]
model = sys.argv[2] if len(sys.argv) > 2 else "medium"
t2s = OpenCC("t2s")
CJK = re.compile(r"[一-鿿]")

# ---- lyric characters
lines, sec = [], None
for raw in (ROOT / "films" / name / "lyrics.txt").read_text(encoding="utf-8").splitlines():
    raw = raw.strip()
    if not raw:
        continue
    if raw.startswith("#"):
        sec = raw[1:].strip()
        continue
    who, text = raw.split("|", 1)
    lines.append({"sec": sec, "who": who, "text": text})
L = []  # (line index, char index in text, simplified char)
for li, ln in enumerate(lines):
    for ci, ch in enumerate(ln["text"]):
        if CJK.match(ch):
            L.append((li, ci, t2s.convert(ch)))

# ---- ASR characters with interpolated times
A = []  # (char, start, end)
for seg in json.loads((ROOT / "work" / f"{name}.asr.{model}.json").read_text(encoding="utf-8")):
    for w in seg["words"]:
        chars = [c for c in w["w"] if CJK.match(c)]
        if not chars:
            continue
        d = (w["e"] - w["s"]) / len(chars)
        for k, c in enumerate(chars):
            A.append((t2s.convert(c), w["s"] + k * d, w["s"] + (k + 1) * d))

py = lambda c: lazy_pinyin(c)[0]
lp, ap = [py(c) for _, _, c in L], [py(c) for c, _, _ in A]


def sim(i, j):
    if L[i][2] == A[j][0]:
        return 2.0
    a, b = lp[i], ap[j]
    if a == b:
        return 1.6
    # same final or same initial: weak evidence
    if a[-2:] == b[-2:] or a[:1] == b[:1]:
        return 0.2
    return -1.0


GAP = -0.6
n, m = len(L), len(A)
S = [[0.0] * (m + 1) for _ in range(n + 1)]
B = [[0] * (m + 1) for _ in range(n + 1)]
for i in range(1, n + 1):
    S[i][0] = i * GAP; B[i][0] = 1
for j in range(1, m + 1):
    S[0][j] = j * GAP; B[0][j] = 2
for i in range(1, n + 1):
    Si, Sp = S[i], S[i - 1]
    for j in range(1, m + 1):
        d = Sp[j - 1] + sim(i - 1, j - 1)
        u = Sp[j] + GAP
        l = Si[j - 1] + GAP
        if d >= u and d >= l:
            Si[j] = d; B[i][j] = 0
        elif u >= l:
            Si[j] = u; B[i][j] = 1
        else:
            Si[j] = l; B[i][j] = 2
match = [None] * n
i, j = n, m
while i > 0 or j > 0:
    b = B[i][j]
    if b == 0:
        if sim(i - 1, j - 1) > 0:
            match[i - 1] = j - 1
        i -= 1; j -= 1
    elif b == 1:
        i -= 1
    else:
        j -= 1

# ---- per-line timing
out = []
k = 0
for li, ln in enumerate(lines):
    idx = [q for q in range(n) if L[q][0] == li]
    ts = [(A[match[q]][1], A[match[q]][2]) if match[q] is not None else None for q in idx]
    known = [(p, t) for p, t in enumerate(ts) if t]
    rate = len(known) / max(1, len(idx))
    chars = []
    if known:
        # fill unmatched characters by linear interpolation / extrapolation at the line's mean character rate
        per = (known[-1][1][1] - known[0][1][0]) / max(1, known[-1][0] - known[0][0] + 1)
        per = min(max(per, 0.12), 0.6)
        for p in range(len(idx)):
            if ts[p]:
                chars.append(ts[p][0])
                continue
            prv = max([q for q, _ in known if q < p], default=None)
            nxt = min([q for q, _ in known if q > p], default=None)
            if prv is not None and nxt is not None:
                a, b = ts[prv][0], ts[nxt][0]
                chars.append(a + (b - a) * (p - prv) / (nxt - prv))
            elif prv is not None:
                chars.append(ts[prv][0] + per * (p - prv))
            else:
                chars.append(ts[nxt][0] - per * (nxt - p))
        start, end = chars[0], max(known[-1][1][1], chars[-1] + per)
    else:
        start = end = None
    out.append({**ln, "start": None if start is None else round(start, 2), "end": None if end is None else round(end, 2),
                "match": round(rate, 2), "ci": [L[q][1] for q in idx], "ct": [round(c, 2) for c in chars]})
    flag = "" if rate > 0.6 else "   <<< LOW"
    print(f"{'-' if start is None else f'{start:7.2f}'} {'-' if end is None else f'{end:7.2f}'} {rate:4.2f} {ln['who']} {ln['text']}{flag}")
(ROOT / "work" / f"{name}.aligned.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
used = {mm for mm in match if mm is not None}
# ASR runs the sheet does not explain (repeats, ad-libs)
run = []
for j in range(m):
    if j not in used:
        run.append(j)
    if (j in used or j == m - 1) and run:
        if len(run) >= 4:
            print(f"  unexplained ASR {A[run[0]][1]:7.2f}-{A[run[-1]][2]:7.2f}: {''.join(A[q][0] for q in run)}")
        run = []
