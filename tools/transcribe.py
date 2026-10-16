"""Word-level ASR of a song with faster-whisper → work/<name>.asr.<model>.json

usage: python tools/transcribe.py <name> [model]
"""
import json
import sys
from pathlib import Path

from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parent.parent
name = sys.argv[1]
model_name = sys.argv[2] if len(sys.argv) > 2 else "medium"

model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=8)
segments, info = model.transcribe(
    str(ROOT / "work" / f"{name}.16k.wav"),
    language="zh",
    word_timestamps=True,
    vad_filter=False,
    beam_size=5,
    condition_on_previous_text=False,
    initial_prompt="以下是一首简体中文歌曲的歌词。",
)
out = []
for s in segments:
    out.append({
        "start": round(s.start, 3), "end": round(s.end, 3), "text": s.text,
        "words": [{"w": w.word, "s": round(w.start, 3), "e": round(w.end, 3), "p": round(w.probability, 3)} for w in (s.words or [])],
    })
    print(f"{s.start:7.2f} {s.end:7.2f} {s.text}", flush=True)
(ROOT / "work" / f"{name}.asr.{model_name.replace('/', '_')}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
