#!/bin/sh
# Rebuild timing.json for the given songs (default: all). ASR is the slow step and is skipped when its output exists.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
for f in ${@:-$(ls songs/*.mp3)}; do
  n=$(basename "$f" .mp3)
  [ -f work/$n.16k.wav ] || ffmpeg -hide_banner -loglevel error -y -i "$f" -vn -ac 1 -ar 16000 work/$n.16k.wav
  [ -f work/$n.48k.wav ] || ffmpeg -hide_banner -loglevel error -y -i "$f" -vn -ar 48000 work/$n.48k.wav
  [ -f work/$n.asr.large-v3.json ] || python tools/transcribe.py $n large-v3 > work/$n.asr.log 2>&1
  python tools/align.py $n large-v3 > work/$n.align.log 2>&1
  python tools/analyze.py $n > work/$n.music.log 2>&1
  python tools/build_timing.py $n > work/$n.timing.log 2>&1
  echo "done $n"
done
