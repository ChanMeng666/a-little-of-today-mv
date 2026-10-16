"""One still per shot (at a fraction of the way through each), tiled and labelled with the shot id.

usage: python tools/shotsheet.py <film-dir> <out.jpg> [fraction=0.5] [cols=6] [id,id,...]
Reads the edit list from the page (window.EDIT) through harness/render.mjs --stills.
"""
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
film, out = Path(sys.argv[1]), sys.argv[2]
frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
cols = int(sys.argv[4]) if len(sys.argv) > 4 else 6
only = sys.argv[5].split(",") if len(sys.argv) > 5 else None
probe = subprocess.run(["node", str(ROOT / "tools" / "edit_list.mjs"), str(film)], capture_output=True, text=True, encoding="utf-8", check=True)
edit = json.loads(probe.stdout.strip().splitlines()[-1])
if only:
    edit = [s for s in edit if s["id"] in only]
ts = [round(s["t0"] + (s["t1"] - s["t0"]) * frac, 2) for s in edit]
sd = film / "out" / "stills"
for f in sd.glob("t*.png"):
    f.unlink()
subprocess.run(["node", str(ROOT / "harness" / "render.mjs"), str(film), "--stills", ",".join(map(str, ts)), "--scale", "0.5"], check=True, stdout=subprocess.DEVNULL)
cw, ch = 480, 270
rows = (len(ts) + cols - 1) // cols
sheet = Image.new("RGB", (cols * (cw + 4) + 4, rows * (ch + 4) + 4), (60, 60, 60))
for i, (s, t) in enumerate(zip(edit, ts)):
    im = Image.open(sd / f"t{t:05.2f}.png").convert("RGB").resize((cw, ch), Image.LANCZOS)
    ImageDraw.Draw(im).text((6, 4), f"{s['id']} {t:.1f}", fill=(255, 60, 60))
    sheet.paste(im, (4 + (i % cols) * (cw + 4), 4 + (i // cols) * (ch + 4)))
sheet.save(out, quality=88)
print(out)
