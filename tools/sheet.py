"""Contact sheet of stills: python tools/sheet.py <film-dir> <cols> <t1,t2,...> <out.jpg> [cell width] [extra render args...]"""
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

film, cols, ts, out = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
cw = int(sys.argv[5]) if len(sys.argv) > 5 else 640
sd = film / "out" / "stills"
shutil.rmtree(sd, ignore_errors=True)
subprocess.run(["node", "harness/render.mjs", str(film), "--stills", ts, "--scale", "0.5", *sys.argv[6:]], check=True, stdout=subprocess.DEVNULL)
files = sorted(sd.glob("t*.png"), key=lambda p: float(p.stem[1:]))
ims = [Image.open(f).convert("RGB") for f in files]
ch = round(cw * ims[0].height / ims[0].width)
rows = (len(ims) + cols - 1) // cols
sheet = Image.new("RGB", (cols * (cw + 6) + 6, rows * (ch + 6) + 6), (70, 70, 70))
for i, (f, im) in enumerate(zip(files, ims)):
    im = im.resize((cw, ch), Image.LANCZOS)
    ImageDraw.Draw(im).text((8, 6), f.stem[1:], fill=(255, 80, 80))
    sheet.paste(im, (6 + (i % cols) * (cw + 6), 6 + (i // cols) * (ch + 6)))
sheet.save(out, quality=88)
print(out)
