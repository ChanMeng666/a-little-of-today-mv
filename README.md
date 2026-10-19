<div align="center">

# Save Me a Little of Today · Music Video

**A three-minute music video drawn entirely in code: Python paints procedural watercolour sets, a web page animates them with rain, handwriting and sticky-note transitions, headless Chrome renders every frame.**

<img src="films/today/poster.jpg" width="860" alt="Three sticky notes on a fridge door: 有空吗？, 牛奶 别忘了问, 明天">

</div>

## The film

### Save Me a Little of Today · 把今天留一小块给我



https://github.com/user-attachments/assets/e5ed1d73-ae7d-4da9-afda-d62076932a7b



▶ Watch on YouTube: https://youtu.be/fgIGrYIczxc

## How it's made

- **Paint.** `films/today/paint/wc.py` (numpy + OpenCV) lays subtractive watercolour washes and pen lines; each `s_*.py` paints one set, several laid out through a real perspective camera in centimetres, and saves depth layers as WebP (82 in all).
- **Animate.** The page exposes `window.renderAt(t)`: parallax camera, light, rain, steam, handwriting written character by character, window light projected onto walls (Canvas 2D).
- **Transitions in WebGL2.** Rain on glass as a height field, a shot that peels off like a sticky note, a small square that opens onto the next shot.
- **Cut to the lyric.** faster-whisper word times aligned to the lyric sheet by pinyin set the shot boundaries (`edit.js`); `tools/qa.py` checks the render for frozen stretches and jumps.
- No image or video generation model is used for the picture. Song by Chan Meng, made with Suno.

## Make your own

Needs Node 20+, Google Chrome and ffmpeg (Python 3.11+ with numpy, opencv-python and Pillow only to repaint). The song and the painted layers are in the repo. From a POSIX shell (Git Bash on Windows):

```sh
npm install
( cd films/today/paint && python build.py )            # optional: repaint every layer
node harness/render.mjs films/today --workers 3 --audio songs/today.mp3 --crf 18
# → films/today/out/video.mp4
```

## Licence

Code: MIT ([LICENSE](LICENSE)). The film, song, painted layers and other media made by Chan Meng: CC BY 4.0 ([LICENSE-MEDIA.md](LICENSE-MEDIA.md)). Excluded and under their own licences: the fonts (SIL OFL 1.1), the render harness adapted from [mg-styles-15](https://github.com/Vincentwei1021/mg-styles-15) (MIT) and the libraries, as listed in [films/today/CREDITS](films/today/CREDITS).

More detail: [docs/overview.md](docs/overview.md)
