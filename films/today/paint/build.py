"""Paint the film's layers into ../art.   usage: python build.py [job ...] [--view=name,name,...]   (no jobs = everything)
Jobs are the keys of each scene module's ALL dict, prefixed by nothing; `python build.py --list` prints them.
With WC_VIEW=<dir> and --view groups, also writes flattened previews of the named layers (back to front) for checking."""
import importlib
import json
import os
import sys
import time

import cv2
import numpy as np

import wc

MODS = ["s_kitchen", "s_fridge", "s_table", "s_sill", "s_entry", "s_block", "s_lane", "s_stall", "s_shop", "s_step", "s_figure", "s_props", "s_misc"]


def main():
    want = [a for a in sys.argv[1:] if not a.startswith("--")]
    jobs = {}
    for mn in MODS:
        try:
            m = importlib.import_module(mn)
        except ModuleNotFoundError as e:
            if e.name != mn:
                raise
            continue
        for k, fn in m.ALL.items():
            jobs[k] = fn
    if "--list" in sys.argv:
        print(" ".join(jobs))
        return
    for k, fn in jobs.items():
        if want and k not in want:
            continue
        t = time.time()
        fn()
        print(k, round(time.time() - t, 1), "s", flush=True)
    man = wc.merge_manifest()
    view = os.environ.get("WC_VIEW")
    if view:
        for group in [a[7:] for a in sys.argv[1:] if a.startswith("--view=")]:
            names = group.split(",")
            base = None
            for n in names:
                at = None
                if "@" in n:
                    n, at = n.split("@")
                    at = [float(v) for v in at.split(":")]
                e = man[n]
                im = cv2.imread(os.path.join(wc.ART, e["f"]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
                if base is None:
                    base = np.ones((e["sh"], e["sw"], 3), np.float32) * wc.PAPER[::-1]
                a = np.ones(im.shape[:2] + (1,), np.float32) if im.shape[2] == 3 else im[..., 3:4]
                # sprites placed with name@x:y go by their anchor; sheets go by their own offset
                x = int(at[0] - e.get("ax", 0) + e["ox"]) if at else e["ox"]
                y = int(at[1] - e.get("ay", 0) + e["oy"]) if at else e["oy"]
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(base.shape[1], x + im.shape[1]), min(base.shape[0], y + im.shape[0])
                if x1 <= x0 or y1 <= y0:
                    continue
                sub, sa = im[y0 - y:y1 - y, x0 - x:x1 - x, :3], a[y0 - y:y1 - y, x0 - x:x1 - x]
                base[y0:y1, x0:x1] = base[y0:y1, x0:x1] * (1 - sa) + sub * sa
            sc = 1600 / base.shape[1]
            out = cv2.resize(base, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(view, names[0].split("@")[0] + ".jpg"), (out * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
