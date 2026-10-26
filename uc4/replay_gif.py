#!/usr/bin/env python3
"""UC4: replay epizody agenta (evaluate.py → replays/sector_XX.json) → schematický GIF.

Loď = zelený trojúhelník (jas podle HP), nepřátelé = oranžové obdélníky, jejich střely = červené tečky.
Vykresluje se každý druhý krok (krok = 4 snímky hry), tedy 7,5 kroku za sekundu herního času.

  replay_gif.py runs/ppo1/eval/replays/sector_05.json --out sector_05.gif
"""
import argparse, json
from PIL import Image, ImageDraw

SCALE = 1.2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("replay"); ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=2); ap.add_argument("--max-frames", type=int, default=400)
    a = ap.parse_args()
    r = json.load(open(a.replay)); W, H = r["field"]; w, h = int(W * SCALE), int(H * SCALE)
    frames = []
    for f in r["frames"][:: a.every][: a.max_frames]:
        im = Image.new("RGB", (w, h), (10, 12, 24)); d = ImageDraw.Draw(im)
        for x, y, size in f["h"]:
            s = max(size * SCALE / 2, 4); d.rectangle([x * SCALE - s, y * SCALE - s, x * SCALE + s, y * SCALE + s], outline=(255, 150, 40), width=2)
        for x, y in f["p"]:
            d.ellipse([x * SCALE - 3, y * SCALE - 3, x * SCALE + 3, y * SCALE + 3], fill=(255, 60, 60))
        vx, vy = f["v"][0] * SCALE, f["v"][1] * SCALE; g = int(80 + 175 * max(f["hp"], 0))
        d.polygon([(vx, vy - 12), (vx - 9, vy + 9), (vx + 9, vy + 9)], fill=(40, g, 90))
        d.text((6, 4), f"sektor {r['sector']}  skore {f['score']}  HP {round(100 * f['hp'])} %", fill=(220, 220, 220))
        frames.append(im)
    if frames:
        frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=133, loop=0, optimize=True)
    print(a.out, len(frames), "snímků", r["end"])


if __name__ == "__main__":
    main()
