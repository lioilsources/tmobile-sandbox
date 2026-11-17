#!/usr/bin/env python3
"""UC29 Stickers masters: the picked seed of every id -> 768 px WebP q 90 with lossless alpha (BRIEF §5).

  masters.py stage  --picks <dir with {paper,clay,marker}.json> --out /cold/stickers/emoji/masters
  (cut:  ../uc28/icons.py cut --out <out>/src/<style> --fill-all --bg auto --clear-bg-holes 14)
  masters.py export --picks <dir> --out /cold/stickers/emoji/masters

stage copies each picked draft (<id>__<style>__<seed>.png from I4, I3, I2, I1 or I0) to <out>/src/<style>/,
so every master gets a fresh cutout with the current hole rule; export writes <out>/<style>/<id>.webp.
"""
import argparse, json, shutil, sys
from pathlib import Path

D = Path("/cold/stickers/emoji")
BATCHES = {"paper": ["I4_paper", "I3_paper", "I2_paper", "I1", "I0"], "clay": ["I4_clay", "I3_clay", "I2_clay", "I1", "I0"],
           "marker": ["I4_marker", "I3_marker", "I2_marker", "I1", "I0"]}
SIZE, MAX_KB = 768, 150  # 150 kB is a guide; edges win over size (lossless was 150-475 kB)


def picks(a, style):
    p = Path(a.picks) / f"{style}.json"
    return {i: v["pick"] for i, v in json.loads(p.read_text()).items() if v.get("pick")} if p.exists() else {}


def stage(a):
    missing = []
    for style in BATCHES:
        dst = Path(a.out) / "src" / style
        dst.mkdir(parents=True, exist_ok=True)
        for iid, seed in picks(a, style).items():
            name = f"{iid}__{style}__{seed}.png"
            src = next((D / b / name for b in BATCHES[style] if (D / b / name).exists()), None)
            if src is None:
                missing.append(name)
                continue
            shutil.copy2(src, dst / name)
    print(f"staged; missing {len(missing)}: {' '.join(missing)}", flush=True)


def export(a):
    from PIL import Image
    big, missing = [], []
    for style in BATCHES:
        (Path(a.out) / style).mkdir(parents=True, exist_ok=True)
        for iid, seed in picks(a, style).items():
            c = Path(a.out) / "src" / style / "cut" / f"{iid}__{style}__{seed}.png"
            if not c.exists():
                missing.append(c.name)
                continue
            o = Path(a.out) / style / f"{iid}.webp"
            Image.open(c).convert("RGBA").resize((SIZE, SIZE), Image.LANCZOS).save(o, "WEBP", quality=90, alpha_quality=100, method=6, exact=True)
            kb = o.stat().st_size / 1024
            if kb > MAX_KB:
                big.append(f"{style}/{iid} {kb:.0f} kB")
    print(f"exported; missing {len(missing)}: {' '.join(missing)}", flush=True)
    print(f"over {MAX_KB} kB: {len(big)}: {'; '.join(big)}", flush=True)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage", "export"])
    ap.add_argument("--picks", required=True)
    ap.add_argument("--out", default=str(D / "masters"))
    a = ap.parse_args()
    {"stage": stage, "export": export}[a.cmd](a)
