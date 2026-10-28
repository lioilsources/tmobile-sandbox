"""UC25: manifest obrázků vygenerovaných v tomhle sandboxu (žádná uživatelská galerie).

  python3 collect_images.py --out /srv/work/w2/uc25/manifest_all.jsonl
  python3 collect_images.py --out m300.jsonl --per-source 60 --seed 1   # stratifikovaný vzorek

Zdroje (vlna 1): UC10 Storyteller ilustrace (FLUX.1-schnell, 7 stylů), UC7 generované
detaily očí (SDXL base + LoRA), UC8 objekty pro 3D (FLUX), UC5/smoke výstupy ComfyUI.
FFHQ referenční fotky z UC7 (dataset/ref, crops) se NEPOUŽÍVAJÍ — nejsou generované.
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

SOURCES = {
    "uc10-storyteller": ("/srv/work/uc10/styles/img", "*.jpg"),
    "uc7-eyes-base": ("/srv/work/uc7/run1/gen/base", "*.png"),
    "uc7-eyes-lora": ("/srv/work/uc7/run1/gen/lora", "*.png"),
    "uc8-objects": ("/srv/work/uc8/img", "*.png"),
    "uc5-comfy": ("/srv/work/comfy-out/uc5", "*.png"),
    "smoke-flux": ("/srv/work/comfy-out", "smoke_flux_*.png"),
}


def stratum(src, p):
    if src == "uc10-storyteller":  # <hash>.<style>.<variant>.jpg
        m = re.match(r"[0-9a-f]+\.([a-z-]+)\.\d+\.jpg", p.name)
        return f"{src}:{m.group(1) if m else '?'}"
    return src


ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--per-source", type=int, default=0, help="0 = vše; jinak max N na stratum")
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()

by = defaultdict(list)
for src, (d, pat) in SOURCES.items():
    for p in sorted(Path(d).glob(pat)):
        by[stratum(src, p)].append(p)
rnd = random.Random(a.seed)
rows = []
for st, ps in sorted(by.items()):
    take = ps if not a.per_source else rnd.sample(ps, min(a.per_source, len(ps)))
    for p in take:
        rows.append({"image": f"{st.split(':')[0]}/{p.stem}", "path": str(p), "stratum": st})
with open(a.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print({st: min(len(ps), a.per_source or len(ps)) for st, ps in sorted(by.items())}, "→", len(rows))
