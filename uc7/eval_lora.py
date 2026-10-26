#!/usr/bin/env python3
"""UC7 krok 4: vyhodnocení LoRA RealEyes.

- FID a KID (clean-fid, Inception) base SDXL vs SDXL+LoRA, obojí proti odloženým reálným výřezům (ref, do tréninku nešly).
- CLIP podobnost k centroidu reálných výřezů (ViT-H-14) — jak moc výstup vypadá jako fotka skutečných očí.
- Mřížka: řádky = 4 prompty × síla LoRA (0 / 0,6 / 1,0), sloupce = epochy 2–10.

  eval_lora.py --run /srv/work/uc7/run1 --ref /srv/work/uc7/dataset/ref
Výstup: <run>/eval.json, <run>/grid.jpg
"""
import argparse, json, pathlib

import numpy as np
import open_clip
import torch
from cleanfid import fid
from PIL import Image, ImageDraw


@torch.no_grad()
def clip_sim(dirs, ref):
    m, _, prep = open_clip.create_model_and_transforms("hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    m = m.cuda().half().eval()

    def emb(d):
        fs = sorted(p for p in pathlib.Path(d).iterdir() if p.suffix in (".png", ".jpg")); out = []
        for i in range(0, len(fs), 64):
            x = torch.stack([prep(Image.open(f).convert("RGB")) for f in fs[i:i + 64]]).cuda().half()
            e = m.encode_image(x).float(); out.append(e / e.norm(dim=-1, keepdim=True))
        return torch.cat(out)

    c = emb(ref).mean(0); c = c / c.norm()
    return {k: round(float((emb(d) @ c).mean()), 4) for k, d in dirs.items()}


def grid(run):
    eps = sorted(run.glob("gen/grid_e*"))
    cols = [sorted(e.glob("*.png")) for e in eps]
    rows = len(cols[0]); w, h = 352, 176
    g = Image.new("RGB", (60 + w * len(cols), 24 + h * rows), "white"); dr = ImageDraw.Draw(g)
    for j, (e, fs) in enumerate(zip(eps, cols)):
        dr.text((60 + j * w + 8, 6), f"epocha {int(e.name[-6:])}", fill="black")
        for i, f in enumerate(fs):
            g.paste(Image.open(f).resize((w, h)), (60 + j * w, 24 + i * h))
    for i in range(rows):
        dr.text((4, 24 + i * h + h // 2), f"P{i // 3 + 1} ×{(0.0, 0.6, 1.0)[i % 3]}", fill="black")
    g.save(run / "grid.jpg", quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--ref", type=pathlib.Path, required=True)
    a = ap.parse_args()
    gens = {"base": a.run / "gen" / "base", "lora": a.run / "gen" / "lora"}
    res = {}
    for k, d in gens.items():
        res[f"fid_{k}"] = round(fid.compute_fid(str(d), str(a.ref), mode="clean", num_workers=4), 2)
        res[f"kid_{k}"] = round(fid.compute_kid(str(d), str(a.ref), mode="clean", num_workers=4), 5)
    res["clip_k_realnym"] = clip_sim(gens, a.ref)
    res["n_gen"] = len(list(gens["lora"].glob("*.png"))); res["n_ref"] = len(list(a.ref.glob("*.jpg")))
    grid(a.run)
    (a.run / "eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
