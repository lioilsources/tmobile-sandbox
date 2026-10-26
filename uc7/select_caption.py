#!/usr/bin/env python3
"""UC7 krok 2: z výřezů (crop_eyes.py) vybere tréninkovou sadu RealEyes a napíše k ní popisky.

CLIP ViT-H-14 (open_clip, laion2B) na GPU: zero-shot vyřadí brýle, děti, kostýmy/malování a zakryté oči, nejméně
ostrou třetinu vyřadí podle rozptylu Laplaciánu. Zbytek se rozdělí: N obrázků pro trénink vybraných na
náhodně a --ref obrázků náhodně stranou jako reálná reference pro FID.
Popisek = spouštěcí slovo + zero-shot atributy (duhovka, líčení, věk, světlo), formát kohya (stejné jméno .txt).

  select_caption.py --crops /srv/work/uc7/crops --out /srv/work/uc7/dataset --n 1500 --ref 500
Výstup: <out>/train/<id>.jpg+.txt, <out>/ref/<id>.jpg, <out>/summary.json, <out>/sheet.jpg
"""
import argparse, collections, json, pathlib, random, shutil

import numpy as np
import open_clip
import torch
from PIL import Image

TRIGGER = "rlyz"
FILTERS = {  # (vyřadit, ponechat) — rozhoduje, která skupina promptů má vyšší pravděpodobnost
    "bryle": (["a close-up photo of eyes behind eyeglasses", "a person wearing glasses with frames", "a person wearing sunglasses"],
              ["a close-up photo of bare human eyes without glasses"]),
    "dite": (["a close-up photo of the eyes of a child", "the face of a kid", "the face of a teenager"],
             ["a close-up photo of the eyes of an adult"]),
    "kostym": (["a face with face paint", "theatrical costume makeup", "colored contact lenses", "a person in a costume",
                "glitter and jewels on the face"], ["a natural photo of an ordinary person's face"]),
    "zakryto": (["eyes covered by hair or a hand", "a blurry photo of a face", "a person winking or squinting"],
                ["a sharp close-up photo of two open eyes"]),
}
FILTER_P = 0.3  # stačí 30 % pravděpodobnosti „vyřadit“ — u tréninku vadí víc propuštěný kostým než ztracená fotka
ATTRS = {
    "duhovka": {"brown eyes": "dark brown eyes", "light brown eyes": "light brown eyes", "blue eyes": "blue eyes",
                "green eyes": "green eyes", "hazel eyes": "hazel eyes", "grey eyes": "grey eyes"},
    "liceni": {"no makeup": "eyes without makeup", "subtle mascara": "eyes with subtle mascara",
               "eyeliner and mascara": "eyes with eyeliner and mascara", "heavy eye makeup": "eyes with heavy eye shadow makeup"},
    "vek": {"young adult": "the eyes of a young adult", "middle-aged": "the eyes of a middle-aged person",
            "elderly": "the wrinkled eyes of an elderly person"},
    "svetlo": {"soft natural light": "eyes in soft natural daylight", "bright sunlight": "eyes in harsh bright sunlight",
               "studio light": "eyes in studio lighting", "dim light": "eyes in dim warm indoor light"},
}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=pathlib.Path, required=True); ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--n", type=int, default=1500); ap.add_argument("--ref", type=int, default=500)
    ap.add_argument("--sharp-drop", type=float, default=0.33); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.crops / "meta.jsonl")]
    stats = collections.Counter(r.get("vyrazeno", "ok") for r in rows)
    ok = [r for r in rows if "vyrazeno" not in r]

    model, _, prep = open_clip.create_model_and_transforms("hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    tok = open_clip.get_tokenizer("hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    model = model.cuda().half().eval()

    def text(ps):
        e = model.encode_text(tok(ps).cuda()); return e / e.norm(dim=-1, keepdim=True)

    emb = []
    for i in range(0, len(ok), 128):
        x = torch.stack([prep(Image.open(a.crops / "img" / f"{r['id']}.jpg").convert("RGB")) for r in ok[i:i + 128]]).cuda().half()
        e = model.encode_image(x); emb.append((e / e.norm(dim=-1, keepdim=True)).float().cpu())
    emb = torch.cat(emb)
    scale = model.logit_scale.exp().float().item()

    keep = torch.ones(len(ok), dtype=torch.bool)
    for name, (bad, good) in FILTERS.items():
        p = (scale * emb.cuda() @ text(bad + good).float().T).softmax(-1).cpu()
        hit = p[:, :len(bad)].sum(1) > FILTER_P
        stats[name] = int((hit & keep).sum()); keep &= ~hit
    sharp = torch.tensor([r["sharp"] for r in ok])
    thr = float(torch.quantile(sharp[keep], a.sharp_drop))
    stats["neostre"] = int((keep & (sharp < thr)).sum()); keep &= sharp >= thr
    pool = keep.nonzero().squeeze(1).tolist()
    stats["kandidatu"] = len(pool)

    for key, opts in ATTRS.items():
        p = (scale * emb.cuda() @ text(list(opts.values())).float().T).softmax(-1).argmax(-1).cpu()
        labels = list(opts)
        for i, r in enumerate(ok):
            r[key] = labels[p[i]]

    random.Random(a.seed).shuffle(pool)
    ref, pool = pool[:a.ref], pool[a.ref:]
    # náhodný vzorek; greedy k-center (max. rozmanitost) v 1. verzi bral přednostně extrémy (kostýmy, malování)
    train = pool[:a.n]

    for d in ("train", "ref"):
        shutil.rmtree(a.out / d, ignore_errors=True); (a.out / d).mkdir(parents=True)
    for i in train:
        r = ok[i]; shutil.copy(a.crops / "img" / f"{r['id']}.jpg", a.out / "train" / f"{r['id']}.jpg")
        cap = f"{TRIGGER}, close-up photo of a pair of human eyes, {r['duhovka']}, {r['liceni']}, {r['vek']}, {r['svetlo']}"
        (a.out / "train" / f"{r['id']}.txt").write_text(cap)
    for i in ref:
        shutil.copy(a.crops / "img" / f"{ok[i]['id']}.jpg", a.out / "ref" / f"{ok[i]['id']}.jpg")

    sheet = Image.new("RGB", (4 * 384, 6 * 192))
    for k, i in enumerate(train[:24]):
        sheet.paste(Image.open(a.crops / "img" / f"{ok[i]['id']}.jpg").resize((384, 192)), ((k % 4) * 384, (k // 4) * 192))
    sheet.save(a.out / "sheet.jpg", quality=85)
    summ = {"tvari": len(rows), "postupne_vyrazeno": dict(stats), "prah_ostrosti": round(thr, 1),
            "train": len(train), "ref": len(ref),
            "atributy_train": {k: dict(collections.Counter(ok[i][k] for i in train)) for k in ATTRS}}
    (a.out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
