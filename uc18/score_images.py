#!/usr/bin/env python3
"""UC18/UC19 per-image scorers: PickScore, SigLIP 2 prompt consistency, WD14 tag recall + rating gate.

  score_images.py --manifest /cold/w2/uc18/manifest.jsonl --refs uc18/prompts/uc18_prompts.jsonl \
      --scorer pick|siglip|wd14 --out /cold/w2/uc18/scores/pick.jsonl [--device cuda] [--batch 16] [--limit N]

Reference per image = its prompt id in --refs, NOT the prompt that was sent: UC18 refs are the hand-written
prose + tags; UC19 refs are the independent English gist (`ref_en`) + `must_tags` / `must_not` of the Czech
request, so a translator can't score well by echoing its own words.

- pick:   PickScore_v1 (CLIP-H fine-tune) = logit_scale · cos(image, ref text). Weights from the local HF cache
          (~/.cache/huggingface, already on the VM — no download); processor from laion/CLIP-ViT-H-14.
- siglip: google/siglip2-so400m-patch16-512 sigmoid(match) + cosine = "CLIP consistency" (no OpenAI CLIP on
          the box; SigLIP 2 is already in /srv/hf from UC25). Also saves the ref-text embeddings per prompt id
          (--text-emb-out) for the router.
- wd14:   wd-swinv2-tagger-v3 via FineTuneGallery taxo/eval_wd14.WD14 (same preprocessing/threshold 0.35 as the
          gallery); recall = |ref tags ∩ detected| / |ref tags ∩ vocabulary|, must_not hits (negation), and the
          rating gate: questionable+explicit > 0.5 → nsfw_flag (counted per checkpoint, image excluded from reports).
Resumable (skips ids already in --out), one fsynced JSONL line per image, per-image errors recorded.
"""
from __future__ import annotations

import argparse, glob, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import JsonlWriter, done_keys, limit_threads, log, read_jsonl  # noqa: E402

limit_threads(int(os.environ.get("SCORE_THREADS", "4")))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

FTG_TAXO = os.environ.get("FTG_TAXO", "/srv/work/w2/uc25/ftg/taxo")


def snapshot(repo: str) -> str:
    """Local snapshot dir in any HF cache we have (home cache holds PickScore + CLIP-H processor)."""
    roots = [os.environ.get("HF_HOME", "/srv/hf"), os.path.expanduser("~/.cache/huggingface"), "/cold/hf"]
    for r in roots:
        hits = sorted(glob.glob(os.path.join(r, "hub", "models--" + repo.replace("/", "--"), "snapshots", "*")))
        if hits:
            return hits[-1]
    raise FileNotFoundError(f"{repo} not in {roots} — see uc18/score.sh DOWNLOADS")


def load_manifest(path: str) -> list[dict]:
    last = {}
    for r in read_jsonl(path):  # last ok row per id wins (resume can re-log an id)
        if not r.get("error") and r.get("path"):
            last[r["id"]] = r
    return list(last.values())


def load_refs(path: str) -> dict:
    out = {}
    for r in read_jsonl(path):
        pid = r.get("id") or r.get("cid")
        text = r.get("ref_en") or r.get("prose")
        tags = r.get("must_tags") if "must_tags" in r else r.get("tags", "")
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        out[pid] = {"text": text, "tags": tags, "must_not": r.get("must_not", []), "style": r.get("style", "")}
    return out


def _open(p):
    im = Image.open(p)
    im.load()
    return im.convert("RGB")


class Pick:
    name = "pickscore_v1"

    def __init__(self, device):
        from transformers import AutoModel, AutoProcessor
        self.dev = device
        self.dt = torch.float16 if device == "cuda" else torch.bfloat16
        self.proc = AutoProcessor.from_pretrained(snapshot("laion/CLIP-ViT-H-14-laion2B-s32B-b79K"))
        self.m = AutoModel.from_pretrained(snapshot("yuvalkirstain/PickScore_v1"), dtype=self.dt).eval().to(device)

    @torch.inference_mode()
    def __call__(self, ims, texts):
        x = self.proc(images=ims, return_tensors="pt")["pixel_values"].to(self.dev, self.dt)
        t = self.proc(text=texts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.dev)
        fi = self.m.get_image_features(pixel_values=x)
        ft = self.m.get_text_features(**t)
        fi = fi if torch.is_tensor(fi) else fi.pooler_output  # transformers 5.x (MODEL_NOTES)
        ft = ft if torch.is_tensor(ft) else ft.pooler_output
        fi = torch.nn.functional.normalize(fi.float(), dim=-1)
        ft = torch.nn.functional.normalize(ft.float(), dim=-1)
        s = self.m.logit_scale.exp().float() * (fi * ft).sum(-1)
        return [{"pickscore": round(float(v), 4)} for v in s]


class Sig:
    name = "siglip2-so400m-patch16-512"

    def __init__(self, device):
        from transformers import AutoModel, AutoProcessor
        repo = "google/siglip2-so400m-patch16-512"
        self.dev = device
        self.dt = torch.float16 if device == "cuda" else torch.bfloat16
        self.proc = AutoProcessor.from_pretrained(repo)
        self.m = AutoModel.from_pretrained(repo, dtype=self.dt).eval().to(device)
        self.cache = {}

    @torch.inference_mode()
    def text(self, texts):
        need = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if need:
            t = self.proc(text=need, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(self.dev)
            f = self.m.get_text_features(**t)
            f = f if torch.is_tensor(f) else f.pooler_output
            for k, v in zip(need, torch.nn.functional.normalize(f.float(), dim=-1)):
                self.cache[k] = v
        return torch.stack([self.cache[t] for t in texts])

    @torch.inference_mode()
    def __call__(self, ims, texts):
        x = self.proc(images=ims, return_tensors="pt")["pixel_values"].to(self.dev, self.dt)
        f = self.m.get_image_features(pixel_values=x)
        f = torch.nn.functional.normalize((f if torch.is_tensor(f) else f.pooler_output).float(), dim=-1)
        cos = (f * self.text(texts)).sum(-1)
        logit = cos * self.m.logit_scale.exp().float() + self.m.logit_bias.float()
        return [{"siglip_cos": round(float(c), 4), "siglip_p": round(float(torch.sigmoid(l)), 4)} for c, l in zip(cos, logit)]


class Wd14:
    name = "wd-swinv2-tagger-v3"

    def __init__(self, device):
        sys.path.insert(0, FTG_TAXO)
        from eval_wd14 import WD14  # FineTuneGallery evaluator (UC25), unchanged
        self.m = WD14(device, int(os.environ.get("SCORE_THREADS", "4")))
        if device == "cuda" and "CUDA" not in self.m.provider:
            raise SystemExit(f"WD14 fell back to {self.m.provider} — onnxruntime-gpu 1.22 needed (MODEL_NOTES)")
        self.vocab = set(self.m.names)

    def __call__(self, ims, refs):
        res = self.m(ims)
        out = []
        for r, ref in zip(res, refs):
            det = set(r["general"])
            want = [t for t in ref["tags"] if t in self.vocab]
            hit = [t for t in want if t in det]
            bad = [t for t in ref["must_not"] if t in det]
            rq = r["rating"].get("questionable", 0) + r["rating"].get("explicit", 0)
            out.append({"wd14_recall": round(len(hit) / len(want), 4) if want else None, "wd14_n_ref": len(want),
                        "wd14_missing": [t for t in want if t not in det][:20], "must_not_hits": bad,
                        "rating": r["rating"], "nsfw_flag": rq > 0.5, "wd14_tags": sorted(det)[:60]})
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--scorer", required=True, choices=["pick", "siglip", "wd14"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--text-emb-out", default="", help="siglip: npz of ref-text embeddings per prompt id")
    a = ap.parse_args(argv)

    refs = load_refs(a.refs)
    rows = [r for r in load_manifest(a.manifest) if r["pid"] in refs]
    done = done_keys(a.out)
    todo = [r for r in rows if r["id"] not in done]
    if a.limit:
        todo = todo[: a.limit]
    log(f"score {a.scorer}: {len(rows)} images, {len(done)} done, {len(todo)} to do, device {a.device}")
    if not todo and not a.text_emb_out:
        return 0
    m = {"pick": Pick, "siglip": Sig, "wd14": Wd14}[a.scorer](a.device)
    w = JsonlWriter(a.out)
    t0, n_ok, n_err, first = time.time(), 0, 0, None
    pool = ThreadPoolExecutor(4)

    def load(chunk):
        ims, ok = [], []
        for r in chunk:
            try:
                ims.append(_open(r["path"]))
                ok.append(r)
            except Exception as e:  # deleted/corrupt PNG must not kill the pass (UC25 lesson)
                w.write({"id": r["id"], "error": f"image: {e}"[:500]})
        return ok, ims

    chunks = [todo[i: i + a.batch] for i in range(0, len(todo), a.batch)]
    futs = [pool.submit(load, ch) for ch in chunks[:2]]
    for i in range(len(chunks)):
        ok, ims = futs[i].result()
        if i + 2 < len(chunks):
            futs.append(pool.submit(load, chunks[i + 2]))
        if not ok:
            continue
        try:
            if a.scorer == "wd14":
                res = m(ims, [refs[r["pid"]] for r in ok])
            else:
                res = m(ims, [refs[r["pid"]]["text"] for r in ok])
        except Exception as e:  # noqa: BLE001
            n_err += len(ok)
            first = first or str(e)
            if n_err == len(ok):
                print(f"first error: {e}", file=sys.stderr, flush=True)
            for r in ok:
                w.write({"id": r["id"], "error": str(e)[:500]})
            if n_ok == 0 and n_err >= 3 * a.batch:
                log(f"score {a.scorer}: first {n_err} images all failed — abort")
                return 3
            continue
        for r, s in zip(ok, res):
            w.write({"id": r["id"], "pid": r["pid"], "ckpt": r["ckpt"], "variant": r.get("variant", ""),
                     "style": refs[r["pid"]]["style"] or r.get("style", ""), "seed_idx": r.get("seed_idx"),
                     "scorer": m.name, **s})
        n_ok += len(ok)
        if i == 0 or (i + 1) % 50 == 0:
            el = time.time() - t0
            log(f"score {a.scorer}: {n_ok}/{len(todo)} ({n_ok / el * 3600:.0f} img/h)")
    w.close()
    if a.scorer == "siglip" and a.text_emb_out:
        pids = sorted(refs)
        emb = torch.cat([m.text([refs[p]["text"] for p in pids[i: i + 64]]) for i in range(0, len(pids), 64)])
        np.savez(a.text_emb_out, pids=np.array(pids), emb=emb.cpu().numpy().astype(np.float16))
    el = time.time() - t0
    print(json.dumps({"scorer": m.name, "images": n_ok, "errors": n_err, "wall_s": round(el, 1),
                      "img_per_h": round(n_ok / el * 3600) if el and n_ok else None}), file=sys.stderr)
    return 0 if n_ok or not todo else 1


if __name__ == "__main__":
    sys.exit(main())
