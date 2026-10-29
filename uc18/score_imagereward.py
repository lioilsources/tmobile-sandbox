#!/usr/bin/env python3
"""ImageReward-v1.0 scorer (optional) — runs in its own venv (/cold/venvs/uc18-ir, see setup_imagereward.sh)
because the `image-reward` package's BLIP code needs transformers < 4.45 (the UC25 venv has 5.x).

  /cold/venvs/uc18-ir/bin/python score_imagereward.py --manifest … --refs … --out …/scores/ir.jsonl [--device cuda]

Same contract as score_images.py: reference text = prompt set's prose / ref_en, resumable, per-image errors.
Model: zai-org/ImageReward (Apache-2.0), ImageReward.pt 1.79 GB + med_config.json, tokenizer bert-base-uncased.
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import JsonlWriter, done_keys, limit_threads, log  # noqa: E402

limit_threads(4)
from score_images import load_manifest, load_refs  # noqa: E402

IR_ROOT = os.environ.get("IR_ROOT", "/cold/hf/imagereward")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import torch
    import ImageReward as RM
    refs = load_refs(a.refs)
    rows = [r for r in load_manifest(a.manifest) if r["pid"] in refs]
    done = done_keys(a.out)
    todo = [r for r in rows if r["id"] not in done][: a.limit or None]
    log(f"imagereward: {len(rows)} images, {len(done)} done, {len(todo)} to do")
    if not todo:
        return 0
    m = RM.load(os.path.join(IR_ROOT, "ImageReward.pt"), device=a.device, med_config=os.path.join(IR_ROOT, "med_config.json"))
    w = JsonlWriter(a.out)
    t0, n = time.time(), 0
    for r in todo:
        try:
            with torch.inference_mode():
                s = float(m.score(refs[r["pid"]]["text"], r["path"]))
            w.write({"id": r["id"], "pid": r["pid"], "ckpt": r["ckpt"], "variant": r.get("variant", ""),
                     "scorer": "ImageReward-v1.0", "imagereward": round(s, 4)})
            n += 1
        except Exception as e:  # noqa: BLE001
            w.write({"id": r["id"], "error": str(e)[:500]})
            if n == 0:
                print(f"first error ({r['id']}): {e}", file=sys.stderr, flush=True)
    el = time.time() - t0
    print(json.dumps({"scorer": "ImageReward-v1.0", "images": n, "wall_s": round(el, 1),
                      "img_per_h": round(n / el * 3600) if n else None}), file=sys.stderr)
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
