#!/usr/bin/env python3
"""UC1 harness: vzorky půlstran (export_pages.py z genealogy-matriky) → OCR engine (vLLM, OpenAI API) souběžně
→ skóre stejné jako matrika-ocr/finetune/eval_cer.py (JSON validita, shoda počtu řádků, CER, field-exact)
+ skupiny polí (jména / data / místa), latence, tokeny. Výstup: <out>/{pred.jsonl, report.json}.

  run_ocr.py --engine qwen3vl8b --dataset /srv/work/uc1/dataset --out /srv/work/uc1/runs/<run> --concurrency 16
"""
import argparse, asyncio, base64, collections, json, pathlib, statistics, sys, time
import httpx
sys.path.insert(0, "/srv/work/uc1/genealogy-matriky/matrika-ocr/finetune")
from eval_cer import cer, parse_model_json, rows_of   # skórování 1:1 s eval_cer.py

# Tvar odpovědi podle promptu matrika-ocr; limity brání smyčkám (max 40 řádků, buňka max 400 znaků).
SCHEMA = {"type": "object", "required": ["folio", "rok", "rows"], "additionalProperties": False, "properties": {
    "folio": {"type": "string", "maxLength": 20}, "rok": {"type": "string", "maxLength": 20},
    "rows": {"type": "array", "maxItems": 40, "items": {"type": "object", "additionalProperties": {"type": "string", "maxLength": 400}}}}}

def group(field):
    if field.startswith("datum") or "_datum_" in field: return "data"
    if "misto" in field or field.endswith("_bydliste"): return "mista"
    if "jmeno" in field or field in ("svedkove", "krtitel", "kmotri"): return "jmena"
    return "ostatni"

async def one(cl, sem, url, s, img_dir, model, decoding):
    b64 = base64.b64encode((img_dir / s["image"]).read_bytes()).decode()
    body = {"model": model, "temperature": 0, "max_tokens": 4096, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}, {"type": "text", "text": s["prompt"]}]}], **decoding}
    async with sem:
        t0 = time.time()
        try:
            r = await cl.post(url, json=body); r.raise_for_status(); j = r.json()
            return s, j["choices"][0]["message"]["content"], j.get("usage", {}), time.time() - t0, None
        except Exception as e:
            return s, "", {}, time.time() - t0, repr(e)

async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--engine", required=True); ap.add_argument("--dataset", default="/srv/work/uc1/dataset")
    ap.add_argument("--files", nargs="*", default=["pages_train.jsonl", "pages_val.jsonl"]); ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions"); ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--model-name", default="ocr", help="served-model-name na endpointu")
    ap.add_argument("--decoding", choices=["greedy", "fix"], default="greedy",
                    help="greedy = jako eval_cer.py; fix = presence_penalty 1.5 + json_schema (proti smyčkám)")
    a = ap.parse_args(); ds = pathlib.Path(a.dataset); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    samples = []
    for fn in a.files:
        split = "val" if "val" in fn else "train"
        samples += [dict(json.loads(l), _split=split) for l in open(ds / fn) if l.strip()]
    decoding = {} if a.decoding == "greedy" else {"presence_penalty": 1.5,
                "response_format": {"type": "json_schema", "json_schema": {"name": "matrika", "schema": SCHEMA}}}
    sem = asyncio.Semaphore(a.concurrency); t0 = time.time()
    async with httpx.AsyncClient(timeout=900) as cl:
        res = await asyncio.gather(*(one(cl, sem, a.url, s, ds, a.model_name, decoding) for s in samples))
    wall = time.time() - t0
    agg = collections.defaultdict(lambda: collections.Counter()); cers = collections.defaultdict(list); lat, tin, tout, errs = [], 0, 0, 0
    with open(out / "pred.jsonl", "w") as f:
        for s, reply, usage, dt, err in res:
            errs += err is not None; lat.append(dt); tin += usage.get("prompt_tokens", 0); tout += usage.get("completion_tokens", 0)
            gt = json.loads(s["target"]); pred = parse_model_json(reply)
            keys = ["vse", s["_split"], s["meta"]["book"]]
            for k in keys: agg[k]["n"] += 1
            if pred is not None:
                gr_rows, pr_rows = rows_of(gt), rows_of(pred)
                for k in keys:
                    agg[k]["json_ok"] += 1; agg[k]["rowcount_ok"] += len(gr_rows) == len(pr_rows)
                for gr, pr in zip(gr_rows, pr_rows):
                    if not isinstance(pr, dict): continue
                    for fld, gv in gr.items():
                        gv, pv = str(gv).strip(), str(pr.get(fld, "") or "").strip()
                        for k in keys + [f"vse/{group(fld)}"]:
                            agg[k]["f_tot"] += 1; agg[k]["f_ok"] += gv == pv
                            if gv: cers[k].append(cer(gv, pv))
            f.write(json.dumps({"id": s["id"], "split": s["_split"], "sec": round(dt, 2), "usage": usage, "err": err, "reply": reply}, ensure_ascii=False) + "\n")
    fmt = lambda k: {"vzorky": agg[k]["n"], "json_valid_%": round(100 * agg[k]["json_ok"] / max(1, agg[k]["n"]), 1),
                     "rowcount_match_%": round(100 * agg[k]["rowcount_ok"] / max(1, agg[k]["n"]), 1),
                     "CER_neprazdnych": round(statistics.mean(cers[k]), 4) if cers[k] else None,
                     "field_exact_%": round(100 * agg[k]["f_ok"] / max(1, agg[k]["f_tot"]), 1)}
    rep = {"engine": a.engine, "dekodovani": a.decoding, "smycky_4096": sum(1 for r in res if r[2].get("completion_tokens", 0) >= 4000), "vzorku": len(samples), "souběh": a.concurrency, "chyb_http": errs, "wall_s": round(wall, 1),
           "pulstran_za_hodinu": round(len(samples) / wall * 3600), "dvojstran_za_hodinu": round(len(samples) / 2 / wall * 3600),
           "latence_s": {"p50": round(statistics.median(lat), 2), "p95": round(sorted(lat)[int(.95 * len(lat)) - 1], 2)},
           "tokeny_prumer": {"vstup": round(tin / len(samples)), "vystup": round(tout / len(samples))},
           "celkem": fmt("vse"), "val": fmt("val"), "train": fmt("train"),
           "skupiny_poli": {g: {"field_exact_%": fmt(f"vse/{g}")["field_exact_%"], "CER": fmt(f"vse/{g}")["CER_neprazdnych"]}
                            for g in ("jmena", "data", "mista", "ostatni") if agg[f"vse/{g}"]["f_tot"]},
           "po_knihach": {b: fmt(b) for b in agg if "[" in b}}
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps(rep, ensure_ascii=False, indent=1))

if __name__ == "__main__": asyncio.run(main())
