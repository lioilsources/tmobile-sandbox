#!/usr/bin/env python3
"""Rozhodčí (Qwen3-32B s přemýšlením, profil A) ohodnotí čistý text každého scénáře. Hovory, kde se rozhodčí
neshodne se zadáním ve všech štítcích, přesune do <calls>_rejected/ → ground truth zůstane jednoznačný.

  validate.py --calls /srv/work/demo1/calls --keep 50
"""
import argparse, asyncio, json, pathlib, shutil
import httpx
from common import LLM, LABELS, BOOLS
from process import SCHEMA, PROMPT

async def judge(cl, sem, f, model):
    call = json.loads(f.read_text())
    dialog = "\n".join(f"{'Operátor' if r['mluvci'] == 'operator' else 'Zákazník'}: {r['text']}" for r in call["repliky"])
    async with sem:
        r = await cl.post(f"{LLM}/chat/completions", json={"model": model, "temperature": 0, "max_tokens": 12000,
            "response_format": {"type": "json_schema", "json_schema": {"name": "analyza", "schema": SCHEMA}},
            "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": dialog}]})
    content = r.json()["choices"][0]["message"]["content"]
    if not content:                                # rozhodčí nedomyslel v limitu → nejednoznačné, zamítnout
        return f, {"rozhodci": ("odpověď", "žádná (limit tokenů)")}
    pred = json.loads(content)
    diff = {k: (call["labels"][k], pred[k]) for k in list(LABELS) + BOOLS if call["labels"][k] != pred[k]}
    return f, diff

async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--calls", default="/srv/work/demo1/calls"); ap.add_argument("--keep", type=int, default=50)
    a = ap.parse_args(); calls = pathlib.Path(a.calls); rej = calls.with_name(calls.name + "_rejected"); rej.mkdir(exist_ok=True)
    async with httpx.AsyncClient(timeout=900) as cl:
        model = (await cl.get(f"{LLM}/models")).json()["data"][0]["id"]; sem = asyncio.Semaphore(16)
        res = await asyncio.gather(*(judge(cl, sem, f, model) for f in sorted(calls.glob("call_*.json"))))
    kept, log = 0, []
    for f, diff in res:
        if diff or kept >= a.keep:
            shutil.move(f, rej / f.name); log.append({"hovor": f.stem, "duvod": {k: f"zadání {g} / rozhodčí {p}" for k, (g, p) in diff.items()} or "nad limit"})
        else: kept += 1
    (calls.parent / "validace.json").write_text(json.dumps({"rozhodci": model, "vygenerovano": len(res), "ponechano": kept,
        "zamitnuto_nesoulad": sum(1 for _, d in res if d), "zamitnuti": log}, ensure_ascii=False, indent=1))
    print(f"ponecháno {kept}/{len(res)}, nesoulad {sum(1 for _, d in res if d)}")

if __name__ == "__main__": asyncio.run(main())
