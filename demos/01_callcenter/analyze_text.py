#!/usr/bin/env python3
"""Kontrola: stejná LLM analýza jako process.py, ale na čistém textu scénáře (bez Whisperu).
Odliší chyby rozpoznávání řeči od chyb analýzy / nejednoznačných štítků.

  analyze_text.py --calls /srv/work/demo1/calls --out /srv/work/demo1/runs/<run>-text
"""
import argparse, asyncio, json, pathlib
import httpx
from common import LLM
from process import SCHEMA, PROMPT

async def one(cl, sem, c, out, model):
    call = json.loads(c.read_text())
    dialog = "\n".join(f"{'Operátor' if r['mluvci'] == 'operator' else 'Zákazník'}: {r['text']}" for r in call["repliky"])
    async with sem:
        r = await cl.post(f"{LLM}/chat/completions", json={"model": model, "temperature": 0, "max_tokens": 600,
            "response_format": {"type": "json_schema", "json_schema": {"name": "analyza", "schema": SCHEMA}},
            "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": dialog}]})
    j = r.json()
    (out / c.name).write_text(json.dumps({"id": call["id"], "stt_s": 0, "llm_s": 0, "usage": j["usage"],
        "prepis": {w: " ".join(t["text"] for t in call["repliky"] if t["mluvci"] == w) for w in ("operator", "zakaznik")},
        "dialog": dialog, "analyza": json.loads(j["choices"][0]["message"]["content"])}, ensure_ascii=False, indent=1))

async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--calls", default="/srv/work/demo1/calls"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=600) as cl:
        model = (await cl.get(f"{LLM}/models")).json()["data"][0]["id"]; sem = asyncio.Semaphore(16)
        await asyncio.gather(*(one(cl, sem, c, out, model) for c in sorted(pathlib.Path(a.calls).glob("call_*.json"))))
    (out / "_run.json").write_text(json.dumps({"model": model, "whisper": "— (čistý text)", "hovoru": 40, "souběh": 16, "wall_s": 1, "audio_s": 1}))

if __name__ == "__main__": asyncio.run(main())
