"""Lokální LLM přes OpenAI-kompatibilní API (vLLM na VM, 127.0.0.1:8000) — bez závislostí.

Model se bere z /v1/models (profil A = qwen3-32b, S = qwen3-8b); thinking vypnutý přes
chat_template_kwargs {"enable_thinking": false}, odpověď vynucená jako JSON objekt.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request

BASE = os.environ.get("UC24_LLM", "http://127.0.0.1:8000")


def _req(path: str, body: dict | None = None, timeout: float = 120):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


def model() -> str:
    return os.environ.get("UC24_LLM_MODEL") or _req("/v1/models")["data"][0]["id"]


def chat_json(system: str, user: str, temperature: float = 0.3, max_tokens: int = 600, mdl: str | None = None) -> tuple[dict, dict]:
    """→ (parsed JSON, meta {model, ms, usage, raw})"""
    mdl = mdl or model()
    t0 = time.time()
    r = _req("/v1/chat/completions", {
        "model": mdl, "temperature": temperature, "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    })
    raw = r["choices"][0]["message"]["content"] or ""
    txt = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip()).strip()
    try:
        d = json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, re.S)
        d = json.loads(m.group(0)) if m else {}
    return d, {"model": mdl, "ms": round((time.time() - t0) * 1000), "usage": r.get("usage"), "raw": raw}
