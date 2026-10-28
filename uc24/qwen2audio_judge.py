"""UC24 krok 2 — Qwen2-Audio-7B-Instruct jako audio-VLM soudce: klip + popis → JSON
{"matches": bool, "heard": [...], "defects": [hiss|rumble|clipping|clicks|distortion|none], "unpleasant": bool}

GPU: bf16 ~16.5 GB VRAM + aktivace (batch 8 ≈ 19 GB) → potřebuje okno, kdy neběží vLLM s velkým
modelem (profil D/ComfyUI bez Wanu, nebo prázdná karta). Skript sám odmítne start, když je volné VRAM
< --min-free-gb. Spouštění přes run_qwen2audio.sh (s GPU logem).

  python qwen2audio_judge.py --manifest /cold/uc24/synth/manifest.jsonl --split test --limit 400 \
         --out /cold/uc24/judges/qwen2audio.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
from scipy import signal

MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
PROMPT = ("You are an audio quality inspector for a children's story app; sounds are played on a phone speaker.\n"
          "Expected sound: \"{prompt}\".\n"
          "Listen and answer ONLY with JSON: {{\"matches\": true|false (is it the expected sound?), "
          "\"heard\": [short list of sound sources you hear], "
          "\"defects\": [any of \"hiss\", \"rumble\", \"clipping\", \"clicks\", \"distortion\"; or \"none\"], "
          "\"unpleasant\": true|false (unpleasant to listen to)}}")


def free_vram_gb() -> float:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
        return float(out.split()[0]) / 1024
    except Exception:
        return 0.0


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    return {"matches": d.get("matches"), "heard": d.get("heard"), "defects": d.get("defects"),
            "unpleasant": d.get("unpleasant"), "parsed": bool(d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/cold/uc24/synth/manifest.jsonl")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", default="/cold/uc24/judges/qwen2audio.jsonl")
    ap.add_argument("--min-free-gb", type=float, default=21.0)
    a = ap.parse_args()
    if (f := free_vram_gb()) < a.min_free_gb:
        sys.exit(f"volné VRAM {f:.1f} GB < {a.min_free_gb} GB — počkej na okno (viz run_qwen2audio.sh)")
    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    proc = AutoProcessor.from_pretrained(MODEL)
    proc.tokenizer.padding_side = "left"
    model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    sr_m = proc.feature_extractor.sampling_rate
    rows = [json.loads(line) for line in open(a.manifest)]
    rows = [r for r in rows if r["split"] == a.split]
    rng = np.random.default_rng(24)
    rng.shuffle(rows)
    rows = rows[: a.limit]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    t0, n = time.time(), 0
    with open(a.out, "w") as f:
        for i in range(0, len(rows), a.bs):
            chunk = rows[i:i + a.bs]
            audios, texts = [], []
            for r in chunk:
                x, sr = sf.read(r["path"], dtype="float32", always_2d=True)
                x = x.mean(axis=1)
                audios.append(signal.resample_poly(x, sr_m, sr).astype(np.float32) if sr != sr_m else x)
                conv = [{"role": "user", "content": [{"type": "audio", "audio_url": r["path"]},
                                                     {"type": "text", "text": PROMPT.format(prompt=r["prompt"])}]}]
                texts.append(proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False))
            inp = proc(text=texts, audio=audios, sampling_rate=sr_m, return_tensors="pt", padding=True).to("cuda")
            with torch.inference_mode():
                out = model.generate(**inp, max_new_tokens=96, do_sample=False)
            dec = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
            for r, txt in zip(chunk, dec):
                f.write(json.dumps({"id": r["id"], "path": r["path"], "raw": txt, **parse(txt)}) + "\n")
            n += len(chunk)
            f.flush()
    dt = time.time() - t0
    print(json.dumps({"judge": "qwen2-audio-7b", "n": n, "s": round(dt, 1), "clips_per_h": round(n / dt * 3600),
                      "bs": a.bs, "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}))


if __name__ == "__main__":
    main()
