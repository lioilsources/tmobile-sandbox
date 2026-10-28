"""UC24 krok 2 — percepční a sémantičtí soudci nad sadou klipů (unikátní cesty z manifestu).

  CUDA_VISIBLE_DEVICES= python judges.py --judge clap-general --device cpu    # → <out>/clap-general.npz
  python judges.py --judge clap-unfused | panns | aesthetics [--limit N] [--device cuda]

Výstupy (<out> = /cold/uc24/judges):
  clap-*.npz      paths, emb (N×512, L2 norm.), model, s/klip
  panns.npz       paths, probs (N×527 AudioSet clipwise), labels
  aesthetics.npz  paths, CE, CU, PC, PQ  (Meta Audiobox Aesthetics, 1–10)
Skóre per řádek manifestu (prompt, mismatch) počítá eval_judges.py. Qwen2-Audio je zvlášť
(qwen2audio_judge.py — potřebuje GPU ~17 GB).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PANNS_DIR = os.environ.get("PANNS_DIR", "/cold/uc24/ext/panns")


def paths_from(manifest: str, limit: int | None) -> list[str]:
    seen, out = set(), []
    for line in open(manifest):
        p = json.loads(line)["path"]
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out[:limit] if limit else out


def load(p):
    x, sr = sf.read(p, dtype="float32", always_2d=False)
    return (x.mean(axis=1) if x.ndim == 2 else x), sr


def run_clap(paths, model_id, device, bs=16):
    from sfxgate.clap import Clap
    c = Clap(model_id, device)
    embs = []
    for i in range(0, len(paths), bs):
        xs = [load(p) for p in paths[i:i + bs]]
        embs.append(c.audio_emb([x for x, _ in xs], [s for _, s in xs]))
    return {"emb": np.concatenate(embs).astype(np.float32), "model": model_id}


def run_panns(paths, device, bs=16):
    import torch
    from scipy import signal
    from panns_inference import AudioTagging, labels
    at = AudioTagging(checkpoint_path=os.path.join(PANNS_DIR, "Cnn14_mAP=0.431.pth"), device=device)
    out = []
    for i in range(0, len(paths), bs):
        xs = []
        for p in paths[i:i + bs]:
            x, sr = load(p)
            xs.append(signal.resample_poly(x, 32000, sr) if sr != 32000 else x)
        L = max(len(x) for x in xs)
        batch = np.stack([np.pad(x, (0, L - len(x))) for x in xs]).astype(np.float32)
        with torch.inference_mode():
            probs, _ = at.inference(batch)
        out.append(probs)
    return {"probs": np.concatenate(out).astype(np.float32), "labels": np.array(labels)}


def run_aesthetics(paths, device, bs=8):
    import torch
    from audiobox_aesthetics.infer import AesPredictor
    pred = AesPredictor(checkpoint_pth=None, precision="32" if device == "cpu" else "bf16")
    pred.model.to(device)
    pred.device = torch.device(device)
    res = {k: [] for k in ("CE", "CU", "PC", "PQ")}
    for i in range(0, len(paths), bs):
        items = []
        for p in paths[i:i + bs]:
            x, sr = load(p)
            items.append({"path": torch.from_numpy(x)[None, :], "sample_rate": sr})
        for row in pred.forward(items):
            for k in res:
                res[k].append(row[k])
    return {k: np.array(v, np.float32) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, choices=["clap-general", "clap-unfused", "panns", "aesthetics"])
    ap.add_argument("--manifest", default="/cold/uc24/synth/manifest.jsonl")
    ap.add_argument("--out", default="/cold/uc24/judges")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    paths = paths_from(a.manifest, a.limit)
    t0 = time.time()
    if a.judge == "clap-general":
        r = run_clap(paths, "laion/larger_clap_general", a.device)
    elif a.judge == "clap-unfused":
        r = run_clap(paths, "laion/clap-htsat-unfused", a.device)
    elif a.judge == "panns":
        r = run_panns(paths, a.device)
    else:
        r = run_aesthetics(paths, a.device)
    dt = time.time() - t0
    audio_s = sum(sf.info(p).duration for p in paths)
    r.update(paths=np.array(paths), seconds=dt, n=len(paths), audio_s=audio_s, device=a.device, threads=a.threads)
    suffix = f"_{a.limit}" if a.limit else ""
    np.savez(os.path.join(a.out, f"{a.judge}{suffix}.npz"), **r)
    print(json.dumps({"judge": a.judge, "n": len(paths), "s": round(dt, 1), "clips_per_h": round(len(paths) / dt * 3600),
                      "audio_s_per_s": round(audio_s / dt, 2), "device": a.device, "threads": a.threads}))


if __name__ == "__main__":
    main()
