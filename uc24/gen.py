"""UC24 — generování klipů za jedním rozhraním: gen(prompt, kind, seed, dur) → (float32 mono, sr, meta)

Backendy (UC24_GEN):
  aistack    AiStack services/audio (POST /v1/audio/sfx|music → 202 job → poll → WAV); stejný kontrakt
             jako Storyteller internal/audiogen. SFX = MOSS-SoundEffect v2.0 (Apache-2.0), hudba = ACE-Step.
             UC24_AUDIO_URL (default http://127.0.0.1:8093). Na L40S VM dnes neběží (jen TTS kontejner).
  comfy-ace  ComfyUI ACE-Step 3.5B (profil D, 127.0.0.1:8188) — jen hudba/ambient smyčky.
  comfy-sao  ComfyUI Stable Audio Open 1.0 — jen pokud je checkpoint (gated HF repo, licence
             Stability Community < 1 M USD obratu → TODO uživatele, viz report).
  sim        SIMULACE pro suchý běh smyčky: procedurální zvuk ze synth.py + vady řízené "rizikovými"
             slovy v promptu (deep/thunderous → rumble, loud/explosive → clipping, vintage/radio → hiss…).
             Měří jen to, že smyčka a přepisovač fungují — NE kvalitu žádného modelu.
"""
from __future__ import annotations

import io
import json
import os
import random
import time
import urllib.request
import zlib

import numpy as np
import soundfile as sf

BACKEND = os.environ.get("UC24_GEN", "sim")


# ---------------- sim ----------------
SIM_TRIGGERS = {
    "rumble": ["deep", "thunder", "rumbl", "boom", "bass", "heavy", "huge", "giant", "dark", "low "],
    "clipping": ["loud", "explosi", "blast", "crash", "smash", "powerful", "roar", "slam"],
    "hiss": ["vintage", "lo-fi", "radio", "old ", "distant", "far away", "misty", "static"],
    "clicks": ["crackl", "glitch", "rattl"],
}
SIM_PROTECT = ["clean", "dry", "no low-end", "no rumble", "no hiss", "no background noise", "moderate level",
               "soft dynamics", "smooth", "no clicks"]
SIM_KEYWORDS = {"owl": "owl", "bell": "bell", "knock": "knock", "step": "footsteps", "bird": "bird", "drip": "drip",
                "whoosh": "whoosh", "coin": "coins", "creak": "creak", "door": "creak", "chime": "chime", "magic": "chime",
                "sparkl": "chime", "bark": "bark", "dog": "bark", "wolf": "bark", "meow": "meow", "cat": "meow",
                "wind": "wind", "rain": "rain", "stream": "stream", "brook": "stream", "river": "stream", "fire": "fire",
                "forest": "forest", "cricket": "crickets", "wave": "waves", "ocean": "waves", "sea": "waves"}


def _sim(prompt: str, kind: str, seed: int, dur: float):
    import synth
    p = prompt.lower()
    typ = next((t for k, t in SIM_KEYWORDS.items() if k in p), None)
    srcs = [k for k, v in synth.SOURCES.items() if v[0] == ("ambient" if kind != "sfx" else "sfx")]
    if typ is None or typ not in synth.SOURCES:
        typ = srcs[zlib.crc32(p.encode()) % len(srcs)]
    k2, fn, _ = synth.SOURCES[typ]
    r = np.random.default_rng(seed)
    n = int((dur or (3.0 if k2 == "sfx" else 8.0)) * synth.SR)
    y = synth.normalize(fn(r, n), r)
    protect = sum(w in p for w in SIM_PROTECT)
    injected = []
    rr = random.Random(seed * 7919)  # jen podle seedu úlohy: změna promptu mění výsledek jen přes slova, ne přes RNG
    for d, words in SIM_TRIGGERS.items():
        hits = sum(w in p for w in words)
        prob = min(0.9, 0.55 * hits) * (0.5 ** protect)
        if rr.random() < prob:
            injected.append(d)
            y = synth.add_defect(y, d, "strong" if hits > 1 else "mild", r)
    if not injected and rr.random() < 0.08 * (0.5 ** protect):  # vada bez slovního spouštěče
        d = rr.choice(["hiss", "clicks", "rumble"])
        injected.append(d)
        y = synth.add_defect(y, d, "mild", r)
    return np.clip(y, -1, 1).astype(np.float32), synth.SR, {"backend": "sim", "sim_type": typ, "sim_injected": injected}


# ---------------- aistack ----------------
def _http(method, url, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def _aistack(prompt, kind, seed, dur):
    base = os.environ.get("UC24_AUDIO_URL", "http://127.0.0.1:8093").rstrip("/")
    if kind == "sfx":
        path, body = "/v1/audio/sfx", {"prompt": prompt, "duration_s": dur or 3.0, "seed": seed, "mono": True, "format": "wav", "variations": 1}
    else:
        path, body = "/v1/audio/music", {"prompt": prompt, "duration_s": dur or 30.0, "seed": seed, "instrumental": True,
                                         "loop": True, "format": "wav", "variations": 1}
    st, b = _http("POST", base + path, body)
    job = json.loads(b)["job_id"]
    t0 = time.time()
    while True:
        time.sleep(1.0)
        _, b = _http("GET", f"{base}/v1/audio/jobs/{job}")
        j = json.loads(b)
        if j["status"] == "done":
            url = j["outputs"][0]["url"]
            _, wav = _http("GET", url if url.startswith("http") else base + url, timeout=120)
            x, sr = sf.read(io.BytesIO(wav), dtype="float32", always_2d=True)
            return x.mean(axis=1), sr, {"backend": "aistack", "job_id": job, "gen_s": round(time.time() - t0, 2)}
        if j["status"] == "error":
            raise RuntimeError(f"audio job {job}: {j.get('error')}")


# ---------------- comfy ----------------
COMFY = os.environ.get("UC24_COMFY", "http://127.0.0.1:8188")
COMFY_OUT = os.environ.get("UC24_COMFY_OUT", "/srv/work/comfy-out")


def _ks(model, pos, neg, lat, steps, cfg, seed, sampler="euler", sched="simple"):
    return {"class_type": "KSampler", "inputs": {"model": model, "positive": pos, "negative": neg, "latent_image": lat,
            "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": sched, "denoise": 1.0}}


def _wf_ace(prompt, seed, dur):  # stejné uzly jako uc5/comfy_bench.py ace_step
    return {"m": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "ace_step_v1_3.5b.safetensors"}},
            "t": {"class_type": "TextEncodeAceStepAudio", "inputs": {"clip": ["m", 1], "tags": prompt,
                  "lyrics": "[instrumental]", "lyrics_strength": 0.99}},
            "n": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["t", 0]}},
            "l": {"class_type": "EmptyAceStepLatentAudio", "inputs": {"seconds": float(dur or 30.0), "batch_size": 1}},
            "k": _ks(["m", 0], ["t", 0], ["n", 0], ["l", 0], 50, 5.0, seed),
            "d": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["k", 0], "vae": ["m", 2]}},
            "s": {"class_type": "SaveAudio", "inputs": {"audio": ["d", 0], "filename_prefix": "uc24/ace"}}}


def _wf_sao(prompt, seed, dur):  # ComfyUI nativní Stable Audio Open 1.0 (checkpoint + t5_base)
    return {"m": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "stable-audio-open-1.0.safetensors"}},
            "c": {"class_type": "CLIPLoader", "inputs": {"clip_name": "t5_base.safetensors", "type": "stable_audio"}},
            "p": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["c", 0], "text": prompt}},
            "q": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["c", 0], "text": "low quality, noise, hiss, distortion"}},
            "l": {"class_type": "EmptyLatentAudio", "inputs": {"seconds": float(dur or 3.0), "batch_size": 1}},
            "k": _ks(["m", 0], ["p", 0], ["q", 0], ["l", 0], 50, 4.98, seed, "dpmpp_3m_sde_gpu", "exponential"),
            "d": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["k", 0], "vae": ["m", 2]}},
            "s": {"class_type": "SaveAudio", "inputs": {"audio": ["d", 0], "filename_prefix": "uc24/sao"}}}


def _comfy(prompt, kind, seed, dur, sao=False):
    wf = _wf_sao(prompt, seed, dur) if sao else _wf_ace(prompt, seed, dur)
    t0 = time.time()
    _, b = _http("POST", COMFY + "/prompt", {"prompt": wf})
    pid = json.loads(b)["prompt_id"]
    while True:
        time.sleep(0.5)
        _, b = _http("GET", f"{COMFY}/history/{pid}")
        h = json.loads(b)
        if pid in h:
            break
    st = h[pid]["status"]
    if st.get("status_str") != "success":
        raise RuntimeError(json.dumps(st)[-600:])
    a = h[pid]["outputs"]["s"]["audio"][0]
    path = os.path.join(COMFY_OUT, a.get("subfolder", ""), a["filename"])
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return x.mean(axis=1), sr, {"backend": "comfy-sao" if sao else "comfy-ace", "file": path, "gen_s": round(time.time() - t0, 2)}


def gen(prompt: str, kind: str = "sfx", seed: int = 0, dur: float | None = None, backend: str | None = None):
    b = backend or BACKEND
    t0 = time.time()
    if b == "sim":
        x, sr, m = _sim(prompt, kind, seed, dur)
    elif b == "aistack":
        x, sr, m = _aistack(prompt, kind, seed, dur)
    elif b in ("comfy-ace", "comfy-sao"):
        x, sr, m = _comfy(prompt, kind, seed, dur, sao=b == "comfy-sao")
    else:
        raise ValueError(b)
    m["gen_ms"] = round((time.time() - t0) * 1000)
    return x, sr, m
