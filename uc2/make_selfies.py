#!/usr/bin/env python3
"""UC2: syntetická „selfie“ jako vstup kamery pro hodnocení shaderů (FLUX.1-schnell přes ComfyUI, profil D).

Žádné fotky skutečných lidí: čtyři vygenerované portréty na výšku 768×1024, různé pleti, světlo a pozadí.

  make_selfies.py --out /srv/work/uc2/selfies
"""
import argparse, json, pathlib, time, urllib.request

API = "http://127.0.0.1:8188"
PROMPTS = [
    "smartphone front camera selfie of a smiling young woman with curly dark hair, indoor window light, plain living room background, photo",
    "smartphone front camera selfie of a middle-aged man with short grey beard, evening street lights behind, slightly grainy, photo",
    "smartphone front camera selfie of a teenage boy with freckles and red hair, bright daylight park background, photo",
    "smartphone front camera selfie of an elderly woman with glasses and silver hair, warm kitchen light, photo",
]


def call(path, data=None):
    req = urllib.request.Request(API + path, json.dumps(data).encode() if data else None, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read() if path.startswith("/view") else json.load(r)


def wf(prompt, seed):
    return {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "flux1-schnell-fp8.safetensors"}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": prompt}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": ""}},
            "4": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 768, "height": 1024, "batch_size": 1}},
            "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0],
                  "seed": seed, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
            "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
            "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "uc2/selfie"}}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(PROMPTS):
        pid = call("/prompt", {"prompt": wf(p, 2000 + i)})["prompt_id"]
        while True:
            h = call(f"/history/{pid}")
            if pid in h: break
            time.sleep(0.5)
        im = [x for o in h[pid]["outputs"].values() for x in o.get("images", [])][0]
        (out / f"selfie_{i}.png").write_bytes(call(f"/view?filename={im['filename']}&subfolder={im['subfolder']}&type={im['type']}"))
        print(out / f"selfie_{i}.png", flush=True)


if __name__ == "__main__":
    main()
