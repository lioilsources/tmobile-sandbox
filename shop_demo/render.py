#!/usr/bin/env python3
"""Render a shop demo batch on L40S ComfyUI with commercially licensed models only.

  render.py <batch.json> <out_dir>

batch.json: [{id, prompt, width, height, seed}]. Items alternate FLUX.1-schnell / Qwen-Image.
Output: <out>/<id>.png and <out>/manifest.json [{id, file, model, seed, width, height, steps}].
Resumable: existing PNGs are kept.
"""
import json, os, sys, time, urllib.request
from pathlib import Path

COMFY = os.environ.get("COMFY", "http://127.0.0.1:8188")
MODELS = ["FLUX.1-schnell", "Qwen-Image"]
NEG = os.environ.get("NEG", "people, person, human, face, text, watermark, signature")


def graph(model, prompt, seed, w, h):
    if model == "Qwen-Image":
        load = {"u": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_fp8_e4m3fn.safetensors", "weight_dtype": "default"}},
                "c": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image"}},
                "v": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
                "ms": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["u", 0], "shift": 3.1}}}
        m, clip, vae, steps, cfg, neg = ["ms", 0], ["c", 0], ["v", 0], 20, 2.5, NEG
    else:  # FLUX.1-schnell (Apache 2.0): 4 steps, no CFG
        load = {"ck": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "flux1-schnell-fp8.safetensors"}}}
        m, clip, vae, steps, cfg, neg = ["ck", 0], ["ck", 1], ["ck", 2], 4, 1.0, ""
    return steps, {**load,
            "p": {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": prompt}},
            "n": {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": neg}},
            "l": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
            "k": {"class_type": "KSampler", "inputs": {"model": m, "positive": ["p", 0], "negative": ["n", 0], "latent_image": ["l", 0],
                                                        "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler",
                                                        "scheduler": "simple", "denoise": 1.0}},
            "d": {"class_type": "VAEDecode", "inputs": {"samples": ["k", 0], "vae": vae}},
            "s": {"class_type": "SaveImage", "inputs": {"images": ["d", 0], "filename_prefix": "shop_demo/img"}}}


def call(path, body=None):
    req = urllib.request.Request(COMFY + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"} if body else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main():
    # Profile D starts ComfyUI without waiting for it; give it up to 5 min to come up.
    for _ in range(150):
        try:
            call("/system_stats")
            break
        except OSError:
            time.sleep(2)
    items, out = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    # Group by model so each one loads once; the model choice still alternates by item index.
    order = sorted(range(len(items)), key=lambda i: i % len(MODELS))
    for i in order:
        it, model = items[i], MODELS[i % len(MODELS)]
        steps, g = graph(model, it["prompt"], it["seed"], it["width"], it["height"])
        o = out / f'{it["id"]}.png'
        if not o.exists():
            pid = json.loads(call("/prompt", {"prompt": g}))["prompt_id"]
            t = time.time()
            while True:
                hist = json.loads(call(f"/history/{pid}")).get(pid)
                if hist and hist.get("status", {}).get("completed"):
                    break
                if hist and hist.get("status", {}).get("status_str") == "error":
                    raise SystemExit(f'ComfyUI error on {it["id"]}')
                time.sleep(2)
            img = next(x for n in hist["outputs"].values() for x in n.get("images", []))
            o.write_bytes(call(f'/view?filename={img["filename"]}&subfolder={img["subfolder"]}&type={img["type"]}'))
            print(f'{it["id"]} {model} {time.time() - t:.0f} s', flush=True)
        manifest.append({"id": it["id"], "file": o.name, "model": model, "seed": it["seed"],
                         "width": it["width"], "height": it["height"], "steps": steps})
    manifest.sort(key=lambda r: r["id"])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"images": {"ok": len(manifest)}}))


if __name__ == "__main__":
    main()
