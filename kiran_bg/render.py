#!/usr/bin/env python3
"""Kiran parallax backgrounds on L40S ComfyUI with a commercially usable model (Qwen-Image or FLUX.1-schnell).

The prompts come unchanged from the Kiran pipeline (`go run ./cmd/generate -asset-type background -dry-run`);
this only swaps the image model, which the pipeline's flux/pony workflows cannot do.

  render.py --dry-run-log <file> --model qwen|schnell --out <dir> [--skins axelay,ikaruga] [--assets layer_0_z0,layer_2] [--pairs galaga/layer_0_z3,...] [--extra "..."] [--seed 1]

Output: <out>/<skin>/backgrounds/<asset>_v1.jpg, the layout `cmd/postprocess -input` reads.
COMFY (default http://127.0.0.1:18190) is the ComfyUI API through the ssh tunnel.
"""
import argparse, io, json, os, re, time, urllib.request
from pathlib import Path

COMFY = os.environ.get("COMFY", "http://127.0.0.1:18190")
SIZES = {("1:2", "2k"): (1024, 2048), ("1:2", "1k"): (512, 1024), ("2:1", "2k"): (2048, 1024), ("1:1", "1k"): (1024, 1024)}


def parse(log):
    """[n] backgrounds/<asset> blocks of the dry run -> (skin, asset, ratio, res, prompt)."""
    skin, out, cur = None, [], None
    for line in Path(log).read_text().splitlines():
        if m := re.match(r"=== DRY RUN: .* \((\w+)\) ===", line):
            skin = m.group(1)
        elif m := re.match(r"\[\d+\] backgrounds/(\S+)", line):
            cur = {"skin": skin, "asset": m.group(1), "prompt": []}
            out.append(cur)
        elif cur is not None and (m := re.match(r"\s+Type: .* \| Ratio: (\S+) \| Res: (\S+)", line)):
            cur["ratio"], cur["res"] = m.groups()
        elif cur is not None and line.strip() == "Prompt:":
            cur["in"] = True
        elif cur is not None and cur.get("in"):
            if not line.strip():
                cur["in"] = False
            else:
                cur["prompt"].append(line.strip())
    return [dict(c, prompt="\n".join(c["prompt"])) for c in out]


def graph(model, prompt, seed, w, h):
    if model == "qwen":
        load = {"u": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_fp8_e4m3fn.safetensors", "weight_dtype": "default"}},
                "c": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image"}},
                "v": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
                "ms": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["u", 0], "shift": 3.1}}}
        m, clip, vae, steps, cfg, neg = ["ms", 0], ["c", 0], ["v", 0], 20, 2.5, "gradient, horizon, sky, vignette, text, watermark"
    else:  # FLUX.1-schnell (Apache 2.0): 4 steps, no CFG
        load = {"ck": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "flux1-schnell-fp8.safetensors"}}}
        m, clip, vae, steps, cfg, neg = ["ck", 0], ["ck", 1], ["ck", 2], 4, 1.0, ""
    return {**load,
            "p": {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": prompt}},
            "n": {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": neg}},
            "l": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
            "k": {"class_type": "KSampler", "inputs": {"model": m, "positive": ["p", 0], "negative": ["n", 0], "latent_image": ["l", 0],
                                                        "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler",
                                                        "scheduler": "simple", "denoise": 1.0}},
            "d": {"class_type": "VAEDecode", "inputs": {"samples": ["k", 0], "vae": vae}},
            "s": {"class_type": "SaveImage", "inputs": {"images": ["d", 0], "filename_prefix": "kiran_bg/" + model}}}


def call(path, body=None):
    req = urllib.request.Request(COMFY + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"} if body else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-log", required=True)
    ap.add_argument("--model", choices=["qwen", "schnell"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--assets", type=lambda s: s.split(","), default=[])
    ap.add_argument("--skins", type=lambda s: s.split(","), default=[])
    ap.add_argument("--pairs", type=lambda s: set(s.split(",")), default=set(), help="exact skin/asset items, overrides --skins/--assets")
    ap.add_argument("--extra", default="", help="appended to every positive prompt")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    from PIL import Image
    jobs = [j for j in parse(a.dry_run_log)
            if (f'{j["skin"]}/{j["asset"]}' in a.pairs) if a.pairs else
            ((not a.assets or j["asset"] in a.assets) and (not a.skins or j["skin"] in a.skins))]
    for j in jobs:
        o = Path(a.out) / j["skin"] / "backgrounds" / f'{j["asset"]}_v1.jpg'
        if o.exists():
            continue
        w, h = SIZES[(j["ratio"], j["res"])]
        pid = json.loads(call("/prompt", {"prompt": graph(a.model, j["prompt"] + (", " + a.extra if a.extra else ""), a.seed, w, h)}))["prompt_id"]
        t = time.time()
        while True:
            hist = json.loads(call(f"/history/{pid}")).get(pid)
            if hist and hist.get("status", {}).get("completed"):
                break
            if hist and hist.get("status", {}).get("status_str") == "error":
                raise SystemExit(f"ComfyUI error on {j['skin']}/{j['asset']}")
            time.sleep(2)
        img = next(i for n in hist["outputs"].values() for i in n.get("images", []))
        data = call(f'/view?filename={img["filename"]}&subfolder={img["subfolder"]}&type={img["type"]}')
        o.parent.mkdir(parents=True, exist_ok=True)
        Image.open(io.BytesIO(data)).convert("RGB").save(o, quality=95)
        print(f'{j["skin"]}/{j["asset"]} {a.model} {time.time() - t:.0f} s', flush=True)
    # cmd/postprocess reads <skin>/manifest.json; list just the backgrounds rendered here.
    for skin in sorted({j["skin"] for j in jobs}):
        assets = [{"name": j["asset"], "type": "background", "dir": "backgrounds", "aspect_ratio": j["ratio"],
                   "resolution": j["res"], "variations": 1} for j in jobs if j["skin"] == skin]
        m = {"version": "1", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": a.model,
             "skin": {"id": skin}, "assets": assets, "directories": ["backgrounds"]}
        (Path(a.out) / skin / "manifest.json").write_text(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
