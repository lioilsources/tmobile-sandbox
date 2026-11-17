#!/usr/bin/env python3
"""UC31: seamless arena tile for BioDefenseRogue art/fx (bg-arena), Qwen-Image in ComfyUI.

  seam.py --manifest manifest.json --id bg-arena --out /cold/biodefense/fx/tile --k 0,1,2,3

Qwen-Image is a transformer, so circular padding does not make it tile. Instead: roll the draft by half
(the old borders meet in a cross through the centre, the new borders are continuous by construction),
repaint a soft cross-shaped band with masked img2img in the same model and prompt, and paste only the band
back. The untouched draft is kept as raw/<name>.png; the healed tile replaces <name>.png.
--flatten first evens out the draft's low-frequency brightness (light falloff, vignette): otherwise the rolled
quadrants differ in brightness and the 2x2 repeat shows a grid even where the texture itself joins.
"""
import argparse, io, json, shutil, sys, time, urllib.request, uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "uc28"))
from icons import API, COMFY_OUT, call, jobs  # noqa: E402


def upload(im, name):
    buf = io.BytesIO(); im.save(buf, "PNG"); b = uuid.uuid4().hex
    body = (f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\nContent-Type: image/png\r\n\r\n'.encode()
            + buf.getvalue() + f'\r\n--{b}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n--{b}--\r\n'.encode())
    req = urllib.request.Request(API + "/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={b}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["name"]


def band(size, w):
    """Soft cross through the centre: 1 inside, fading to 0 over w/2 on each side."""
    m = np.zeros((size, size), np.float32); c = size // 2
    m[:, c - w // 2:c + w // 2] = 1; m[c - w // 2:c + w // 2, :] = 1
    return Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(w / 4))


def heal_graph(pos, neg, seed, img, mask, prefix, denoise):
    enc = lambda t: {"class_type": "CLIPTextEncode", "inputs": {"clip": ["c", 0], "text": t}}
    return {"u": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_fp8_e4m3fn.safetensors", "weight_dtype": "default"}},
            "c": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image"}},
            "v": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
            "ms": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["u", 0], "shift": 3.1}},
            "p": enc(pos), "n": enc(neg),
            "i": {"class_type": "LoadImage", "inputs": {"image": img}},
            "m": {"class_type": "LoadImageMask", "inputs": {"image": mask, "channel": "red"}},
            "e": {"class_type": "VAEEncode", "inputs": {"pixels": ["i", 0], "vae": ["v", 0]}},
            "lm": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["e", 0], "mask": ["m", 0]}},
            "k": {"class_type": "KSampler", "inputs": {"model": ["ms", 0], "positive": ["p", 0], "negative": ["n", 0], "latent_image": ["lm", 0],
                  "seed": seed, "steps": 20, "cfg": 2.5, "sampler_name": "euler", "scheduler": "simple", "denoise": denoise}},
            "d": {"class_type": "VAEDecode", "inputs": {"samples": ["k", 0], "vae": ["v", 0]}},
            "s": {"class_type": "SaveImage", "inputs": {"images": ["d", 0], "filename_prefix": prefix}}}


def wait(pid):
    while True:
        h = call(f"/history/{pid}").get(pid)
        if h:
            if h.get("status", {}).get("status_str") != "success":
                raise RuntimeError(str(h.get("status"))[:400])
            return next(i for o in h["outputs"].values() for i in o.get("images", []))
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--id", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1, 2, 3])
    ap.add_argument("--base", type=int, default=5000); ap.add_argument("--band", type=int, default=160)
    ap.add_argument("--denoise", type=float, default=0.6)
    ap.add_argument("--flatten", type=float, default=0, help="even out brightness: gaussian sigma = size / N (e.g. 8)")
    ap.add_argument("--redo", action="store_true", help="heal again from raw/ (e.g. with --flatten)")
    a = ap.parse_args()
    m = json.load(open(a.manifest)); out = Path(a.out); raw = out / "raw"; raw.mkdir(exist_ok=True)
    for ic, seed, prompt in jobs(m, [a.id], a.k, base=a.base):
        p = out / f"{a.id}_{seed}.png"; r = raw / p.name
        if r.exists() and not a.redo:
            continue  # already healed
        im = Image.open(r if r.exists() else p).convert("RGB"); s = im.width
        if a.flatten:
            from scipy import ndimage
            rgb = np.asarray(im, np.float32); lum = rgb.mean(axis=2)
            low = ndimage.gaussian_filter(lum, s / a.flatten, mode="reflect")
            im = Image.fromarray(np.clip(rgb * (lum.mean() / np.maximum(low, 1))[..., None], 0, 255).astype(np.uint8))
        rolled = Image.fromarray(np.roll(np.asarray(im), (s // 2, s // 2), axis=(0, 1)))
        mk = band(s, a.band)
        g = heal_graph(prompt, m.get("negative", ""), seed, upload(rolled, f"seam_{p.name}"), upload(mk, f"seammask_{p.name}"),
                       f"biodefense/seam_{p.stem}", a.denoise)
        img = wait(call("/prompt", {"prompt": g})["prompt_id"])
        f = COMFY_OUT / img["subfolder"] / img["filename"]
        healed = Image.open(f).convert("RGB"); f.unlink()
        w = np.asarray(mk, np.float32)[..., None] / 255
        res = np.asarray(rolled, np.float32) * (1 - w) + np.asarray(healed, np.float32) * w
        if not r.exists():
            shutil.move(p, r)
        Image.fromarray(res.round().astype(np.uint8)).save(p)
        print(f"seam {p.name}", flush=True)


if __name__ == "__main__":
    main()
