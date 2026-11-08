#!/usr/bin/env python3
"""UC28: photoreal cell/pathogen icons for BioDefenseRogue (art/icons/BRIEF.md on branch art/icons).

  icons.py gen   --manifest manifest.json --ids makrofag,kok --out /cold/biodefense/icons/drafts
  icons.py cut   --out /cold/biodefense/icons/drafts          # BiRefNet RGBA cutouts (ComfyUI venv)
  icons.py sheet --manifest manifest.json --ids ... --out ... --sheets <repo>/art/icons/sheets

Model: Qwen-Image fp8 (Apache-2.0) in ComfyUI; seeds = 3000 + 10*index + k.
"""
import argparse, io, json, os, shutil, sys, time, urllib.request
from pathlib import Path

API = "http://127.0.0.1:8188"
COMFY_OUT = Path("/srv/work/comfy-out")


def call(path, data=None):
    req = urllib.request.Request(API + path, json.dumps(data).encode() if data else None, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def graph(pos, neg, seed, prefix, size=(1024, 1024)):
    enc = lambda t: {"class_type": "CLIPTextEncode", "inputs": {"clip": ["c", 0], "text": t}}
    return {"u": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_fp8_e4m3fn.safetensors", "weight_dtype": "default"}},
            "c": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image"}},
            "v": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
            "ms": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["u", 0], "shift": 3.1}},
            "p": enc(pos), "n": enc(neg),
            "l": {"class_type": "EmptySD3LatentImage", "inputs": {"width": size[0], "height": size[1], "batch_size": 1}},
            "k": {"class_type": "KSampler", "inputs": {"model": ["ms", 0], "positive": ["p", 0], "negative": ["n", 0], "latent_image": ["l", 0],
                  "seed": seed, "steps": 20, "cfg": 2.5, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
            "d": {"class_type": "VAEDecode", "inputs": {"samples": ["k", 0], "vae": ["v", 0]}},
            "s": {"class_type": "SaveImage", "inputs": {"images": ["d", 0], "filename_prefix": prefix}}}


def jobs(m, ids, ks, retry=False):
    order = [ic["id"] for ic in m["icons"]]
    for iid in ids:
        ic = m["icons"][order.index(iid)]
        prompt = f'{m["styles"][ic["style"]]}, {m["tones"][ic["tone"]]}, {ic["subject"]}'
        seeds = ic["retry_seeds"] if retry else [3000 + 10 * order.index(iid) + k for k in ks]
        for seed in seeds:
            yield ic, seed, prompt


def gen(a):
    m = json.load(open(a.manifest))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for _ in range(90):  # qctl starts cmd jobs right after switch.sh; ComfyUI needs ~20 s
        try:
            call("/system_stats"); break
        except OSError:
            time.sleep(5)
    pending = {}
    for ic, seed, prompt in jobs(m, a.ids, a.k, a.retry):
        dst = out / f'{ic["id"]}_{seed}.png'
        if dst.exists():
            continue
        pid = call("/prompt", {"prompt": graph(prompt, m.get("negative", ""), seed, f'biodefense/{ic["id"]}_{seed}', ic.get("size", m["size"]))})["prompt_id"]
        pending[pid] = (dst, time.time())
    print(f"queued {len(pending)}", flush=True)
    t0, times = time.time(), []
    while pending:
        time.sleep(2)
        for pid in list(pending):
            h = call(f"/history/{pid}").get(pid)
            if not h:
                continue
            dst, _ = pending.pop(pid)
            st = h.get("status", {})
            if st.get("status_str") != "success":
                print(f"FAIL {dst.name}: {st}", flush=True); continue
            img = next(i for o in h["outputs"].values() for i in o.get("images", []))
            shutil.move(COMFY_OUT / img["subfolder"] / img["filename"], dst)
            times.append(time.time())
            print(f"ok {dst.name} ({time.time() - t0:.0f} s)", flush=True)
    if len(times) > 2:
        gaps = sorted(b - a for a, b in zip(times, times[1:]))
        print(f"median s/img warm: {gaps[len(gaps) // 2]:.1f}", flush=True)


def cut(a):
    import numpy as np, torch
    from PIL import Image
    from scipy import ndimage
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation
    net = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).cuda().eval().half()
    tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    src = Path(a.out); dst = src / "cut"; dst.mkdir(exist_ok=True)
    for p in sorted(src.glob("*.png")):
        o = dst / p.name
        if o.exists():
            continue
        im = Image.open(p).convert("RGB")
        with torch.no_grad():
            mask = net(tf(im).unsqueeze(0).cuda().half())[-1].sigmoid()[0, 0].float().cpu()
        a_ = np.asarray(transforms.functional.to_pil_image(mask).resize(im.size, Image.LANCZOS), np.float32) / 255
        # BiRefNet leaves dark cell interiors semi-transparent: force the hole-filled core opaque
        rgb = np.asarray(im, np.float32)
        lit = rgb.max(axis=2) > 16  # black background enclosed between pseudopodia stays transparent
        core = ndimage.binary_erosion(ndimage.binary_fill_holes(a_ > 0.5) & lit, iterations=3)
        a_ = np.maximum(a_, core.astype(np.float32))
        # background is pure black, so edge pixels are alpha * colour: un-premultiply to drop the dark halo
        rgb = np.where(a_[..., None] > 0.05, np.clip(rgb / np.maximum(a_[..., None], 1e-3), 0, 255), 0)
        alpha = Image.fromarray((a_ * 255).astype(np.uint8))
        rgba = Image.fromarray(rgb.astype(np.uint8)); rgba.putalpha(alpha)
        edge = np.concatenate([a_[:3].ravel(), a_[-3:].ravel(), a_[:, :3].ravel(), a_[:, -3:].ravel()])
        if (edge > 0.5).mean() > 0.002:  # object touches the frame = cut off (BRIEF: whole object visible)
            print(f"EDGE {p.name}", flush=True)
        bbox = alpha.point(lambda v: 255 if v > 24 else 0).getbbox()
        if not bbox:
            print(f"EMPTY mask {p.name}", flush=True); continue
        rgba = rgba.crop(bbox)
        side = int(max(rgba.size) / (1 - 2 * 0.06))
        sq = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        sq.paste(rgba, ((side - rgba.width) // 2, (side - rgba.height) // 2))
        sq.resize((1024, 1024), Image.LANCZOS).save(o)
        print(f"cut {p.name}", flush=True)


def checker(size, sq=16):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (size, size), (200, 200, 200)); d = ImageDraw.Draw(im)
    for y in range(0, size, sq):
        for x in range(0, size, sq):
            if (x // sq + y // sq) % 2:
                d.rectangle([x, y, x + sq - 1, y + sq - 1], fill=(250, 250, 250))
    return im


def sheet(a):
    from PIL import Image, ImageDraw, ImageFont
    m = json.load(open(a.manifest))
    out, sheets = Path(a.out), Path(a.sheets); sheets.mkdir(parents=True, exist_ok=True)
    T, L = 384, 28
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    by = {}
    for ic, seed, _ in jobs(m, a.ids, a.k, a.retry):
        by.setdefault(ic["id"], []).append(seed)
    for iid, seeds in by.items():
        seeds = [s for s in seeds if (out / f"{iid}_{s}.png").exists()]
        rows = 2 if any((out / "cut" / f"{iid}_{s}.png").exists() for s in seeds) else 1
        W = Image.new("RGB", (T * len(seeds), rows * T + L), (24, 24, 24)); d = ImageDraw.Draw(W)
        for i, s in enumerate(seeds):
            raw = Image.open(out / f"{iid}_{s}.png").convert("RGB"); raw.thumbnail((T, T), Image.LANCZOS)  # portrait drafts keep aspect
            W.paste(raw, (i * T + (T - raw.width) // 2, (T - raw.height) // 2))
            d.text((i * T + 8, T + 4), f"seed {s}", fill=(230, 230, 230), font=font)
            c = out / "cut" / f"{iid}_{s}.png"
            if c.exists():
                bg = checker(T); fg = Image.open(c).resize((T, T), Image.LANCZOS)
                bg.paste(fg, (0, 0), fg); W.paste(bg, (i * T, T + L))
        for q in (88, 80, 72, 64):
            buf = io.BytesIO(); W.save(buf, "JPEG", quality=q, optimize=True)
            if buf.tell() <= 400_000:
                break
        (sheets / f"{iid}.jpg").write_bytes(buf.getvalue())
        print(f"sheet {iid}.jpg {buf.tell() // 1024} kB q{q}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "cut", "sheet"])
    ap.add_argument("--manifest")
    ap.add_argument("--ids", type=lambda s: s.split(","), default=[])
    ap.add_argument("--k", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1, 2, 3])
    ap.add_argument("--out", default="/cold/biodefense/icons/drafts")
    ap.add_argument("--sheets")
    ap.add_argument("--retry", action="store_true", help="use each icon's retry_seeds instead of --k")
    a = ap.parse_args()
    {"gen": gen, "cut": cut, "sheet": sheet}[a.cmd](a)
