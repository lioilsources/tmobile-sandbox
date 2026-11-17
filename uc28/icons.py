#!/usr/bin/env python3
"""UC28: photoreal cell/pathogen icons for BioDefenseRogue (art/icons/BRIEF.md on branch art/icons).

  icons.py gen   --manifest manifest.json --ids makrofag,kok --out /cold/biodefense/icons/drafts
  icons.py cut   --out /cold/biodefense/icons/drafts          # BiRefNet RGBA cutouts (ComfyUI venv)
  icons.py sheet --manifest manifest.json --ids ... --out ... --sheets <repo>/art/icons/sheets

Model: Qwen-Image fp8 (Apache-2.0) in ComfyUI; seeds = 3000 + 10*index + k (--base).
Also art/fx (G3, G4): manifest "items" instead of "icons", --base 5000; tile items get a seam fix (../uc31/seam.py).
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


def jobs(m, ids, ks, retry=False, base=3000):
    items = m.get("icons") or m["items"]
    order = [ic["id"] for ic in items]
    for iid in ids:
        ic = items[order.index(iid)]
        prompt = f'{m["styles"][ic["style"]]}, {m["tones"][ic["tone"]]}, {ic["subject"]}'
        seeds = ic["retry_seeds"] if retry else [base + 10 * order.index(iid) + k for k in ks]
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
    todo = [(ic, seed, prompt) for ic, seed, prompt in jobs(m, a.ids, a.k, a.retry, a.base)
            if not (out / f'{ic["id"]}_{seed}.png').exists()]
    print(f"todo {len(todo)}{' (serial)' if a.serial else ''}", flush=True)
    pending, t0, times = {}, time.time(), []

    def submit(ic, seed, prompt):
        g = graph(prompt, m.get("negative", ""), seed, f'biodefense/{ic["id"]}_{seed}', ic.get("size") or m["size"])
        pending[call("/prompt", {"prompt": g})["prompt_id"]] = (out / f'{ic["id"]}_{seed}.png', time.time())

    if not a.serial:  # serial: one prompt in the shared ComfyUI queue at a time, so other users' jobs interleave
        for job in todo:
            submit(*job)
        todo = []
    while pending or todo:
        if not pending:
            submit(*todo.pop(0))
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


def clear_holes(rgb, a_, ctol, bg, below=70, above=4, min_px=400):
    """Gaps inside the object (donut hole, padlock shackle, OK ring) still show the light grey background after
    --fill-all: clear enclosed, opaque, neutral-grey regions whose rim is opaque all round. A gap is the background
    or its shadow, never brighter: white parts of the object (plaster, mirror tiles, eye whites) stay."""
    import numpy as np
    from scipy import ndimage
    lum, bg_lum = rgb.mean(axis=2), float(np.mean(bg))
    grey = ((rgb.max(axis=2) - rgb.min(axis=2) <= ctol) & (lum >= bg_lum - below) & (lum <= bg_lum + above)
            & (a_ > 0.5))
    lab, n = ndimage.label(grey)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(n + 1))
    for i in range(1, n + 1):
        if sizes[i] < min_px:
            continue
        m = lab == i
        if lum[m].mean() > bg_lum - 3:  # as bright as the backdrop or brighter: a white or grey part of the object
            continue
        rim = ndimage.binary_dilation(m, iterations=3) & ~m
        # Mostly framed by white, neutral material = the shaded side of that material (plaster, mirror tile), keep it.
        white = (rgb.max(axis=2) - rgb.min(axis=2) <= ctol) & (lum > bg_lum + above)
        if white[rim].mean() > 0.5:
            continue
        if (a_[rim] > 0.5).mean() > 0.97:  # touches the transparent outside = a grey part of the object, keep it
            a_ = np.where(ndimage.binary_dilation(m, iterations=2), 0, a_)
    return a_


def cut(a):
    import numpy as np, torch
    from PIL import Image
    from scipy import ndimage
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation
    dev = "cpu" if a.cpu else "cuda"
    net = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).to(dev).eval()
    net = net.half() if dev == "cuda" else net
    tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    src = Path(a.out); dst = src / "cut"; dst.mkdir(exist_ok=True)
    for p in sorted(src.glob(a.only or "*.png")):
        o = dst / p.name
        if o.exists() and not a.only:
            continue
        im = Image.open(p).convert("RGB")
        with torch.no_grad():
            mask = net(tf(im).unsqueeze(0).to(dev).to(next(net.parameters()).dtype))[-1].sigmoid()[0, 0].float().cpu()
        a_ = np.asarray(transforms.functional.to_pil_image(mask).resize(im.size, Image.LANCZOS), np.float32) / 255
        # BiRefNet leaves dark cell interiors semi-transparent: force the hole-filled core opaque
        rgb = np.asarray(im, np.float32)
        lit = rgb.max(axis=2) > 16  # black background enclosed between pseudopodia stays transparent
        # --fill-all: a dark nucleus inside the cell is not background (cytotoxin), so fill it too
        core = ndimage.binary_erosion(ndimage.binary_fill_holes(a_ > 0.5) & (True if a.fill_all else lit), iterations=3)
        a_ = np.maximum(a_, core.astype(np.float32))
        # edge pixels are alpha * colour + (1 - alpha) * background: un-mix to drop the halo
        # (UC28 background is pure black; --bg auto takes the median border colour, e.g. the light grey of UC29 emoji)
        bg = np.median(np.concatenate([rgb[:8].reshape(-1, 3), rgb[-8:].reshape(-1, 3)]), axis=0) if a.bg == "auto" else np.zeros(3)
        am = np.maximum(a_[..., None], 1e-3)
        rgb = np.where(a_[..., None] > 0.05, np.clip((rgb - (1 - am) * bg) / am, 0, 255), 0)
        if a.drop_dark:  # black background caught between loose particles (pyrogen droplets): only for bright subjects
            # soft luminance key on neutral/warm pixels only (the cyan cell body keeps its dark shading):
            # fully gone below N, fades in up to 2.5 N, so warm droplets lose their dark rims
            key = np.clip((rgb.max(axis=2) - a.drop_dark) / (1.5 * a.drop_dark), 0, 1)
            a_ = np.where(rgb[..., 0] >= rgb[..., 2] - 6, a_ * key, a_)
        if a.clear_bg_holes:
            a_ = clear_holes(rgb, a_, a.clear_bg_holes, bg)
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
    kind = {}
    for ic, seed, _ in jobs(m, a.ids, a.k, a.retry, a.base):
        by.setdefault(ic["id"], []).append(seed); kind[ic["id"]] = ic.get("kind")
    for iid, seeds in by.items():
        seeds = [s for s in seeds if (out / f"{iid}_{s}.png").exists()]
        cut = any((out / "cut" / f"{iid}_{s}.png").exists() for s in seeds)
        tile = kind[iid] == "tile"  # 2x2 repeat below each draft so a seam shows
        S = 24 if cut else 0  # fx: the cutout shrunk to 24 px (in game 12-20 px), next to the seed label
        rows = 2 if cut or tile else 1
        W = Image.new("RGB", (T * len(seeds), rows * T + L), (24, 24, 24)); d = ImageDraw.Draw(W)
        for i, s in enumerate(seeds):
            raw = Image.open(out / f"{iid}_{s}.png").convert("RGB"); raw.thumbnail((T, T), Image.LANCZOS)  # portrait drafts keep aspect
            W.paste(raw, (i * T + (T - raw.width) // 2, (T - raw.height) // 2))
            d.text((i * T + 8, T + 4), f"seed {s}", fill=(230, 230, 230), font=font)
            c = out / "cut" / f"{iid}_{s}.png"
            if c.exists():
                bg = checker(T); fg = Image.open(c).resize((T, T), Image.LANCZOS)
                bg.paste(fg, (0, 0), fg); W.paste(bg, (i * T, T + L))
                th = Image.new("RGB", (S, S), (24, 24, 24)); sm = Image.open(c).resize((S, S), Image.LANCZOS)
                th.paste(sm, (0, 0), sm); W.paste(th, (i * T + T - S - 4, T + 2))
            if tile:
                h = T // 2; q4 = Image.open(out / f"{iid}_{s}.png").convert("RGB").resize((h, h), Image.LANCZOS)
                for dx, dy in ((0, 0), (h, 0), (0, h), (h, h)):
                    W.paste(q4, (i * T + dx, T + L + dy))
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
    ap.add_argument("--only", help="cut: glob of drafts to (re)cut, overwrites existing cutouts")
    ap.add_argument("--fill-all", action="store_true", help="cut: fill every enclosed hole, also dark ones")
    ap.add_argument("--clear-bg-holes", type=int, default=0, metavar="TOL",
                    help="cut: make enclosed neutral-grey regions (chroma <= TOL, >= 400 px) transparent: background showing through gaps")
    ap.add_argument("--cpu", action="store_true", help="cut: BiRefNet on the CPU (slow; for tests while the GPU is busy)")
    ap.add_argument("--drop-dark", type=int, default=0, help="cut: make every pixel darker than N transparent (bright subjects only)")
    ap.add_argument("--bg", choices=["black", "auto"], default="black", help="cut: background colour to un-mix from edge pixels")
    ap.add_argument("--serial", action="store_true", help="submit the next prompt only after the previous one finished")
    ap.add_argument("--retry", action="store_true", help="use each icon's retry_seeds instead of --k")
    ap.add_argument("--base", type=int, default=3000, help="seed = base + 10 * index + k (art/fx: 5000)")
    a = ap.parse_args()
    {"gen": gen, "cut": cut, "sheet": sheet}[a.cmd](a)
