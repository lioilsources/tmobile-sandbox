#!/usr/bin/env python3
"""UC30: BiRefNet cutouts of Kačeny Wan clip frames, f_NN.png -> c_NN.png in the same folder (ComfyUI venv).

  cut_frames.py --root /cold/kaceny/pilot --clips deco/seko-3393930636:0-20 cga/seko-884165268:0-9 ...

Same matte as uc28 icons.py cut (--fill-all --bg auto: hole-filled core, edge colours un-mixed against the
light grey background), but the frame keeps its size and position: no crop, no square, no resize.
"""
import argparse
from pathlib import Path

import numpy as np, torch
from PIL import Image
from scipy import ndimage
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--clips", nargs="+", required=True, help="<id>/<clip>:<first>-<last>")
    a = ap.parse_args()
    net = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).cuda().eval().half()
    tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    n = 0
    for spec in a.clips:
        clip, rng = spec.split(":"); lo, hi = map(int, rng.split("-"))
        d = Path(a.root) / clip
        for i in range(lo, hi + 1):
            p, o = d / f"f_{i:02d}.png", d / f"c_{i:02d}.png"
            if not p.exists():
                print(f"MISSING {p}", flush=True); continue
            im = Image.open(p).convert("RGB")
            with torch.no_grad():
                mask = net(tf(im).unsqueeze(0).cuda().half())[-1].sigmoid()[0, 0].float().cpu()
            al = np.asarray(transforms.functional.to_pil_image(mask).resize(im.size, Image.LANCZOS), np.float32) / 255
            al = np.maximum(al, ndimage.binary_erosion(ndimage.binary_fill_holes(al > 0.5), iterations=3).astype(np.float32))
            rgb = np.asarray(im, np.float32)
            bg = np.median(np.concatenate([rgb[:8].reshape(-1, 3), rgb[-8:].reshape(-1, 3)]), axis=0)
            am = np.maximum(al[..., None], 1e-3)
            rgb = np.where(al[..., None] > 0.05, np.clip((rgb - (1 - am) * bg) / am, 0, 255), 0)
            out = Image.fromarray(rgb.astype(np.uint8)); out.putalpha(Image.fromarray((al * 255).astype(np.uint8)))
            out.save(o); n += 1
        print(f"cut {clip} {lo}-{hi}", flush=True)
    print(f"done {n} frames", flush=True)


if __name__ == "__main__":
    main()
