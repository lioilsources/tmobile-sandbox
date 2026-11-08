#!/usr/bin/env python3
"""Pitch visuals (Prompts/08 P1.5): contact mosaics from finished sandbox outputs, CPU only.

  python bench/pitch_visuals.py --out /cold/pitch/visuals

Writes uc10_styles.jpg (motifs x 7 styles), uc8_objects.jpg (FLUX images of 3D objects),
uc28_icons.jpg (selected BioDefense icons on a dark tile) and uc4_kirian.gif (one agent clip).
Each mosaic is a JPEG <= 600 kB so it can go into reports/pitch/ and the deck.
"""
import argparse, io, random, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

STYLES = ["watercolor", "papercut", "crayon", "anime-lite", "woodcut", "clay", "folk"]


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def save_jpeg(im, path, limit=600_000):
    for q in (88, 80, 72, 64, 56):
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=q, optimize=True)
        if buf.tell() <= limit:
            break
    path.write_bytes(buf.getvalue())
    print(f"{path.name} {buf.tell() // 1024} kB q{q}", flush=True)


def uc10(src, out, rows=4, tile=320):
    motifs = sorted({p.name.split(".")[0] for p in src.glob("*.watercolor.0.jpg")})
    random.Random(7).shuffle(motifs)
    motifs = [m for m in motifs if all((src / f"{m}.{s}.0.jpg").exists() for s in STYLES)][:rows]
    head = 44
    W = Image.new("RGB", (tile * len(STYLES), head + tile * len(motifs)), (20, 32, 58))
    d = ImageDraw.Draw(W)
    for c, s in enumerate(STYLES):
        d.text((c * tile + 12, 10), s, fill=(230, 235, 245), font=font(24))
        for r, m in enumerate(motifs):
            W.paste(Image.open(src / f"{m}.{s}.0.jpg").convert("RGB").resize((tile, tile), Image.LANCZOS),
                    (c * tile, head + r * tile))
    save_jpeg(W, out / "uc10_styles.jpg")


def grid(paths, out_path, cols, tile, bg=(20, 32, 58), alpha=False):
    rows = (len(paths) + cols - 1) // cols
    W = Image.new("RGB", (cols * tile, rows * tile), bg)
    for i, p in enumerate(paths):
        im = Image.open(p)
        im = im.convert("RGBA" if alpha else "RGB").resize((tile, tile), Image.LANCZOS)
        W.paste(im, ((i % cols) * tile, (i // cols) * tile), im if alpha else None)
    save_jpeg(W, out_path)


def gif(clip, out_path, seconds=8, width=360, fps=12):
    pal = out_path.with_suffix(".png")
    vf = f"fps={fps},scale={width}:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-t", str(seconds), "-i", str(clip),
                    "-vf", f"{vf},palettegen=max_colors=96", str(pal)], check=True)
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-t", str(seconds), "-i", str(clip), "-i", str(pal),
                    "-lavfi", f"{vf}[x];[x][1:v]paletteuse=dither=bayer", str(out_path)], check=True)
    pal.unlink()
    print(f"{out_path.name} {out_path.stat().st_size // 1024} kB", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/cold/pitch/visuals")
    ap.add_argument("--uc10", default="/srv/work/uc10/styles/img")
    ap.add_argument("--uc8", default="/srv/work/uc8/img")
    ap.add_argument("--icons", default="/cold/biodefense/icons/selected")
    ap.add_argument("--clip", default="/cold/uc4/trailer/pick15/camp10_s0_223739_c07_kill_streak.mp4")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    uc10(Path(a.uc10), out)
    objs = sorted(Path(a.uc8).glob("*.png"))
    grid(objs[::10][:20], out / "uc8_objects.jpg", cols=5, tile=256, bg=(246, 245, 240))
    icons = sorted(Path(a.icons).glob("*.png"))
    if icons:
        grid(icons, out / "uc28_icons.jpg", cols=7, tile=256, alpha=True)
    if Path(a.clip).exists():
        gif(Path(a.clip), out / "uc4_kirian.gif")
