#!/usr/bin/env python3
"""UC29: own emoji set for Handwritten Stickers (tools/emoji/BRIEF.md on branch emoji-own-set).

  emoji.py gen   --jobs I0.json --out /cold/stickers/emoji/I0          # Qwen-Image fp8 in ComfyUI, one batch
  (cut: ../uc28/icons.py cut --out <out> --fill-all --bg auto, ComfyUI venv)
  emoji.py sheet --jobs I0.json --out /cold/stickers/emoji/I0 --sheets <dir> --batch I0

Jobs come finished from Stickers (make_jobs.py): {file, id, style, seed, size, prompt, negative}; prompts are used as is.
Graph and sampler settings are the UC28 ones (20 steps, cfg 2.5, shift 3.1, euler/simple).
"""
import argparse, io, json, shutil, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "uc28"))
from icons import COMFY_OUT, call, checker, graph  # noqa: E402


def gen(a):
    jobs = json.load(open(a.jobs))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for _ in range(90):  # qctl starts cmd jobs right after switch.sh; ComfyUI needs ~20 s
        try:
            call("/system_stats"); break
        except OSError:
            time.sleep(5)
    todo = [j for j in jobs if not (out / j["file"]).exists()]
    print(f"todo {len(todo)}", flush=True)
    pending, t0, times = {}, time.time(), []
    for j in todo:  # one batch: serial submits reload the model when ComfyUI is shared (MODEL_NOTES, 8. 11.)
        g = graph(j["prompt"], j["negative"], j["seed"], f'stickers/{Path(j["file"]).stem}', (j["size"], j["size"]))
        pending[call("/prompt", {"prompt": g})["prompt_id"]] = out / j["file"]
    while pending:
        time.sleep(2)
        for pid in list(pending):
            h = call(f"/history/{pid}").get(pid)
            if not h:
                continue
            dst = pending.pop(pid)
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


def save_jpeg(im, path, limit):
    for q in (90, 85, 80, 72, 64, 56):
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=q, optimize=True)
        if buf.tell() <= limit:
            break
    path.write_bytes(buf.getvalue())
    return buf.tell() // 1024, q


def sheet(a):
    """One sheet per style: row = emoji, column = seed, 256 px cutout on a checkerboard, seed below;
    with --strip also a 64 px strip of every cutout (the app shows emoji at 64 px)."""
    from PIL import Image, ImageDraw, ImageFont
    jobs = json.load(open(a.jobs))
    out, sheets = Path(a.out), Path(a.sheets); sheets.mkdir(parents=True, exist_ok=True)
    T, L = 256, 22
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    def cell(j, size):
        c = out / "cut" / j["file"]
        bg = checker(size, max(4, size // 16))
        if c.exists():
            fg = Image.open(c).resize((size, size), Image.LANCZOS); bg.paste(fg, (0, 0), fg)
        return bg

    # --by-category: one sheet per style and category (id prefix before "_", e.g. hand_thumbs_up -> hand)
    cat = (lambda j: j["id"].split("_")[0]) if a.by_category else (lambda j: "")
    for style, c in dict.fromkeys((j["style"], cat(j)) for j in jobs):
        sj = [j for j in jobs if j["style"] == style and cat(j) == c]
        name = f"{a.batch}_{style}" + (f"_{c}" if c else "")
        ids = list(dict.fromkeys(j["id"] for j in sj))
        cols = max(sum(j["id"] == i for j in sj) for i in ids)
        W = Image.new("RGB", (T * cols, len(ids) * (T + L)), (24, 24, 24)); d = ImageDraw.Draw(W)
        for r, iid in enumerate(ids):
            for c, j in enumerate(j for j in sj if j["id"] == iid):
                W.paste(cell(j, T), (c * T, r * (T + L)))
                d.text((c * T + 6, r * (T + L) + T + 3), f"{iid} · seed {j['seed']}", fill=(230, 230, 230), font=font)
        kb, q = save_jpeg(W, sheets / f"{name}.jpg", 1_500_000)
        print(f"sheet {name}.jpg {kb} kB q{q}", flush=True)
        if a.strip:
            S = Image.new("RGB", (64 * len(sj), 64), (24, 24, 24))
            for k, j in enumerate(sj):
                S.paste(cell(j, 64), (64 * k, 0))
            kb, q = save_jpeg(S, sheets / f"{name}_64px.jpg", 300_000)
            print(f"strip {name}_64px.jpg {kb} kB", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "sheet"])
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheets")
    ap.add_argument("--batch", default="I0")
    ap.add_argument("--strip", action="store_true", help="sheet: also a 64 px strip per style")
    ap.add_argument("--by-category", action="store_true", help="sheet: one sheet per style and category")
    a = ap.parse_args()
    {"gen": gen, "sheet": sheet}[a.cmd](a)
