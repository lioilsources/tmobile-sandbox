#!/usr/bin/env python3
"""UC2: z archivů evoluce vybere nejlepší shadery, ověří je překladačem Flutteru (impellerc, všechny cíle vč. SkSL)
a vyrenderuje animované náhledy (WebP, 12 snímků za 2 s nad syntetickou selfie) pro galerii v reportu.

  gallery.py --mirrorbooth /srv/work/uc2/MirrorBooth --selfies /srv/work/uc2/selfies --run /srv/work/uc2/runs/evo1 \
             --flutter-root /srv/work/uc4/flutter --top 8
Výstup: <run>/gallery/<téma>_<id>.webp, <run>/gallery/gallery.json (skóre, impellerc, původ, kód).
"""
import argparse, json, pathlib, sys
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirrorbooth", type=pathlib.Path, required=True); ap.add_argument("--selfies", type=pathlib.Path, required=True)
    ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--flutter-root", required=True)
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    sys.path.insert(0, str(a.mirrorbooth / "pipeline"))
    from checks.compile import compile_shader
    from preview.render import create_context, render
    ctx = create_context(); out = a.run / "gallery"; out.mkdir(exist_ok=True)
    selfie = np.asarray(Image.open(sorted(a.selfies.glob("*.png"))[0]).convert("RGB").resize((384, 512)), dtype=np.uint8)
    items = []
    for tdir in sorted(p for p in a.run.iterdir() if (p / "archive.json").exists()):
        arch = json.load(open(tdir / "archive.json"))
        # nejlepší semeno tématu kvůli srovnání + nejlepší děti (nová generace, ne semena)
        best_seed = max((r for r in arch if r["id"].startswith("seed_")), key=lambda r: r["fit"], default=None)
        kids = [r for r in arch if not r["id"].startswith("seed_")][: a.top]
        for r in ([best_seed] if best_seed else []) + kids:
            code = (tdir / "shaders" / f"{r['id']}.frag").read_text()
            comp = compile_shader(code, flutter_root=a.flutter_root)
            times = tuple(i * 2 / 12 for i in range(12)) if r.get("needs_time") else (0.0,)
            res = render(code, selfie, r.get("needs_time", False), r.get("needs_face", False), times=times, ctx=ctx)
            frames = [Image.fromarray(f) for f in res.frames]
            name = f"{tdir.name}_{r['id']}.webp"
            frames[0].save(out / name, save_all=True, append_images=frames[1:], duration=167, loop=0, quality=70)
            items.append({"tema": tdir.name, "id": r["id"], "semeno": r["id"].startswith("seed_"), "fit": r["fit"],
                          "pick": r["pick"], "face_keep": r["face_keep"], "motion": r["motion"], "flicker": r["flicker"],
                          "composite": r["composite"], "frame_ms": r["frame_ms"], "alu": r["alu"], "tex_reads": r["tex_reads"],
                          "rodic": r.get("parent"), "impellerc": comp.impeller_verified,
                          "impellerc_chyby": comp.errors[:3], "nahled": name, "kod": code})
            print(tdir.name, r["id"], r["fit"], "impellerc", comp.impeller_verified, flush=True)
    (out / "gallery.json").write_text(json.dumps(items, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
