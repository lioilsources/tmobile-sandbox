#!/usr/bin/env python3
"""UC8: obrázek → 3D (Hunyuan3D 2.1 přes nativní uzly ComfyUI) pro UGC předměty.

Fáze A: FLUX.1-schnell vyrobí obrázky předmětů na čistém pozadí (prompty = kategorie × motiv).
Fáze B: každý obrázek → Hunyuan3D 2.1 (30 kroků, cfg 5, octree 256, surface net) → GLB.
Měří s/obrázek a s/model; kontrola meshe přes trimesh (vodotěsnost, počet trojúhelníků,
limit Roblox UGC doplňku 4 000 trojúhelníků → kolik by šlo bez decimace).

  gen3d.py --out /srv/work/uc8 --n 200
"""
import argparse, itertools, json, pathlib, random, time, urllib.request
import trimesh

API = "http://127.0.0.1:8188"
COMFY_OUT = pathlib.Path("/srv/work/comfy-out")  # --output-directory ComfyUI
KINDS = ["knight helmet", "crown", "magic staff", "wooden shield", "treasure chest", "potion bottle", "dragon pet",
         "cat backpack", "fairy wings", "pirate hat", "sword", "lantern", "flower pot", "toy robot", "teapot",
         "mushroom house", "rocket", "cupcake", "owl figurine", "viking axe"]
MOODS = ["golden ornate", "cute pastel", "rusty medieval", "glowing crystal", "candy-colored", "stone carved",
         "frozen ice", "jungle vine-covered", "steampunk brass", "cartoon low-poly"]
FRAME = "3D game asset render of a {m} {k}, single object centered, whole object visible, plain white background, soft studio lighting, three-quarter view"


def call(path, data=None, raw=None, ctype="application/json"):
    req = urllib.request.Request(API + path, raw if raw is not None else (json.dumps(data).encode() if data else None), {"Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def wait(pid, poll=0.5):
    while True:
        h = call(f"/history/{pid}")
        if pid in h: return h[pid]
        time.sleep(poll)


def img_wf(prompt, seed, prefix):
    return {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "flux1-schnell-fp8.safetensors"}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": prompt}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": ""}},
            "4": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
            "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0],
                  "seed": seed, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
            "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
            "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": prefix}}}


def mesh_wf(image_name, seed, prefix):
    return {"1": {"class_type": "ImageOnlyCheckpointLoader", "inputs": {"ckpt_name": "hunyuan_3d_v2.1.safetensors"}},
            "2": {"class_type": "LoadImage", "inputs": {"image": image_name}},
            "3": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 1.0}},
            "13": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["1", 1], "image": ["2", 0], "crop": "center"}},
            "6": {"class_type": "Hunyuan3Dv2Conditioning", "inputs": {"clip_vision_output": ["13", 0]}},
            "4": {"class_type": "EmptyLatentHunyuan3Dv2", "inputs": {"resolution": 4096, "batch_size": 1}},
            "7": {"class_type": "KSampler", "inputs": {"model": ["3", 0], "positive": ["6", 0], "negative": ["6", 1], "latent_image": ["4", 0],
                  "seed": seed, "steps": 30, "cfg": 5.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
            "8": {"class_type": "VAEDecodeHunyuan3D", "inputs": {"samples": ["7", 0], "vae": ["1", 2], "num_chunks": 8000, "octree_resolution": 256}},
            "9": {"class_type": "VoxelToMesh", "inputs": {"voxel": ["8", 0], "algorithm": "surface net", "threshold": 0.6}},
            "10": {"class_type": "SaveGLB", "inputs": {"mesh": ["9", 0], "filename_prefix": prefix}}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--resume", help="run.log předchozího běhu: hotové modely (glb/NNN.glb) se přeskočí")
    a = ap.parse_args(); out = pathlib.Path(a.out); (out / "img").mkdir(parents=True, exist_ok=True); (out / "glb").mkdir(exist_ok=True)
    combos = list(itertools.product(MOODS, KINDS)); random.Random(8).shuffle(combos); combos = combos[: a.n]
    rows = []
    prev = {}
    if a.resume and pathlib.Path(a.resume).exists():
        for line in open(a.resume):
            if line.startswith("{"):
                r = json.loads(line)
                if (out / "glb" / f"{r['i']:03d}.glb").exists(): prev[r["i"]] = r
    # A: obrázky (fronta 3 úloh); při --resume jen chybějící
    t0 = time.time(); pend = []
    for i, (m, k) in enumerate(combos):
        if i in prev: continue
        pend.append((i, m, k, call("/prompt", {"prompt": img_wf(FRAME.format(m=m, k=k), 1000 + i, f"uc8/img_{i:03d}")})["prompt_id"]))
    for i, m, k, pid in pend:
        h = wait(pid); im = [x for o in h["outputs"].values() for x in o.get("images", [])][0]
        rows.append({"i": i, "predmet": f"{m} {k}", "img": f"{im['subfolder']}/{im['filename']}"})
    t_img = time.time() - t0; n_img = len(rows)
    print(f"A: {n_img} obrázků za {t_img:.0f} s", flush=True)
    # B: 3D (sekvenčně kvůli měření času na model; ComfyUI stejně běží po jednom)
    t1 = time.time()
    for r in rows:
        if r["i"] in prev: continue
        src = COMFY_OUT / r["img"]
        up = call("/upload/image", raw=_multipart(src), ctype=f"multipart/form-data; boundary={BOUND}")
        ts = time.time()
        h = wait(call("/prompt", {"prompt": mesh_wf(up["name"], 2000 + r["i"], f"uc8/glb_{r['i']:03d}")})["prompt_id"], poll=1)
        r["sec_3d"] = round(time.time() - ts, 1); r["ok"] = h["status"]["status_str"] == "success"
        glbs = sorted((COMFY_OUT / "uc8").glob(f"glb_{r['i']:03d}*.glb"))
        if r["ok"] and glbs:
            g = out / "glb" / f"{r['i']:03d}.glb"; g.write_bytes(glbs[-1].read_bytes())
            mesh = trimesh.load(g, force="mesh")
            r.update(tris=len(mesh.faces), watertight=bool(mesh.is_watertight), mb=round(g.stat().st_size / 1e6, 2))
        (out / "img" / f"{r['i']:03d}.png").write_bytes(src.read_bytes())
        print(json.dumps(r, ensure_ascii=False), flush=True)
    rows = sorted(rows + list(prev.values()), key=lambda r: r["i"])
    ok = [r for r in rows if r.get("tris")]
    s3d = [r["sec_3d"] for r in rows if "sec_3d" in r]
    tris = sorted(r["tris"] for r in ok)
    rep = {"predmetu": len(rows), "s_na_obrazek": round(t_img / max(n_img, 1), 2), "s_na_model": round(sum(s3d) / max(len(s3d), 1), 1), "navazano": len(prev),
           "modelu_ok": len(ok), "vodotesnych_%": round(100 * sum(r["watertight"] for r in ok) / max(len(ok), 1), 1),
           "trojuhelniku_median": tris[len(tris) // 2] if tris else None, "pod_4000_%": round(100 * sum(t <= 4000 for t in tris) / max(len(tris), 1), 1)}
    (out / "models.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1)); print(json.dumps(rep, ensure_ascii=False))


BOUND = "----uc8boundary"


def _multipart(path):
    head = (f"--{BOUND}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{path.name}\"\r\n"
            "Content-Type: image/png\r\n\r\n").encode()
    tail = f"\r\n--{BOUND}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{BOUND}--\r\n".encode()
    return head + path.read_bytes() + tail


if __name__ == "__main__":
    main()
