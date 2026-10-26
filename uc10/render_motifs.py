#!/usr/bin/env python3
"""UC10: ilustrace motivů pro storyteller dávkou přes ComfyUI (FLUX.1-schnell fp8, 1024², 4 kroky).

Motivy: veřejné rag/data/cz_cards.json + cz_tpe_cards.json ({id, text_en}). Styly: sedm stylů
z infra/seed/models_styles.sql (prompt_prefix + motiv + prompt_suffix), --variants seedů na styl.
Varianta 0 má seed jako storyteller render-motifs (low 32 bitů contentkey.Seed(sha256("motif-card\\x1f"+id))),
takže composition sedí s existujícími kartami; styl seed nemění (MODELS_PLAN §0.1).
Fronta drží --depth úloh v ComfyUI, aby GPU nečekala na klienta.

  render_motifs.py --cards a.json b.json --styles storyteller/infra/seed/models_styles.sql --out /srv/work/uc10/styles
"""
import argparse, hashlib, io, json, pathlib, re, time, urllib.request
from PIL import Image

API = "http://127.0.0.1:8188"


def seed(motif_id, variant=0):
    h = hashlib.sha256(("motif-card\x1f" + motif_id + (f"#{variant}" if variant else "")).encode()).hexdigest()
    return (int(h[:16], 16) & 0x7FFFFFFFFFFFFFFF) & 0xFFFFFFFF


def load_styles(sql):
    """(id, prefix, suffix) ze seedu: každý styl je n-tice na řádcích id / name_i18n / modely / prefix / suffix / …"""
    t = open(sql).read()
    i = t.index("INSERT INTO styles"); t = t[i:t.index("ON CONFLICT", i)]
    out = []
    for block in t.split("\n  ('")[1:]:
        lines = block.splitlines()
        sid = lines[0].split("'")[0]
        prefix, suffix = (re.match(r"\s*'((?:[^']|'')*)'", l).group(1).replace("''", "'") for l in lines[3:5])
        out.append((sid, prefix, suffix))
    return out


def workflow(prompt, s):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "flux1-schnell-fp8.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": prompt}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": ""}},
        "4": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
              "latent_image": ["4", 0], "seed": s, "steps": 4, "cfg": 1.0, "sampler_name": "euler",
              "scheduler": "simple", "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "uc10/motif"}},
    }


def call(path, data=None):
    req = urllib.request.Request(API + path, json.dumps(data).encode() if data else None, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read() if path.startswith("/view") else json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", nargs="+", required=True); ap.add_argument("--styles", required=True)
    ap.add_argument("--variants", type=int, default=4); ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--depth", type=int, default=3)
    a = ap.parse_args()
    out = pathlib.Path(a.out); (out / "img").mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in (out / "img").glob("*.jpg")}
    motifs = {}
    for f in a.cards:
        for c in json.load(open(f)): motifs.setdefault(c["id"], c["text_en"])
    styles = load_styles(a.styles)
    # pořadí: varianta → styl → motiv, aby byla po přerušení hotová celá galerie stylů pro variantu 0
    todo = [(f"{mid}.{sid}.{v}", mid, pre + text + suf, v) for v in range(a.variants) for sid, pre, suf in styles
            for mid, text in sorted(motifs.items()) if f"{mid}.{sid}.{v}" not in have]
    if a.limit: todo = todo[: a.limit]
    print(f"motivů {len(motifs)}, stylů {len(styles)} ({', '.join(s[0] for s in styles)}), hotových {len(have)}, k vykreslení {len(todo)}", flush=True)
    mf = open(out / "manifest.jsonl", "a")
    pending, t0, done = {}, time.time(), 0
    it = iter(todo)
    while True:
        while len(pending) < a.depth:
            nxt = next(it, None)
            if nxt is None: break
            name, mid, prompt, v = nxt; s = seed(mid, v)
            pid = call("/prompt", {"prompt": workflow(prompt, s)})["prompt_id"]
            pending[pid] = (name, prompt, s, time.time())
        if not pending: break
        time.sleep(0.2)
        for pid in list(pending):
            h = call(f"/history/{pid}")
            if pid not in h: continue
            mid, prompt, s, ts = pending.pop(pid)
            imgs = [i for o in h[pid]["outputs"].values() for i in o.get("images", [])]
            ok = h[pid]["status"]["status_str"] == "success" and imgs
            if ok:
                i = imgs[0]
                raw = call(f"/view?filename={i['filename']}&subfolder={i['subfolder']}&type={i['type']}")
                Image.open(io.BytesIO(raw)).convert("RGB").save(out / "img" / f"{mid}.jpg", quality=88)
            mf.write(json.dumps({"id": mid, "ok": bool(ok), "seed": s, "prompt": prompt, "sec": round(time.time() - ts, 2)}) + "\n")
            done += 1
            if done % 100 == 0:
                el = time.time() - t0
                print(f"{done}/{len(todo)}  {done / el * 3600:.0f} obr/h", flush=True); mf.flush()
    el = time.time() - t0
    rep = {"obrazku": done, "sekund": round(el), "obr_za_hodinu": round(done / el * 3600) if el else 0}
    (out / "report.json").write_text(json.dumps(rep)); print(json.dumps(rep))


if __name__ == "__main__":
    main()
