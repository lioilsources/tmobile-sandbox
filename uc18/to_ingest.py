#!/usr/bin/env python3
"""UC18 matrix → FineTuneGallery ingest manifests (one session per checkpoint × seed), the same payload the app
sends (server/ingest.go Manifest) — the gallery needs no change and does not know about the L40S.

  to_ingest.py --dir /cold/w2/uc18 --refs uc18/prompts/uc18_prompts.jsonl --out /cold/w2/uc18/ingest   # files only
  to_ingest.py … --push http://<gallery>:8080      # POST manifest → PUT needed blobs → finalize (user's decision)

Skipped by default: FLUX.1-dev rows (non-commercial licence; --include-restricted to override) and every image the
WD14 rating gate flagged (scores/wd14.jsonl nsfw_flag). prompt = the hand-written prose (what a user would type),
promptSent = what reached the checkpoint, translator = "uc18-native@v1" for the booru checkpoints.
"""
import argparse, json, os, sys, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_jsonl  # noqa: E402
from models import CKPTS  # noqa: E402


def call(url, method, data=None, ctype="application/json"):
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read() or b"{}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--push", default="")
    ap.add_argument("--include-restricted", action="store_true")
    a = ap.parse_args()
    refs = {r["id"]: r for r in read_jsonl(a.refs)}
    flagged = {s["id"] for s in read_jsonl(os.path.join(a.dir, "scores", "wd14.jsonl")) if s.get("nsfw_flag")}
    rows = {}
    for r in read_jsonl(os.path.join(a.dir, "manifest.jsonl")):
        if not r.get("error") and r.get("sha256"):
            rows[r["id"]] = r
    sessions = {}
    skipped = {"restricted": 0, "nsfw_flag": 0}
    for r in rows.values():
        if r.get("licence_restricted") and not a.include_restricted:
            skipped["restricted"] += 1
            continue
        if r["id"] in flagged:
            skipped["nsfw_flag"] += 1
            continue
        c = CKPTS[r["ckpt"]]
        sid = f"l40s-uc18-{r['ckpt']}-s{r['seed_idx']}"
        s = sessions.setdefault(sid, {"session": {"id": sid, "title": f"L40S UC18 matrix — {r['ckpt']} seed {r['seed_idx']}",
                                                  "modelId": c["gallery_id"], "updatedAt": "2026-10-29T00:00:00Z"},
                                      "nodes": [], "_paths": {}})
        nid = f"{sid}-{r['pid']}"
        s["nodes"].append({"id": nid, "parentId": None, "prompt": refs[r["pid"]]["prose"], "origin": "generated",
                           "modelId": c["gallery_id"], "styleId": r["style"], "promptSent": r["prompt"],
                           "translator": "uc18-native@v1" if c["dialect"] in ("pony", "illustrious") else None,
                           "seed": r["seed"], "negativePrompt": r["negative"], "width": 1024, "height": 1024,
                           "steps": r["steps"], "cfg": r["cfg"], "denoise": 1.0, "samplerName": r["sampler"],
                           "scheduler": r["scheduler"],
                           "images": [{"id": f"{nid}-0", "sha256": r["sha256"], "size": r["size"], "idx": 0}]})
        s["_paths"][r["sha256"]] = r["path"]
    os.makedirs(a.out, exist_ok=True)
    for sid, s in sessions.items():
        paths = s.pop("_paths")
        json.dump(s, open(os.path.join(a.out, sid + ".json"), "w"), ensure_ascii=False)
        if a.push:
            need = call(f"{a.push}/api/ingest/manifest", "POST", json.dumps(s).encode())["needed"]
            for sha in need:
                call(f"{a.push}/api/ingest/images/{sha}", "PUT", open(paths[sha], "rb").read(), "image/png")
            call(f"{a.push}/api/ingest/sessions/{sid}/finalize", "POST", b"{}")
            print(f"pushed {sid}: {len(s['nodes'])} nodes, {len(need)} new blobs", file=sys.stderr)
    print(json.dumps({"sessions": len(sessions), "nodes": sum(len(s["nodes"]) for s in sessions.values()), "skipped": skipped}))


if __name__ == "__main__":
    main()
