#!/usr/bin/env python3
"""UC25: souhrn běhu → bench/results/UC25_<tag>.json (jen čísla, žádné obrázky ani prompty).

  summarize.py --dir /srv/work/w2/uc25/out/cpu --tag cpu --out ~/tmobile-sandbox/bench/results

Čte výstupy taxo/ (wd14/nsfw/siglip/vlm/head .jsonl, *.err, stats.json, head/meta.json,
head_bench.json, bands_*.json, calib_guard.json) a počítá:
  - obrázků/h per hodnotitel (wall-clock z logu evaluátoru, včetně dekódování obrázků),
  - podíl unsure (rating, per uzel, obrázky s ≥1 unsure uzlem) a pásma,
  - shodu WD14 × SigLIP per uzel (kde odpověděli oba),
  - F1 hlavy vs VLM na held-out (pokud je VLM), F1 hodnotitelů vs zamýšlené atributy (kalibrační sada).
"""
import argparse
import json
import platform
import re
import subprocess
from collections import defaultdict
from pathlib import Path


def jl(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def jload(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def throughput(d: Path) -> dict:
    out = {}
    t = (d / "wd14.err").read_text() if (d / "wd14.err").exists() else ""
    m = re.search(r"wd14\[(\w+)\]: (\d+) images in ([\d.]+)s → (\d+) img/h", t)
    if m:
        out["wd14"] = {"provider": m.group(1), "images": int(m.group(2)), "wall_s": float(m.group(3)), "img_per_h": int(m.group(4))}
    t = (d / "nsfw.err").read_text() if (d / "nsfw.err").exists() else ""
    for m in re.finditer(r"(\w+)\[(\w+)\]: (\d+) images in ([\d.]+)s → (\d+) img/h", t):
        out[m.group(1)] = {"device": m.group(2), "images": int(m.group(3)), "wall_s": float(m.group(4)), "img_per_h": int(m.group(5))}
    for name in ("siglip", "vlm"):
        p = d / f"{name}.err"
        if p.exists():
            for line in p.read_text().splitlines():
                if line.startswith("{") and "img_per_h" in line:
                    out[name] = json.loads(line)
    for k, f in (("wd14", "wd14.err"), ("nsfw", "nsfw.err"), ("siglip", "siglip.err")):
        p = d / f
        if p.exists():
            m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", p.read_text())
            if m and k in out:
                out[k]["max_rss_mb"] = round(int(m.group(1)) / 1024)
    return out


def labels_by(recs):
    return {r["image"]: r["labels"] for r in recs}


def agreement(a: dict, b: dict) -> dict:
    """exact agreement per node where both answered (multi: Jaccard mean)."""
    acc = defaultdict(lambda: [0, 0.0])
    for iid, la in a.items():
        lb = b.get(iid)
        if not lb:
            continue
        for node, x in la.items():
            y = lb.get(node)
            if y is None:
                continue
            if isinstance(x["value"], list):
                sa, sb = set(x["value"]), set(y["value"])
                s = 1.0 if not (sa | sb) else len(sa & sb) / len(sa | sb)
            else:
                s = float(x["value"] == y["value"])
            acc[node][0] += 1
            acc[node][1] += s
    return {n: {"n": c, "agreement": round(s / c, 4)} for n, (c, s) in sorted(acc.items())}


def f1_vs(ref: dict, pred: dict, ids=None) -> dict:
    """macro-F1 per node of pred against ref (ref = VLM or intended); abstention = miss."""
    per = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # node → value → tp, fp, fn
    nimg = defaultdict(int)
    for iid, lr in ref.items():
        if ids is not None and iid not in ids:
            continue
        lp = pred.get(iid, {})
        for node, x in lr.items():
            if node == "rating":
                continue
            rv = set(x["value"]) if isinstance(x["value"], list) else {x["value"]}
            y = lp.get(node)
            pv = set() if y is None else (set(y["value"]) if isinstance(y["value"], list) else {y["value"]})
            nimg[node] += 1
            for v in rv | pv:
                c = per[node][v]
                if v in rv and v in pv:
                    c[0] += 1
                elif v in pv:
                    c[1] += 1
                else:
                    c[2] += 1
    out = {}
    for node, vals in sorted(per.items()):
        fs = [2 * tp / (2 * tp + fp + fn) for tp, fp, fn in vals.values() if tp + fn > 0]
        out[node] = {"n": nimg[node], "macro_f1": round(sum(fs) / len(fs), 4) if fs else None}
    return out


def mean_f1(d):
    xs = [v["macro_f1"] for v in d.values() if v["macro_f1"] is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def intended_labels(manifest):
    out = {}
    for r in manifest:
        lab = {}
        for k, v in r.get("intended", {}).items():
            if k != "rating":
                lab[k] = {"value": v}
        out[r["image"]] = lab
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-manifest", default="/cold/w2/uc25/calib/manifest.jsonl")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    d = Path(a.dir)

    wd14, siglip, vlm, head = (labels_by(jl(d / f"{n}.jsonl")) for n in ("wd14", "siglip", "vlm", "head"))
    res = {
        "uc": "UC25", "tag": a.tag, "host": platform.node(), "note": a.note,
        "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip(),
        "data": "obrázky vygenerované v sandboxu (UC10 Storyteller, UC7 oči, UC8 objekty, UC5/smoke ComfyUI); "
                "žádná uživatelská galerie; kalibrační sada jen pokud existuje",
        "throughput": throughput(d),
        "stats": jload(d / "stats.json"),
        "stats_2eval": jload(d / "stats_2eval.json"),
        "agreement_wd14_siglip": agreement(wd14, siglip),
    }
    hm = jload(d / "head" / "meta.json")
    if hm:
        res["head"] = {k: hm[k] for k in ("label_source", "train", "val_metrics", "val_temperature_mae", "dim")}
        res["head"]["bench"] = jload(d / "head_bench.json")
        if vlm:
            val = set(hm.get("val_ids", []))
            res["f1_vs_vlm_heldout"] = {
                "head": f1_vs(vlm, head, val), "wd14": f1_vs(vlm, wd14, val), "siglip": f1_vs(vlm, siglip, val)}
            res["f1_vs_vlm_heldout_mean"] = {k: mean_f1(v) for k, v in res["f1_vs_vlm_heldout"].items()}
    if vlm:
        res["agreement_wd14_vlm"] = agreement(wd14, vlm)
        res["agreement_siglip_vlm"] = agreement(siglip, vlm)
    cm = jl(a.calib_manifest)
    if cm:
        ref = intended_labels(cm)
        res["f1_vs_intended_calib"] = {n: f1_vs(ref, lab) for n, lab in
                                       (("wd14", wd14), ("siglip", siglip), ("vlm", vlm), ("head", head)) if lab}
        res["f1_vs_intended_calib_mean"] = {k: mean_f1(v) for k, v in res["f1_vs_intended_calib"].items()}
    for k in ("bands_pseudo", "bands_synth", "calib_guard"):
        v = jload(d / f"{k}.json")
        if v:
            if k == "calib_guard":
                v = {kk: vv for kk, vv in v.items() if kk != "deleted_ids"}
            res[k] = v
    out = Path(a.out).expanduser() / f"UC25_{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()
