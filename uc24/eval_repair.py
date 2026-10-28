"""UC24 krok 5 — oprava hraničních klipů (verdikt "repair") + CLAP non-regrese, na test části syntetické sady.

  python eval_repair.py --limit 300 --out ../bench/results/UC24_repair.json

Pro každý klip s verdiktem repair: řetězec repair.clean → znovu gate → CLAP(laion/larger_clap_general)
k popisu před/po; přijato = projde gate a CLAP neklesl o > 0.05. Výstup per vada: kolik oprav prošlo,
medián ΔCLAP, ms/klip.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfxgate.gate import gate  # noqa: E402
from sfxgate.repair import repair  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/cold/uc24/synth/manifest.jsonl")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--clap", default="laion/larger_clap_general")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    from sfxgate.clap import Clap
    clap = Clap(a.clap, "cpu").score
    rows = [json.loads(line) for line in open(a.manifest)]
    rows = [r for r in rows if r["split"] == "test" and r["defect"] != "mismatch"]
    rng = np.random.default_rng(24)
    rng.shuffle(rows)
    st = defaultdict(lambda: {"n": 0, "accepted": 0, "gate_pass": 0, "content_ok": 0, "dclap": [], "ms": []})
    n = 0
    for r in rows:
        if n >= a.limit:
            break
        x, sr = sf.read(r["path"], dtype="float32")
        g = gate(x, sr, r["kind"])
        if g["verdict"] != "repair":
            continue
        n += 1
        t0 = time.perf_counter()
        rp = repair(x, sr, r["kind"], r["prompt"], clap, first=g)
        key = r["defect"] if r["defect"] == "clean" else f'{r["defect"]}.{r["severity"]}'
        s = st[key]
        s["n"] += 1
        s["accepted"] += rp["accepted"]
        s["gate_pass"] += rp["gate"]["pass"]
        s["content_ok"] += rp["content_ok"]
        s["dclap"].append(rp["clap_after"] - rp["clap_before"])
        s["ms"].append((time.perf_counter() - t0) * 1000)
    out = {"uc": "UC24", "what": "krok 5: oprava verdiktu repair + CLAP non-regrese (max pokles 0.05)", "n": n, "by_defect": {}}
    tot = {"n": 0, "accepted": 0}
    for k, s in sorted(st.items()):
        out["by_defect"][k] = {"n": s["n"], "accepted": round(s["accepted"] / s["n"], 3), "gate_pass": round(s["gate_pass"] / s["n"], 3),
                               "content_ok": round(s["content_ok"] / s["n"], 3), "dclap_median": round(float(np.median(s["dclap"])), 4),
                               "dclap_min": round(float(np.min(s["dclap"])), 4), "ms_median_incl_clap": round(float(np.median(s["ms"])))}
        tot["n"] += s["n"]
        tot["accepted"] += s["accepted"]
    out["accepted_total"] = round(tot["accepted"] / max(1, tot["n"]), 3)
    js = json.dumps(out, indent=1, ensure_ascii=False)
    if a.out:
        open(a.out, "w").write(js)
    print(js)


if __name__ == "__main__":
    main()
