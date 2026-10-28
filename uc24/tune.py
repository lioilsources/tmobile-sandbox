"""UC24 — pomůcka pro ruční prahy: percentily (p10/p50/p90) příznaků per druh × vada na dev části.

  python tune.py /cold/uc24/gate_dump.jsonl [feature ...]
"""
import json
import sys
from collections import Counter

import numpy as np

R = [json.loads(line) for line in open(sys.argv[1])]
R = [r for r in R if r["split"] == "dev"]
FEATS = sys.argv[2:] or ["floor_air_db", "floor_high_db", "flat_air", "flat_high", "hf_ratio_8k", "low_ratio_250",
                         "sub_ratio_80", "phone_dist_db", "phone_drop_db", "clicks_per_s", "lead_silence_s", "true_peak_dbtp", "clip_frac"]
DEF = ["clean"] + [f"{d}.{s}" for d in ["rumble", "clipping", "hiss", "haze", "clicks", "dc_offset", "lead_silence"] for s in ("mild", "strong")]
key = lambda r: r["defect"] if r["defect"] == "clean" else f'{r["defect"]}.{r["severity"]}'  # noqa: E731
for kind in ["sfx", "ambient", "music"]:
    print(f"===== {kind}")
    c = Counter((r["type"], tuple(r["codes"])) for r in R if r["kind"] == kind and r["defect"] == "clean" and r["codes"])
    print("  FP na čistých:", dict(c))
    for f in FEATS:
        print(f"  {f}")
        for d in DEF:
            v = [r["features"][f] for r in R if r["kind"] == kind and key(r) == d]
            if v:
                print(f"     {d:20s} {np.percentile(v, 10):9.4g} {np.median(v):9.4g} {np.percentile(v, 90):9.4g}   min {min(v):.4g} max {max(v):.4g}")
