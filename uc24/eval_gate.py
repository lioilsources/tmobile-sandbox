"""UC24 — sfxgate nad syntetickou sadou: detekce per vada, falešné poplachy na čistých, ms/klip.

  python eval_gate.py --manifest /cold/uc24/synth/manifest.jsonl --out ../bench/results/UC24_gate_synth.json \
         --dump /cold/uc24/gate_features.jsonl

detekce = verdikt ≠ pass; "správný kód" = mezi důvody je kód odpovídající vložené vadě.
Čas = jen gate (příznaky + phone sim + prahy) nad už načteným polem, 1 vlákno; načtení WAV zvlášť.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfxgate.gate import gate  # noqa: E402

EXPECT = {  # vložená vada → kódy, které ji správně pojmenují
    "rumble": {"rumble", "phone_drop", "phone_distortion"},
    "clipping": {"clipping", "true_peak", "phone_distortion"},
    "hiss": {"hiss"}, "haze": {"hiss"}, "clicks": {"clicks"},
    "dc_offset": {"dc_offset"}, "lead_silence": {"lead_silence"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/cold/uc24/synth/manifest.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None, help="jsonl s příznaky+verdiktem per klip (pro krok 3)")
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.manifest) if '"mismatch"' not in line]
    stats = defaultdict(lambda: defaultdict(lambda: {"n": 0, "det": 0, "code": 0, "reject": 0}))
    ms, load_ms = [], []
    dump = open(a.dump, "w") if a.dump else None
    for r in rows:
        t0 = time.perf_counter()
        x, sr = sf.read(r["path"], dtype="float32")
        load_ms.append((time.perf_counter() - t0) * 1000)
        g = gate(x, sr, r["kind"])
        ms.append(g["ms"])
        codes = {q["code"] for q in g["reasons"]}
        key = r["defect"] if r["defect"] == "clean" else f'{r["defect"]}.{r["severity"]}'
        for grp in (f'{r["split"]}', f'{r["split"]}.{r["kind"]}'):
            s = stats[grp][key]
            s["n"] += 1
            s["det"] += g["verdict"] != "pass"
            s["reject"] += g["verdict"] == "reject"
            s["code"] += bool(codes & EXPECT.get(r["defect"], set()))
        if dump:
            dump.write(json.dumps({"id": r["id"], **{k: r[k] for k in ("kind", "type", "defect", "severity", "split")},
                                   "verdict": g["verdict"], "codes": sorted(codes), "features": g["features"]}) + "\n")
    res = {}
    for grp, d in sorted(stats.items()):
        res[grp] = {}
        for key, s in sorted(d.items()):
            n = s["n"]
            res[grp][key] = {"n": n, "flagged": round(s["det"] / n, 3), "reject": round(s["reject"] / n, 3),
                             **({} if key == "clean" else {"right_code": round(s["code"] / n, 3)})}
    ms_a = np.array(ms)
    out = {
        "uc": "UC24", "what": "sfxgate krok 1 (ruční prahy) nad syntetickou sadou se známými vadami",
        "n_clips": len(rows),
        "timing_ms_per_clip": {"median": round(float(np.median(ms_a)), 2), "p95": round(float(np.percentile(ms_a, 95)), 2),
                               "max": round(float(ms_a.max()), 2), "load_wav_median": round(float(np.median(load_ms)), 2),
                               "note": "1 vlákno (OMP=1), VM 8 vCPU; sfx 3 s, ambient/hudba 8 s, 44.1 kHz mono"},
        "timing_by_kind_median_ms": {k: round(float(np.median([m for m, r in zip(ms, rows) if r["kind"] == k])), 2)
                                     for k in sorted({r["kind"] for r in rows})},
        "results": res,
    }
    js = json.dumps(out, indent=1, ensure_ascii=False)
    if a.out:
        open(a.out, "w").write(js)
    print(js)


if __name__ == "__main__":
    main()
