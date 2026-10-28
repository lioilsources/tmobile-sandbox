"""UC24 — ms/klip sfxgate na aktuálním stroji (1 vlákno), nad procedurálními klipy ze synth.py.

  OMP_NUM_THREADS=1 python bench_gate.py [--reps 50]
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth  # noqa: E402
from sfxgate.gate import gate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=50)
a = ap.parse_args()
out = {"machine": platform.machine(), "processor": platform.processor() or platform.platform(), "threads": os.environ.get("OMP_NUM_THREADS")}
for typ, dur in [("bell", 3.0), ("footsteps", 3.0), ("rain", 8.0), ("wind", 30.0)]:
    kind, fn, _ = synth.SOURCES[typ]
    r = np.random.default_rng(1)
    x = synth.normalize(fn(r, int(dur * synth.SR)), r).astype(np.float32)
    gate(x, synth.SR, kind)
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        gate(x, synth.SR, kind)
        ts.append((time.perf_counter() - t0) * 1000)
    out[f"{typ}_{dur:g}s"] = {"median_ms": round(float(np.median(ts)), 2), "p95_ms": round(float(np.percentile(ts, 95)), 2)}
print(json.dumps(out))
