#!/usr/bin/env python3
"""UC3: jak často je tah s nejvyšší policy lc0 (bez prohledávání) řešením úlohy, podle ratingu úlohy (report UC3).

  rating_stats.py   (v /srv/work/uc3: positions.jsonl + runs/full-bt4/analysis.s*.jsonl)
"""
import collections, glob, json

rat = {}
for l in open("positions.jsonl"):
    p = json.loads(l); rat[p["id"]] = p.get("rating")
b = collections.defaultdict(lambda: [0, 0, 0.0]); n = top = 0
for f in glob.glob("runs/full-bt4/analysis.s*.jsonl"):
    for l in open(f):
        a = json.loads(l); r = rat.get(a["id"])
        if r is None: continue
        x = b[min(r // 250 * 250, 3000)]; x[0] += 1; x[1] += a["rank_sol"] == 0; x[2] += a["p_sol"]
        n += 1; top += a["rank_sol"] == 0
print("pozic", n, "top1 %", round(100 * top / n, 1))
for k in sorted(b):
    x = b[k]; print(k, x[0], round(100 * x[1] / x[0], 1), round(x[2] / x[0], 3))
