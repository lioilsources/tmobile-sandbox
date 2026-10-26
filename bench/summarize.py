#!/usr/bin/env python3
"""CSV z `nvidia-smi dmon -s pucvmt -o T` → avg/p95 GPU util, výkon, energie (Wh), čas → /srv/work/bench/results/<label>.json.
Volitelně --extra '{"stran": 120}' pro jednotky práce (přidá Wh/jednotku a jednotek/h).

  summarize.py <csv> --label uc1_qwen3vl8b [--extra '{"stran": 120}'] [--merge results.json]
"""
import argparse, json, pathlib, statistics

def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--label", required=True)
    ap.add_argument("--extra", default="{}"); ap.add_argument("--merge", help="JSON s dalšími metrikami (např. report z eval)")
    a = ap.parse_args(); cols = None; rows = []
    for line in open(a.csv):
        if line.startswith("#"):
            if cols is None: cols = line[1:].split()
            continue
        v = line.split()
        if cols and len(v) == len(cols): rows.append(dict(zip(cols, v)))
    num = lambda k: [float(r[k]) for r in rows if r.get(k, "-") not in ("-", "")]
    sm, pw, fb = num("sm"), num("pwr"), num("fb")
    secs = len(rows)
    res = {"label": a.label, "csv": a.csv, "sekund": secs, "gpu_util_avg": round(statistics.mean(sm), 1) if sm else None,
           "gpu_util_p95": pct(sm, .95), "gpu_util_idle_podil": round(sum(1 for x in sm if x < 10) / len(sm), 3) if sm else None,
           "power_avg_w": round(statistics.mean(pw), 1) if pw else None, "energie_wh": round(sum(pw) / 3600, 2),
           "vram_peak_mib": max(fb) if fb else None, "teplota_max": max(num("gtemp")) if num("gtemp") else None}
    extra = json.loads(a.extra)
    for unit, n in extra.items():
        if n: res[f"wh_na_{unit}"] = round(res["energie_wh"] / n, 4); res[f"{unit}_za_hodinu"] = round(n / secs * 3600, 1)
    res.update(extra)
    if a.merge: res["metriky"] = json.load(open(a.merge))
    out = pathlib.Path("/srv/work/bench/results"); out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.label}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "metriky"}, ensure_ascii=False))

if __name__ == "__main__": main()
