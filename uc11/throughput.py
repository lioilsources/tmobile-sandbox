#!/usr/bin/env python3
"""UC11: souhrn propustnosti gridu z logů run_grid.sh → UC11_throughput.json.

    python throughput.py --logs logs --labels grid30b_core grid30b_ext --results runs/grid30b_results.jsonl \
        --out UC11_throughput.json [--kc-per-h 55] [--api-in-usd 0.15 --api-out-usd 0.60 --czk-per-usd 23]

Zdroje: <label>_stats.json (grid runner), <label>_gpu_summary.txt (bench/summarize.py nad nvidia-smi
dmon), <label>_vllm_metrics.csv (vzorky /metrics po 10 s: běžící requesty, KV, tokeny).
Ceny API jsou PARAMETRY (předpoklad, ověřit aktuální ceník) — jen pro řádové srovnání Kč/simulace.
"""
import argparse
import csv
import json
import pathlib
import statistics


def vllm_stats(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({k: float(v) for k, v in r.items()})
            except (TypeError, ValueError):
                continue
    if len(rows) < 3:
        return {}
    t = rows[-1]["ts"] - rows[0]["ts"]
    busy = [r for r in rows if r["running"] > 0]
    return {
        "samples": len(rows), "span_s": round(t),
        "running_avg": round(statistics.mean(r["running"] for r in busy), 1) if busy else 0,
        "running_p50": statistics.median(r["running"] for r in busy) if busy else 0,
        "share_saturated_ge_240": round(sum(r["running"] >= 240 for r in rows) / len(rows), 3),
        "kv_usage_avg": round(statistics.mean(r["kv_usage"] for r in busy), 3) if busy else 0,
        "gen_tok_per_s": round((rows[-1]["gen_tokens_total"] - rows[0]["gen_tokens_total"]) / t),
        "prompt_tok_per_s": round((rows[-1]["prompt_tokens_total"] - rows[0]["prompt_tokens_total"]) / t),
        "preemptions": rows[-1]["preemptions"] - rows[0]["preemptions"],
        "note": "vLLM počítadla jsou globální (zahrnují případné jiné klienty téhož serveru)",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--results")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kc-per-h", type=float, default=55.0, help="předpokládaný náklad VM v Kč/h")
    ap.add_argument("--api-in-usd", type=float, default=0.15, help="USD / 1M input tokenů (předpoklad)")
    ap.add_argument("--api-out-usd", type=float, default=0.60, help="USD / 1M output tokenů (předpoklad)")
    ap.add_argument("--czk-per-usd", type=float, default=23.0)
    ap.add_argument("--extra", nargs="*", default=[], help="další JSON soubory k přiložení (A/B apod.)")
    a = ap.parse_args()
    L = pathlib.Path(a.logs)
    out = {"phases": {}}
    for lab in a.labels:
        ph = {}
        p = L / f"{lab}_stats.json"
        if p.exists():
            ph["runner"] = json.loads(p.read_text())
        p = L / f"{lab}_gpu_summary.txt"
        if p.exists():
            for line in p.read_text().splitlines():
                if line.startswith("{"):
                    g = json.loads(line)
                    ph["gpu"] = {k: g.get(k) for k in ("sekund", "gpu_util_avg", "gpu_util_p95", "gpu_util_idle_podil",
                                                       "power_avg_w", "energie_wh", "vram_peak_mib")}
        p = L / f"{lab}_vllm_metrics.csv"
        if p.exists():
            ph["vllm"] = vllm_stats(p)
        out["phases"][lab] = ph

    if a.results:
        rows = [json.loads(l) for l in open(a.results)]
        dec = sum(r["decisions"] for r in rows)
        pt = sum(r["prompt_tokens"] for r in rows)
        ct = sum(r["completion_tokens"] for r in rows)
        by_fmt = {}
        for r in rows:
            d = by_fmt.setdefault(r["format"], {"runs": 0, "decisions": 0, "prompt_tokens": 0, "completion_tokens": 0, "wall_s": 0.0})
            d["runs"] += 1
            d["decisions"] += r["decisions"]
            d["prompt_tokens"] += r["prompt_tokens"]
            d["completion_tokens"] += r["completion_tokens"]
            d["wall_s"] += r["wall_s"]
        for d in by_fmt.values():
            n = d["runs"]
            d.update(decisions_per_run=round(d["decisions"] / n), tokens_per_run=round((d["prompt_tokens"] + d["completion_tokens"]) / n),
                     wall_s_per_run=round(d["wall_s"] / n, 1))
            d["api_kc_per_run"] = round((d["prompt_tokens"] * a.api_in_usd + d["completion_tokens"] * a.api_out_usd) / 1e6 / n * a.czk_per_usd, 3)
        out["totals"] = {"runs": len(rows), "decisions": dec, "prompt_tokens": pt, "completion_tokens": ct,
                         "decisions_per_run": round(dec / max(1, len(rows)), 1), "by_format": by_fmt}
        # throughput přes všechny fáze (stěnový čas z runneru)
        el = sum(ph.get("runner", {}).get("elapsed_s", 0) for ph in out["phases"].values())
        done = sum(ph.get("runner", {}).get("done", 0) for ph in out["phases"].values())
        if not el:
            # průběžný snímek (runner ještě neskončil): čas z vzorků vLLM metrik
            el = sum(ph.get("vllm", {}).get("span_s", 0) for ph in out["phases"].values())
            done = len(rows)
            out["totals"]["interim"] = True
        if el:
            sph = done / el * 3600
            out["totals"].update(sims_per_h=round(sph, 1), elapsed_h=round(el / 3600, 2),
                                 decisions_per_s=round(dec / el, 1) if done == len(rows) else None)
            api_total = (pt * a.api_in_usd + ct * a.api_out_usd) / 1e6 * a.czk_per_usd
            out["cost"] = {
                "kc_per_h_vm": a.kc_per_h, "kc_per_sim_l40s": round(a.kc_per_h / sph, 3),
                "kc_per_sim_api": round(api_total / len(rows), 3),
                "api_assumption": f"{a.api_in_usd} USD/1M in, {a.api_out_usd} USD/1M out, {a.czk_per_usd} Kč/USD "
                                  "(třída malého komerčního modelu; ověřit aktuální ceník)",
            }
    for p in a.extra:
        out.setdefault("extra", {})[pathlib.Path(p).stem] = json.loads(pathlib.Path(p).read_text())
    pathlib.Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out.get("totals", {}), ensure_ascii=False)[:2000])
    print(json.dumps(out.get("cost", {}), ensure_ascii=False))


if __name__ == "__main__":
    main()
