#!/usr/bin/env python3
"""UC11: A/B dlouhý (2–3 věty reasoning) vs kompaktní (1 věta) prompt na stejných specifikacích.

    python ab_prompts.py long.jsonl compact.jsonl --out UC11_prompt_ab.json

Párově přes run_id: rozdíl výnosu/retail, shoda vítězné role, LEAVE, tokeny na rozhodnutí.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from analyze import boot_ci, load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("long")
    ap.add_argument("compact")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = load(a.long).set_index("run_id")
    C = load(a.compact).set_index("run_id")
    ids = L.index.intersection(C.index)
    L, C = L.loc[ids], C.loc[ids]
    res = {"runs_paired": int(len(ids)), "by_format": {}}
    for name, d in (("long", L), ("compact", C)):
        res[name] = {"completion_tokens_per_decision": round(float(d["completion_tokens"].sum() / d["decisions"].sum()), 1),
                     "prompt_tokens_per_decision": round(float(d["prompt_tokens"].sum() / d["decisions"].sum()), 1),
                     "fallback_rate": round(float(d["fallbacks"].sum() / d["decisions"].sum()), 4),
                     "mean_latency_ms": round(float((d["mean_latency_ms"] * d["decisions"]).sum() / d["decisions"].sum()))}
    for f in sorted(L["format"].unique()):
        m = L["format"] == f
        diff = (C.loc[m, "revenue_ratio"] - L.loc[m, "revenue_ratio"]).values
        res["by_format"][f] = {
            "n": int(m.sum()),
            "revenue_ratio_long": round(float(L.loc[m, "revenue_ratio"].mean()), 4),
            "revenue_ratio_compact": round(float(C.loc[m, "revenue_ratio"].mean()), 4),
            "diff_compact_minus_long": round(float(diff.mean()), 4), "diff_ci95": boot_ci(diff),
            "corr": round(float(np.corrcoef(L.loc[m, "revenue_ratio"], C.loc[m, "revenue_ratio"])[0, 1]), 4),
            "winner_role_agreement": round(float((L.loc[m, "winner_role"] == C.loc[m, "winner_role"]).mean()), 4),
            "leave_rate_long": round(float(L.loc[m, "leave_rate"].mean()), 4),
            "leave_rate_compact": round(float(C.loc[m, "leave_rate"].mean()), 4),
            "ticks_long": round(float(L.loc[m, "ticks"].mean()), 2),
            "ticks_compact": round(float(C.loc[m, "ticks"].mean()), 2),
        }
    pathlib.Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
