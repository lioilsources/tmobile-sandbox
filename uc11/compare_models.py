#!/usr/bin/env python3
"""UC11 krok 3: porovnání LLM modelů na stejných specifikacích (párově přes run_id).

    python compare_models.py results_A.jsonl results_B.jsonl [...] --out UC11_model_vs_model.json

Otázka z plánu: kolik „iracionality“ je vlastnost modelu, ne role. Pro každý model:
výnos/retail per formát a per role (útrata), a shoda „nejlepšího formátu“ per zboží+populace.
Rozcházejí-li se modely, je závěr nestabilní → reportovat jako nález.
"""
import argparse
import itertools
import json
import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from analyze import ROLES, boot_ci, load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dfs = {}
    for p in a.results:
        d = load(p)
        dfs[d["model"].iloc[0]] = d
    common = set.intersection(*(set(d["run_id"]) for d in dfs.values()))
    dfs = {m: d[d["run_id"].isin(common)].set_index("run_id").sort_index() for m, d in dfs.items()}
    out = {"models": list(dfs), "common_runs": len(common), "per_model": {}, "pairs": {}}

    best = {}
    for m, d in dfs.items():
        pm = {"fallback_rate": round(float(d["fallbacks"].sum() / max(1, d["decisions"].sum())), 4),
              "by_format": {}, "role_outlay": {}, "leave_rate": round(float(d["leave_rate"].mean()), 4)}
        for f, g in d.groupby("format"):
            pm["by_format"][f] = {"revenue_ratio": round(float(g["revenue_ratio"].mean()), 4),
                                  "ci95": boot_ci(g["revenue_ratio"]), "sold": round(float(g["sold"].mean()), 4),
                                  "emotion_mean": round(float(g["emotion_mean"].mean()), 4),
                                  "conversion": round(float(g["conversion"].mean()), 4)}
        for r in ROLES:
            c = f"outlay_{r.lower()}"
            if c in d:
                pm["role_outlay"][r] = {f: round(float(g[c].mean()), 4) for f, g in d.groupby("format")}
        # nejlepší formát per zboží+populace (průměr přes replikace a parametry)
        piv = d.groupby(["value", "elasticity", "population", "format"])["revenue_ratio"].mean().unstack()
        best[m] = piv.idxmax(axis=1)
        pm["best_format_counts"] = best[m].value_counts().to_dict()
        out["per_model"][m] = pm

    for a_, b_ in itertools.combinations(dfs, 2):
        da, db = dfs[a_], dfs[b_]
        diff = (da["revenue_ratio"] - db["revenue_ratio"])
        pair = {"best_format_agreement": round(float((best[a_] == best[b_]).mean()), 4),
                "revenue_ratio_corr": round(float(np.corrcoef(da["revenue_ratio"], db["revenue_ratio"])[0, 1]), 4),
                "by_format_diff": {}}
        for f in sorted(da["format"].unique()):
            msk = da["format"] == f
            pair["by_format_diff"][f] = {"mean_diff": round(float(diff[msk].mean()), 4), "ci95": boot_ci(diff[msk])}
        out["pairs"][f"{a_} vs {b_}"] = pair
    pathlib.Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_model"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
