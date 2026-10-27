#!/usr/bin/env python3
"""UC11: analýza výsledků gridu → souhrn s 95% CI, Sobolova citlivost, surrogate pro /recommend.

    python analyze.py grid.yaml results.jsonl --out-dir DIR [--tag qwen3-30b-a3b] [--model-json recommend_model.json]

Výstupy (malé JSON, bez dat T-Mobile):
  DIR/UC11_grid_summary[_tag].json  — per buňka a per formát×populace: výnos/retail s 95% CI
                                       (bootstrap), konverze, Gini, LEAVE, emoce; nejlepší formát per
                                       zboží+populace s párovým srovnáním (stejní kupující napříč formáty)
  DIR/UC11_sobol[_tag].json          — Sobolovy indexy (SALib) nad surrogate + model-free 1. řád z gridu
  DIR/UC11_surrogate[_tag].json      — metriky surrogate (CV R², MAE) a feature importance
  --model-json                       — stromy gradient boostingu pro Go engine (RECOMMEND_MODEL_PATH)
"""
import argparse
import json
import math
import pathlib
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict

ROLES = ["GAMBLER", "CAUTIOUS", "INTERESTED", "RESELLER", "COLLECTOR", "SNIPER", "SOCIAL"]
FORMATS = ["DUTCH", "VICKREY", "PENNY"]
EMOTIONS = ["satisfied", "calm", "determined", "frustrated", "panicked"]


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return [None, None]
    if len(x) == 1:
        return [float(x[0]), float(x[0])]
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def gini(v):
    v = np.sort(np.asarray(v, float))
    if v.sum() <= 0:
        return 0.0
    n = len(v)
    return float((2 * np.sum(np.arange(1, n + 1) * v)) / (n * v.sum()) - (n + 1) / n)


def load(path):
    rows = []
    for line in open(path):
        r = json.loads(line)
        f = r.pop("factors")
        es = r.pop("emotion_share") or {}
        ro = r.pop("role_outlay") or {}
        r.update(f)
        for e in EMOTIONS:
            r[f"emo_{e}"] = es.get(e, 0.0)
        for k, v in ro.items():
            r[f"outlay_{k.lower()}"] = v
        rows.append(r)
    df = pd.DataFrame(rows)
    df["fallback_rate"] = df["fallbacks"] / df["decisions"].clip(lower=1)
    df["pair"] = df["population"] + "|" + df["value"] + "|" + df["elasticity"] + "|" + df["rep"].astype(str)
    return df


METRICS = ["revenue_ratio", "sold", "conversion", "gini_outlay", "leave_rate", "emotion_mean", "bidder_welfare",
           "efficiency", "winner_over_budget", "hit_max_ticks", "fallback_rate", "ticks", "decisions"]


def summarize_group(g):
    out = {"n": int(len(g))}
    for m in METRICS:
        x = g[m].astype(float)
        out[m] = round(float(x.mean()), 4)
    out["revenue_ratio_ci95"] = boot_ci(g["revenue_ratio"])
    out["emotion_mean_ci95"] = boot_ci(g["emotion_mean"])
    out["bidder_welfare_ci95"] = boot_ci(g["bidder_welfare"])
    out["emotion_share"] = {e: round(float(g[f"emo_{e}"].mean()), 4) for e in EMOTIONS}
    wins = g["winner_role"].fillna("").value_counts().to_dict()
    out["wins_by_role"] = {k: int(v) for k, v in wins.items() if k}
    return out


def win_gini(g, shares):
    """Gini „kdo vyhrává“: výhry per role normalizované podílem role v populaci (1 = férové)."""
    wins = g["winner_role"].fillna("").value_counts()
    vals = []
    for r in ROLES:
        s = shares.get(r, 0)
        if s > 0:
            vals.append(wins.get(r, 0) / s)
    return round(gini(vals), 4) if sum(vals) > 0 else None


def paired_best(df, keys=("value", "elasticity", "population")):
    """Per zboží+populace: formáty seřazené podle výnosu; rozdíl vítěz − druhý s párovým bootstrapem
    přes stejné páry kupujících (common random numbers)."""
    res = []
    for k, g in df.groupby(list(keys)):
        piv = g.pivot_table(index="pair", columns="format", values="revenue_ratio")
        means = piv.mean().sort_values(ascending=False)
        item = dict(zip(keys, k))
        item["ranking"] = [{"format": f, "revenue_ratio": round(float(m), 4),
                            "ci95": boot_ci(piv[f].dropna())} for f, m in means.items()]
        if len(means) >= 2:
            a, b = means.index[0], means.index[1]
            d = (piv[a] - piv[b]).dropna()
            ci = boot_ci(d)
            item["best"] = a
            item["margin_vs_second"] = round(float(d.mean()), 4) if len(d) else None
            item["margin_ci95"] = ci
            item["significant"] = bool(ci[0] is not None and ci[0] > 0)
        res.append(item)
    return res


# ---------- surrogate ----------

PARAM_FEATS = ["p_start_mult", "p_step_pct", "p_tick_s", "p_reserve_mult", "p_bid_fee_pct",
               "p_increment_pct", "p_timer_s"]


def features(df, grid):
    X = pd.DataFrame(index=df.index)
    for f in FORMATS:
        X[f"fmt_{f.lower()}"] = (df["format"] == f).astype(float)
    for r in ROLES:
        X[f"share_{r.lower()}"] = df.get(f"share_{r.lower()}", 0.0)
    X["log10_retail"] = np.log10(df["retail"].astype(float))
    el = grid["goods"]["elasticity"]
    X["elasticity_level"] = df["elasticity"].map(lambda e: el[e]["level"])
    X["elasticity_spread"] = df["elasticity"].map(lambda e: el[e]["spread"])
    for p in PARAM_FEATS:
        X[p] = df[p].astype(float).fillna(0.0) if p in df else 0.0
    return X


def fit_gbr(X, y, groups, loss="squared_error", alpha=0.5, cv=True):
    kw = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)
    if loss == "quantile":
        kw.update(loss="quantile", alpha=alpha)
    m = GradientBoostingRegressor(**kw)
    metrics = {}
    if cv:
        k = min(5, len(set(groups)))
        pred = cross_val_predict(GradientBoostingRegressor(**kw), X, y, groups=groups, cv=GroupKFold(k))
        ss = ((y - y.mean()) ** 2).sum()
        metrics = {"cv_r2": round(float(1 - ((y - pred) ** 2).sum() / ss), 4) if ss > 0 else None,
                   "cv_mae": round(float(np.abs(y - pred).mean()), 4), "cv": f"GroupKFold({k}) po buňkách"}
    m.fit(X, y)
    return m, metrics


def export_gbr(m):
    trees = []
    for est in m.estimators_[:, 0]:
        t = est.tree_
        trees.append({"f": t.feature.tolist(), "t": [round(float(v), 6) for v in t.threshold],
                      "l": t.children_left.tolist(), "r": t.children_right.tolist(),
                      "v": [round(float(v), 6) for v in t.value[:, 0, 0]]})
    init = float(np.ravel(m.init_.predict(np.zeros((1, m.n_features_in_))))[0]) if hasattr(m.init_, "predict") else 0.0
    return {"init": init, "lr": m.learning_rate, "trees": trees}


def predict_exported(e, X):
    """Stejný výpočet jako Go evaluátor — kontrola parity exportu."""
    out = np.full(len(X), e["init"])
    for t in e["trees"]:
        for i, x in enumerate(X):
            n = 0
            while t["l"][n] != -1:
                n = t["l"][n] if x[t["f"][n]] <= t["t"][n] else t["r"][n]
            out[i] += e["lr"] * t["v"][n]
    return out


# ---------- Sobol ----------

def sobol(model, grid, feat_names, fmt=None, n=1024):
    from SALib.analyze import sobol as sa
    from SALib.sample import sobol as ss

    pops = list(grid["populations"])
    vals = list(grid["goods"]["value"])
    els = list(grid["goods"]["elasticity"])
    fmts = [fmt] if fmt else FORMATS
    pranges = {f: {k: v for k, v in grid["formats"][f].items() if isinstance(v, list) and len(v) == 2} for f in FORMATS}
    names = (["format"] if not fmt else []) + ["population", "value", "elasticity"]
    nparam = max(len(pranges[f]) for f in fmts)
    pnames = list(pranges[fmt]) if fmt else [f"param_{i+1}" for i in range(nparam)]
    names += pnames
    problem = {"num_vars": len(names), "names": names, "bounds": [[0, 1]] * len(names)}
    U = ss.sample(problem, n, calc_second_order=False, seed=0)

    rows = []
    for u in U:
        i = 0
        f = fmts[min(int(u[i] * len(fmts)), len(fmts) - 1)] if not fmt else fmt
        if not fmt:
            i += 1
        pop = pops[min(int(u[i] * len(pops)), len(pops) - 1)]
        val = vals[min(int(u[i + 1] * len(vals)), len(vals) - 1)]
        el = els[min(int(u[i + 2] * len(els)), len(els) - 1)]
        i += 3
        d = {"format": f, "retail": float(grid["goods"]["value"][val]), "elasticity": el}
        tot = sum(grid["populations"][pop].values())
        for r in ROLES:
            d[f"share_{r.lower()}"] = grid["populations"][pop].get(r, 0) / tot
        for j, (k, (lo, hi)) in enumerate(pranges[f].items()):
            d[f"p_{k}"] = lo + u[i + j] * (hi - lo)
        rows.append(d)
    X = features(pd.DataFrame(rows), grid)[feat_names].values
    Y = model.predict(X)
    S = sa.analyze(problem, Y, calc_second_order=False, seed=0)
    return {nm: {"S1": round(float(S["S1"][k]), 4), "S1_conf": round(float(S["S1_conf"][k]), 4),
                 "ST": round(float(S["ST"][k]), 4), "ST_conf": round(float(S["ST_conf"][k]), 4)}
            for k, nm in enumerate(names)}


def first_order_raw(df, y, factors):
    """Model-free: Var(E[Y|faktor]) / Var(Y) přímo z gridu (diskrétní faktory)."""
    v = df[y].var()
    return {f: round(float(df.groupby(f)[y].mean().reindex(df[f]).values.var() / v), 4) if v > 0 else None
            for f in factors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("results")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--model-json")
    a = ap.parse_args()
    grid = yaml.safe_load(open(a.grid))
    df = load(a.results)
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sfx = f"_{a.tag}" if a.tag else ""
    model_name = df["model"].iloc[0] if len(df) else None

    # --- souhrn ---
    shares = {p: {r: c / sum(m.values()) for r, c in m.items()} for p, m in grid["populations"].items()}
    by_cell = {c: summarize_group(g) for c, g in df.groupby("cell")}
    by_fp = {}
    for (f, p), g in df.groupby(["format", "population"]):
        s = summarize_group(g)
        s["win_gini_roles"] = win_gini(g, shares[p])
        by_fp[f"{f}|{p}"] = s
    by_fmt = {f: summarize_group(g) for f, g in df.groupby("format")}
    by_fv = {f"{f}|{v}": summarize_group(g) for (f, v), g in df.groupby(["format", "value"])}
    summary = {
        "model": model_name, "runs": int(len(df)), "cells": int(df["cell"].nunique()),
        "cells_expected": len(FORMATS) * len(grid["populations"]) * len(grid["goods"]["value"]) * len(grid["goods"]["elasticity"]),
        "metric_notes": {
            "revenue_ratio": "(finální cena + všechny bid fees) / retail cena",
            "conversion": "podíl bidderů, kteří aspoň jednou zaplatili (bid/fee)",
            "gini_outlay": "Gini útrat (cena + fees) přes bidery v běhu",
            "win_gini_roles": "Gini výher per role normalizovaných podílem role v populaci",
            "emotion_mean": "satisfied +1, calm/determined 0, frustrated −0,5, panicked −1 (engine mění emoci jen při "
                            "platbě/odchodu → v Dutch/Vickrey ≈ 1/16 = jen vítěz; slabý proxy)",
            "bidder_welfare": "průměrný čistý výsledek bidderů / retail (vítěz: valuace − cena − fees, ostatní: −fees)",
            "hit_max_ticks": "podíl běhů useknutých stropem ticků (penny) → výnos je dolní odhad",
        },
        "by_format": by_fmt, "by_format_value": by_fv, "by_format_population": by_fp,
        "best_format": paired_best(df), "by_cell": by_cell,
        "throughput": {
            "decisions": int(df["decisions"].sum()), "prompt_tokens": int(df["prompt_tokens"].sum()),
            "completion_tokens": int(df["completion_tokens"].sum()),
            "fallback_rate": round(float(df["fallbacks"].sum() / max(1, df["decisions"].sum())), 4),
        },
    }
    (out / f"UC11_grid_summary{sfx}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))

    # --- surrogate ---
    X = features(df, grid)
    feat_names = list(X.columns)
    groups = df["cell"].values
    y = df["revenue_ratio"].values.astype(float)
    m_mean, met_mean = fit_gbr(X.values, y, groups)
    m_q05, _ = fit_gbr(X.values, y, groups, loss="quantile", alpha=0.05, cv=False)
    m_q95, _ = fit_gbr(X.values, y, groups, loss="quantile", alpha=0.95, cv=False)
    ye = df["bidder_welfare"].values.astype(float)
    m_emo, met_emo = fit_gbr(X.values, ye, groups)
    ys = df["sold"].astype(float).values
    m_sold, met_sold = fit_gbr(X.values, ys, groups)
    sur = {
        "model": model_name, "features": feat_names,
        "targets": {"revenue_ratio": met_mean, "bidder_welfare": met_emo, "sold": met_sold},
        "importance_revenue": dict(sorted(((f, round(float(v), 4)) for f, v in zip(feat_names, m_mean.feature_importances_)),
                                          key=lambda t: -t[1])),
        "gbr": "sklearn GradientBoostingRegressor(n_estimators=300, max_depth=3, lr=0.05, subsample=0.8)",
    }

    # --- Sobol ---
    sob = {"model": model_name, "method": "SALib Sobol (Saltelli, N=1024) nad surrogate revenue_ratio; "
                                         "diskrétní faktory (formát, populace, zboží) jako uniformní [0,1) → úroveň",
           "global": sobol(m_mean, grid, feat_names),
           "per_format": {f: sobol(m_mean, grid, feat_names, fmt=f) for f in FORMATS if (df["format"] == f).any()},
           "raw_first_order": {t: first_order_raw(df, t, ["format", "population", "value", "elasticity"])
                               for t in ["revenue_ratio", "bidder_welfare", "emotion_mean", "conversion", "leave_rate"]}}
    (out / f"UC11_sobol{sfx}.json").write_text(json.dumps(sob, ensure_ascii=False, indent=1))

    if a.model_json:
        exp = {"version": 1, "model": model_name, "runs": int(len(df)), "n_bidders": grid["n_bidders"],
               "features": feat_names,
               "models": {"revenue_mean": export_gbr(m_mean), "revenue_q05": export_gbr(m_q05),
                          "revenue_q95": export_gbr(m_q95), "bidder_welfare": export_gbr(m_emo),
                          "sold_prob": export_gbr(m_sold)},
               "populations": shares,
               "goods": {"value": grid["goods"]["value"], "elasticity": grid["goods"]["elasticity"]},
               "param_ranges": {f: {k: v for k, v in grid["formats"][f].items() if isinstance(v, list) and len(v) == 2}
                                for f in FORMATS},
               "cells": {c: {"n": s["n"], "revenue_ratio": s["revenue_ratio"], "ci95": s["revenue_ratio_ci95"],
                             "emotion_mean": s["emotion_mean"], "bidder_welfare": s["bidder_welfare"],
                             "sold": s["sold"]} for c, s in by_cell.items()}}
        # parita exportu se sklearn
        xs = X.values[: min(50, len(X))]
        diff = float(np.abs(predict_exported(exp["models"]["revenue_mean"], xs) - m_mean.predict(xs)).max())
        sur["export_max_abs_diff"] = diff
        assert diff < 1e-3, diff
        pathlib.Path(a.model_json).write_text(json.dumps(exp, separators=(",", ":")))
        sur["model_json_bytes"] = pathlib.Path(a.model_json).stat().st_size
    (out / f"UC11_surrogate{sfx}.json").write_text(json.dumps(sur, ensure_ascii=False, indent=1))
    print(json.dumps({"runs": len(df), "cells": summary["cells"], "surrogate": sur["targets"],
                      "sobol_global_ST": {k: v["ST"] for k, v in sob["global"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
