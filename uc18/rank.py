#!/usr/bin/env python3
"""UC18 ranking: per-image metrics → Bradley–Terry + TrueSkill per (model, style) → kNN router + LOO evaluation.

  rank.py --dir /cold/w2/uc18 --prompts uc18/prompts/uc18_prompts.jsonl --out /cold/w2/uc18/rank.json \
          [--router-out /cold/w2/uc18/router.json] [--group-key ckpt|variant_ckpt]

Inputs in --dir: manifest.jsonl, scores/{pick,siglip,wd14,ir}.jsonl (any subset), scores/vlm_pairs.jsonl
(optional), scores/text_emb.npz (router; optional).

1. Per image: PickScore, SigLIP p, WD14 recall, ImageReward (whatever exists). Images with the WD14 rating
   gate (nsfw_flag) are excluded and counted per model.
2. Composite per (prompt, seed): each metric z-scored across the models of that prompt, then averaged →
   comparable across prompts of different difficulty.
3. Pairs: (a) metric pairs from the composite (|Δz| < 0.1 = tie); (b) VLM pairs (adherence), where an
   order-swapped re-ask that disagrees turns the pair into a tie (position-bias control).
4. Bradley–Terry (MM, Hunter 2004; tie = ½ win each; 0.5 pseudo-win vs. a virtual average player) per style and
   overall, 200× bootstrap over prompts → log-strength, 90 % CI, P(best). TrueSkill 1v1 with draws per
   (model, style), same pairs, 5 shuffled passes averaged.
5. Router prototype: utility(model | prompt) = kNN (k=15, SigLIP text embedding cosine) mean composite of the
   neighbours + style BT prior; leave-one-prompt-out: top-1 = oracle best, regret, vs. global-best /
   style-best / random baselines. p_best = smoothed share of neighbours where the model won.
"""
from __future__ import annotations

import argparse, json, math, os, random, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import limit_threads, read_jsonl  # noqa: E402

limit_threads(4)
import numpy as np  # noqa: E402

# composite uses SigLIP cosine, not sigmoid p: p is ≈ 0.000x for most image/prompt pairs and its rounding kills
# the per-prompt z-scores (seen in the smoke test); cos is monotone in p for a fixed text.
METRICS = ["pickscore", "siglip_cos", "wd14_recall", "imagereward"]


def load_scores(d, group_key):
    man = {}
    for r in read_jsonl(os.path.join(d, "manifest.jsonl")):
        if not r.get("error") and r.get("path"):
            man[r["id"]] = r
    rows = {i: {"id": i, "pid": r["pid"], "style": r.get("style", ""), "seed_idx": r["seed_idx"],
                "model": r["ckpt"] if group_key == "ckpt" else f"{r.get('variant', '')}:{r['ckpt']}",
                "ckpt": r["ckpt"], "variant": r.get("variant", ""), "exec_s": r.get("exec_s")} for i, r in man.items()}
    for f in ("pick", "siglip", "wd14", "ir"):
        for s in read_jsonl(os.path.join(d, "scores", f"{f}.jsonl")):
            if s.get("error") or s["id"] not in rows:
                continue
            row = rows[s["id"]]
            for k in METRICS + ["siglip_p", "nsfw_flag", "must_not_hits", "wd14_n_ref"]:
                if k in s:
                    row[k] = s[k]
            if s.get("style"):
                row["style"] = s["style"]
    return rows


def composite(rows):
    groups = defaultdict(list)
    for r in rows.values():
        if not r.get("nsfw_flag"):
            groups[(r["pid"], r["seed_idx"])].append(r)
    for g in groups.values():
        zs = defaultdict(list)
        for m in METRICS:
            vals = [r[m] for r in g if r.get(m) is not None]
            if len(vals) < 2:
                continue
            mu, sd = float(np.mean(vals)), float(np.std(vals)) or 1.0
            for r in g:
                if r.get(m) is not None:
                    zs[r["id"]].append((r[m] - mu) / sd)
        for r in g:
            r["composite"] = float(np.mean(zs[r["id"]])) if zs[r["id"]] else None
    return groups


def metric_pairs(groups, tie=0.1):
    out = []
    for (pid, s), g in groups.items():
        g = [r for r in g if r.get("composite") is not None]
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g[i], g[j]
                d = a["composite"] - b["composite"]
                out.append((pid, a["style"], a["model"], b["model"], 0.5 if abs(d) < tie else float(d > 0)))
    return out


def vlm_pairs(d, rows_by_pid_model):
    recs = [r for r in read_jsonl(os.path.join(d, "scores", "vlm_pairs.jsonl")) if not r.get("error")]
    base = {r["id"]: r for r in recs if not r.get("swapped")}
    swaps = {r["id"][: -len("/swap")]: r for r in recs if r.get("swapped")}
    out, agree, n_sw = [], 0, 0
    for i, r in base.items():
        a, b = sorted([r["first"], r["second"]])
        w = r.get("adherence_winner")
        if i in swaps:
            n_sw += 1
            w2 = swaps[i].get("adherence_winner")
            agree += (w == w2)
            if w != w2:
                w = "tie"
        res = 0.5 if w == "tie" else float(w == a)
        out.append((r["pid"], r.get("style", ""), a, b, res))
    return out, {"pairs": len(out), "swap_checked": n_sw, "position_consistency": round(agree / n_sw, 3) if n_sw else None}


def bradley_terry(pairs, models, iters=200, prior=0.5):
    idx = {m: i for i, m in enumerate(models)}
    n = len(models)
    W = np.full(n, prior)          # wins (incl. ½ ties) + pseudo-win vs virtual player of strength 1
    N = np.zeros((n, n))
    for _, _, a, b, res in pairs:
        i, j = idx[a], idx[b]
        W[i] += res
        W[j] += 1 - res
        N[i, j] += 1
        N[j, i] += 1
    p = np.ones(n)
    for _ in range(iters):
        denom = (N / (p[:, None] + p[None, :])).sum(1) + 2 * prior / (p + 1)
        p_new = W / denom
        p_new /= np.exp(np.mean(np.log(p_new)))
        if np.max(np.abs(p_new - p)) < 1e-9:
            p = p_new
            break
        p = p_new
    return {m: float(np.log(p[idx[m]])) for m in models}


def bt_bootstrap(pairs, models, B=200, seed=0):
    by_pid = defaultdict(list)
    for x in pairs:
        by_pid[x[0]].append(x)
    pids = sorted(by_pid)
    rng = random.Random(seed)
    point = bradley_terry(pairs, models)
    samples = defaultdict(list)
    best = defaultdict(int)
    for _ in range(B):
        bs = [x for p in (rng.choice(pids) for _ in pids) for x in by_pid[p]]
        s = bradley_terry(bs, models, iters=100)
        for m in models:
            samples[m].append(s[m])
        best[max(s, key=s.get)] += 1
    return {m: {"log_strength": round(point[m], 3), "ci90": [round(float(np.percentile(samples[m], 5)), 3),
                                                            round(float(np.percentile(samples[m], 95)), 3)],
                "p_best": round(best[m] / B, 3)} for m in models}


# ---- TrueSkill (1v1 with draws; Herbrich et al. 2006) -------------------------------------------
MU0, SIG0 = 25.0, 25.0 / 3
BETA, DRAW_P = SIG0 / 2, 0.10


def _pdf(x):
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def _cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _ppf(p):  # inverse normal cdf (bisection; called once)
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _cdf(mid) < p else (lo, mid)
    return (lo + hi) / 2


EPS = _ppf((DRAW_P + 1) / 2) * math.sqrt(2) * BETA


def _vw_win(t, e):
    d = max(_cdf(t - e), 1e-12)
    v = _pdf(t - e) / d
    return v, v * (v + t - e)


def _vw_draw(t, e):
    d = max(_cdf(e - t) - _cdf(-e - t), 1e-12)
    v = (_pdf(-e - t) - _pdf(e - t)) / d
    w = v * v + ((e - t) * _pdf(e - t) + (e + t) * _pdf(-e - t)) / d
    return v, w


def trueskill(pairs, models, passes=5, seed=0):
    acc = defaultdict(list)
    for p in range(passes):
        r = {m: [MU0, SIG0] for m in models}
        ps = list(pairs)
        random.Random(seed + p).shuffle(ps)
        for _, _, a, b, res in ps:
            if res == 0.0:
                a, b, res = b, a, 1.0
            (ma, sa), (mb, sb) = r[a], r[b]
            c = math.sqrt(2 * BETA ** 2 + sa ** 2 + sb ** 2)
            t = (ma - mb) / c
            v, w = _vw_draw(t, EPS / c) if res == 0.5 else _vw_win(t, EPS / c)
            r[a] = [ma + sa ** 2 / c * v, sa * math.sqrt(max(1 - sa ** 2 / c ** 2 * w, 1e-6))]
            r[b] = [mb - sb ** 2 / c * v, sb * math.sqrt(max(1 - sb ** 2 / c ** 2 * w, 1e-6))]
        for m in models:
            acc[m].append(r[m])
    return {m: {"mu": round(float(np.mean([x[0] for x in acc[m]])), 2), "sigma": round(float(np.mean([x[1] for x in acc[m]])), 2),
                "conservative": round(float(np.mean([x[0] - 3 * x[1] for x in acc[m]])), 2)} for m in models}


# ---- router ---------------------------------------------------------------------------------------
def router_eval(groups, style_prior, emb_path, k=15, alpha=0.7):
    # per prompt: mean composite per model over seeds
    util = defaultdict(lambda: defaultdict(list))
    style_of = {}
    for (pid, _s), g in groups.items():
        for r in g:
            if r.get("composite") is not None:
                util[pid][r["model"]].append(r["composite"])
                style_of[pid] = r["style"]
    util = {p: {m: float(np.mean(v)) for m, v in d.items()} for p, d in util.items()}
    pids = sorted(util)
    models = sorted({m for d in util.values() for m in d})
    if not pids or len(models) < 2:
        return None
    E = None
    if emb_path and os.path.exists(emb_path):
        z = np.load(emb_path, allow_pickle=False)
        pos = {p: i for i, p in enumerate(z["pids"].tolist())}
        if all(p in pos for p in pids):
            E = z["emb"].astype(np.float32)[[pos[p] for p in pids]]
            E /= np.linalg.norm(E, axis=1, keepdims=True)
    glob_mean = {m: float(np.mean([util[p][m] for p in pids if m in util[p]])) for m in models}
    global_best = max(glob_mean, key=glob_mean.get)
    res = {"top1": 0, "regret": [], "base_global_top1": 0, "base_global_regret": [], "base_style_top1": 0,
           "base_style_regret": [], "base_random_regret": [], "n": 0, "knn": E is not None, "k": k, "alpha": alpha}
    for i, p in enumerate(pids):
        u = util[p]
        if len(u) < 2:
            continue
        oracle = max(u, key=u.get)
        st = style_of[p]
        prior = style_prior.get(st, {})
        if E is not None:
            sims = E @ E[i]
            sims[i] = -9
            nb = [pids[j] for j in np.argsort(-sims)[:k]]
        else:  # no embeddings: neighbours = other prompts of the same style
            nb = [q for q in pids if q != p and style_of[q] == st]
        knn = {m: float(np.mean([util[q][m] for q in nb if m in util[q]] or [0.0])) for m in u}
        pz = {m: prior.get(m, {}).get("log_strength", 0.0) for m in u}
        score = {m: alpha * knn[m] + (1 - alpha) * pz[m] for m in u}
        pred = max(score, key=score.get)
        style_best = max(pz, key=pz.get)
        res["n"] += 1
        res["top1"] += pred == oracle
        res["regret"].append(u[oracle] - u[pred])
        res["base_global_top1"] += global_best == oracle
        res["base_global_regret"].append(u[oracle] - u.get(global_best, min(u.values())))
        res["base_style_top1"] += style_best == oracle
        res["base_style_regret"].append(u[oracle] - u[style_best])
        res["base_random_regret"].append(u[oracle] - float(np.mean(list(u.values()))))
    n = max(res["n"], 1)
    out = {"n_prompts": res["n"], "knn_embeddings": res["knn"], "k": k, "alpha": alpha, "global_best": global_best,
           "router_top1": round(res["top1"] / n, 3), "router_regret": round(float(np.mean(res["regret"] or [0])), 3),
           "style_best_top1": round(res["base_style_top1"] / n, 3),
           "style_best_regret": round(float(np.mean(res["base_style_regret"] or [0])), 3),
           "global_best_top1": round(res["base_global_top1"] / n, 3),
           "global_best_regret": round(float(np.mean(res["base_global_regret"] or [0])), 3),
           "random_regret": round(float(np.mean(res["base_random_regret"] or [0])), 3),
           "chance_top1": round(1 / len(models), 3)}
    return out, util, style_of, pids, E


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--router-out", default="")
    ap.add_argument("--group-key", default="ckpt", choices=["ckpt", "variant_ckpt"])
    ap.add_argument("--bootstrap", type=int, default=200)
    a = ap.parse_args(argv)

    rows = load_scores(a.dir, a.group_key)
    groups = composite(rows)
    models = sorted({r["model"] for r in rows.values()})
    styles = sorted({r["style"] for r in rows.values()})
    out = {"images": len(rows), "models": models, "styles": styles}

    # per-model metric means + rating gate + generation speed
    pm = {}
    for m in models:
        rs = [r for r in rows.values() if r["model"] == m]
        e = sorted(x["exec_s"] for x in rs if x.get("exec_s"))
        pm[m] = {"images": len(rs), "nsfw_flagged": sum(bool(r.get("nsfw_flag")) for r in rs),
                 "must_not_violations": sum(bool(r.get("must_not_hits")) for r in rs),
                 "exec_median_s": round(e[len(e) // 2], 2) if e else None}
        for k in METRICS + ["siglip_p", "composite"]:
            v = [r[k] for r in rs if r.get(k) is not None and not r.get("nsfw_flag")]
            pm[m][k] = round(float(np.mean(v)), 4) if v else None
    out["per_model"] = pm
    out["per_model_style"] = {}
    for st in styles:
        out["per_model_style"][st] = {}
        for m in models:
            v = [r.get("composite") for r in rows.values() if r["model"] == m and r["style"] == st and r.get("composite") is not None]
            out["per_model_style"][st][m] = round(float(np.mean(v)), 3) if v else None

    mp = metric_pairs(groups)
    vp, vstat = vlm_pairs(a.dir, None)
    vp = [x for x in vp if x[2] in models and x[3] in models]
    out["vlm"] = vstat
    for name, pairs in (("metric", mp), ("vlm", vp)):
        if not pairs:
            continue
        out[f"bt_{name}"] = {"all": bt_bootstrap(pairs, models, a.bootstrap)}
        out[f"trueskill_{name}"] = {"all": trueskill(pairs, models)}
        for st in styles:
            ps = [x for x in pairs if x[1] == st]
            if ps:
                out[f"bt_{name}"][st] = bt_bootstrap(ps, models, max(50, a.bootstrap // 4))
                out[f"trueskill_{name}"][st] = trueskill(ps, models)
    # judge vs metrics: share of non-tie VLM verdicts the composite agrees with
    mdict = {(x[0], x[2], x[3]): x[4] for x in mp}
    agree = [(res == mdict[(p, a_, b_)]) for p, _, a_, b_, res in vp if res != 0.5 and mdict.get((p, a_, b_), 0.5) != 0.5]
    out["vlm_vs_metric_agreement"] = round(float(np.mean(agree)), 3) if agree else None

    prior_src = out.get("bt_vlm") or out.get("bt_metric") or {}
    r = router_eval(groups, prior_src, os.path.join(a.dir, "scores", "text_emb.npz"))
    if r:
        out["router_loo"], util, style_of, pids, E = r
        if a.router_out:
            json.dump({"version": "uc18-router-v0", "style_prior": prior_src, "utility": util, "style_of": style_of,
                       "pids": pids, "emb": E.round(4).tolist() if E is not None else None,
                       "k": out["router_loo"]["k"], "alpha": out["router_loo"]["alpha"]}, open(a.router_out, "w"))
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: out.get(k) for k in ("images", "vlm", "vlm_vs_metric_agreement", "router_loo")}, indent=1), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
