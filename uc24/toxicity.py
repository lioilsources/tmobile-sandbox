"""UC24 krok 4 — slovníková analýza: regrese slovo v promptu → p(vada) → "toxická" a "ochranná" slova.

  python toxicity.py --bulk /cold/uc24/bulk_sim.jsonl --out ../bench/results/UC24_toxicity_sim.json [--code rumble]

L2-logistická regrese: příznaky = modifikátory z gen_bulk.MODS (faktoriální, nezávislé na zadání) + one-hot
zadání jako kontrola (slova zadání jsou s typem zvuku zmatená — "creaking" je toxické, protože vrzání JE
sled impulzů, ne kvůli slovu). --all-words přidá i slova zadání (jen pro průzkum).
koeficient = log-odds změna p(vada); 95% interval bootstrapem (200×). Toxické = dolní mez > 0,
ochranné = horní mez < 0. Se sim backendem je pravda známá (gen.SIM_TRIGGERS) → validace metody.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

import numpy as np


def toks(prompt: str) -> set[str]:
    parts = [p.strip() for p in prompt.lower().split(",")]
    out = set()
    for p in parts:
        if p.startswith("clear") or p.startswith("close") or p.startswith("no ") or "children" in p:
            continue
        if p in ("high quality", "studio recording", "lo-fi", "close-up"):
            out.add(p)
            continue
        out |= {w for w in re.findall(r"[a-z][a-z\-]+", p) if len(w) > 2}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bulk", required=True)
    ap.add_argument("--code", default=None, help="jen tato vada (rumble/clipping/hiss/clicks…); jinak verdikt ≠ pass")
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--all-words", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    R = [json.loads(line) for line in open(a.bulk)]
    R = [r for r in R if "verdict" in r]
    if a.all_words:
        T = [toks(r["prompt"]) for r in R]
    else:
        T = [set(r.get("mods") or []) for r in R]
    cnt = Counter(w for t in T for w in t)
    vocab = sorted(w for w, c in cnt.items() if c >= a.min_count)
    tasks = sorted({r.get("task") for r in R})
    Xw = np.array([[w in t for w in vocab] for t in T], np.float32)
    Xt = np.array([[r.get("task") == k for k in tasks] for r in R], np.float32)
    X = np.hstack([Xw, Xt])
    y = np.array([(a.code in r["codes"]) if a.code else (r["verdict"] != "pass") for r in R], int)
    m = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
    coef = m.coef_[0]
    rng = np.random.default_rng(24)
    boots = []
    for _ in range(200):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        boots.append(LogisticRegression(C=1.0, max_iter=2000).fit(X[i], y[i]).coef_[0])
    B = np.array(boots)
    lo, hi = np.percentile(B, 2.5, axis=0), np.percentile(B, 97.5, axis=0)
    rows = []
    for j, w in enumerate(vocab):
        has = Xw[:, j] > 0
        rows.append({"word": w, "n": int(has.sum()), "coef": round(float(coef[j]), 3), "lo": round(float(lo[j]), 3),
                     "hi": round(float(hi[j]), 3), "p_defect_with": round(float(y[has].mean()), 3),
                     "p_defect_without": round(float(y[~has].mean()), 3)})
    rows.sort(key=lambda r: -r["coef"])
    res = {"uc": "UC24", "what": f"slovo → p({a.code or 'verdikt≠pass'})", "n": len(R), "base_rate": round(float(y.mean()), 3),
           "backend": Counter(r.get("backend") for r in R).most_common(1)[0][0],
           "toxic": [r for r in rows if r["lo"] > 0], "protective": [r for r in rows if r["hi"] < 0][::-1], "all": rows}
    try:
        import gen
        if res["backend"] == "sim":
            truth = {w.strip() for ws in gen.SIM_TRIGGERS.values() for w in ws}
            tv = {w for w in vocab if any(t in w for t in truth)}
            found = {r["word"] for r in res["toxic"]}
            res["sim_validation"] = {"true_toxic_in_vocab": sorted(tv), "found_toxic": sorted(found),
                                     "precision": round(len(found & tv) / max(1, len(found)), 3),
                                     "recall": round(len(found & tv) / max(1, len(tv)), 3)}
    except Exception:  # noqa: BLE001
        pass
    js = json.dumps(res, indent=1, ensure_ascii=False)
    if a.out:
        open(a.out, "w").write(js)
    print(json.dumps({k: res[k] for k in ("n", "base_rate", "backend")} | {"toxic": [(r["word"], r["coef"], r["n"]) for r in res["toxic"]][:15],
                      "protective": [(r["word"], r["coef"], r["n"]) for r in res["protective"]][:15]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
