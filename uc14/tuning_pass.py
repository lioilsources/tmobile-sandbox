"""UC14 krok 3 (ZJEDNODUŠENÁ verze) — tuning pass jako optimalizace „skill curve“ (Optuna).

ZJEDNODUŠENÍ oproti plánu: místo naučené PPO policy (krok 2) hraje **šumový plánovač**:
pro každý serve vybere z pevné sady míření (ox × oy × síla, oblouk 0, timing = ideál) to,
které má pod jeho vlastním šumem nejvyšší odhad p(gól) (M vzorků), a pak se změří na
čerstvých vzorcích. Šum je **odhad**, ne změřená lidská data (swipe log export zatím není):
  začátečník: σ_timing 60 ms, σ_kontakt 0.15 (v jednotkách poloměru), síla ×N(1, 0.15)
  pokročilý:  σ_timing 25 ms, σ_kontakt 0.07, síla ×N(1, 0.08)
Timing šum je kvantovaný na tick (8.3 ms), stejně jako ve hře (ROADMAP D6).

Cíl (VOLLEY_PLAN tuning pass, plán UC14 §4 krok 3): začátečník ~30 % gólů, pokročilý ~60 %,
oracle-lite (bez šumu, nejlepší míření ze sady ± 3 ticky) ~100 % a P(gól | rating GOOD) ≥ 0.5.

  python tuning_pass.py --trials 80 --serves 300 --out UC14_tuning.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import volley_jax as vj  # noqa: E402
import oracle as orc  # noqa: E402
import jax.numpy as jnp  # noqa: E402

SKILLS = {"beginner": dict(st=0.060, sxy=0.15, sp=0.15), "advanced": dict(st=0.025, sxy=0.07, sp=0.08)}
TARGET = {"beginner": 0.30, "advanced": 0.60}
AIMS = orc.make_grid(np.linspace(-0.4, 0.4, 5), np.linspace(-0.8, 0.0, 9), np.array([0.0]),
                     np.array([0.7, 0.85, 1.0]), np.array([0]))
SEARCH = {  # konstanty tuning.gd, které tuning pass smí měnit (rozsahy)
    "TIMING_SIGMA": (0.03, 0.12),
    "SWING_LIFT": (0.0, 0.8),
    "RATING_PERFECT": (0.02, 0.045),
    "RATING_GOOD": (0.05, 0.10),
    "GOAL_DISTANCE": (14.0, 22.0),
}


def play(win, T, n_serves, skill, m_plan, m_eval, rng):
    """Šumový plánovač: vrátí (p_goal, p_goal_given_GOOD, n_GOOD)."""
    A = len(AIMS[0])
    sk = SKILLS[skill]

    def noisy(idx_s, aim_i, m):
        n = len(idx_s) * m
        ii = np.repeat(idx_s, m); ai = np.repeat(aim_i, m)
        ox = AIMS[0][ai] + rng.normal(0, sk["sxy"], n)
        oy = AIMS[1][ai] + rng.normal(0, sk["sxy"], n)
        pw = np.clip(AIMS[3][ai] * rng.normal(1, sk["sp"], n), 0, 1)
        dt = np.round(rng.normal(0, sk["st"], n) * vj.TICKS_PER_SEC).astype(np.int32)
        whiff = np.abs(dt) > orc.W   # |err| > 116.7 ms → mimo okno (RATING_LATE_EARLY 120 ms)
        out = orc.eval_flat(win, ii, ox.astype(np.float32), oy.astype(np.float32), np.zeros(n, np.float32),
                            pw.astype(np.float32), np.clip(dt, -orc.W, orc.W), T)
        goal = (out[0] == vj.GOAL) & ~whiff
        rating = np.asarray(vj.rate_timing(jnp.asarray(dt * vj.DT), T))
        return goal, rating

    # plánování: každý serve × každé míření × m_plan vzorků
    s_idx = np.repeat(np.arange(n_serves), A); a_idx = np.tile(np.arange(A), n_serves)
    g, _ = noisy(s_idx, a_idx, m_plan)
    p = g.reshape(n_serves, A, m_plan).mean(-1)
    best = p.argmax(1)
    g, r = noisy(np.arange(n_serves), best, m_eval)
    good = r == vj.GOOD
    return float(g.mean()), float(g[good].mean()) if good.any() else 0.0, int(good.sum())


def oracle_lite(win, T, n_serves):
    A = len(AIMS[0]); dts = np.arange(-3, 4)
    ii = np.repeat(np.arange(n_serves), A * len(dts))
    ai = np.tile(np.repeat(np.arange(A), len(dts)), n_serves)
    dt = np.tile(dts, n_serves * A).astype(np.int32)
    out = orc.eval_flat(win, ii, AIMS[0][ai].astype(np.float32), AIMS[1][ai].astype(np.float32),
                        np.zeros(len(ii), np.float32), AIMS[3][ai].astype(np.float32), dt, T)
    return float((out[0] == vj.GOAL).reshape(n_serves, -1).any(1).mean())


def evaluate(overrides, n_serves, seed, m_plan=8, m_eval=32):
    tun, T = orc.load(overrides)
    seeds = np.arange(100000, 100000 + n_serves)  # jiné seedy než oracle
    _, _, win = orc.prep_serves(seeds, tun, T)
    rng = np.random.default_rng(seed)
    res = {"oracle_lite": oracle_lite(win, T, n_serves)}
    for sk in SKILLS:
        pg, pgg, ng = play(win, T, n_serves, sk, m_plan, m_eval, rng)
        res[sk] = pg; res[f"{sk}_p_goal_given_GOOD"] = pgg; res[f"{sk}_n_GOOD"] = ng
    good = [res[f"{s}_p_goal_given_GOOD"] for s in SKILLS if res[f"{s}_n_GOOD"] > 20]
    res["p_goal_given_GOOD"] = float(np.mean(good)) if good else 0.0
    loss = sum((res[s] - TARGET[s]) ** 2 for s in SKILLS) + 2.0 * (1.0 - res["oracle_lite"]) ** 2 \
        + max(0.0, 0.5 - res["p_goal_given_GOOD"]) ** 2
    res["loss"] = float(loss)
    return res


def main():
    import optuna
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--serves", type=int, default=300)
    ap.add_argument("--out")
    a = ap.parse_args()
    tun = vj.parse_tuning(vj.default_tuning_path())
    t0 = time.time()
    base = evaluate({}, a.serves, 0)
    base_s = time.time() - t0
    print("baseline", json.dumps(base), f"{base_s:.1f}s", flush=True)

    def objective(trial):
        ov = {k: trial.suggest_float(k, lo, hi) for k, (lo, hi) in SEARCH.items()}
        if ov["RATING_PERFECT"] >= ov["RATING_GOOD"]:
            raise optuna.TrialPruned()
        r = evaluate(ov, a.serves, trial.number)
        for k, v in r.items():
            trial.set_user_attr(k, v)
        return r["loss"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=14))
    study.enqueue_trial({k: float(tun[k]) for k in SEARCH})
    t0 = time.time()
    study.optimize(objective, n_trials=a.trials,
                   callbacks=[lambda s, t: print(t.number, t.value, t.params, flush=True)])
    opt_s = time.time() - t0
    best = study.best_trial
    # přeměření nejlepšího kandidáta na víc serve a čerstvém šumu
    confirm = evaluate(best.params, a.serves * 3, 999)
    base_confirm = evaluate({}, a.serves * 3, 999)
    diff = [f"-const {k} := {tun[k]}\n+const {k} := {round(v, 4)}" for k, v in best.params.items()]
    rep = {
        "SIMPLIFICATION": "šumový plánovač s odhadnutým lidským šumem místo naučené PPO policy; šum není z reálných swipe logů",
        "skills": SKILLS, "targets": TARGET, "search_space": SEARCH, "trials": a.trials, "serves_per_trial": a.serves,
        "seconds_per_trial": opt_s / a.trials, "baseline": base, "baseline_confirm": base_confirm,
        "best_params": best.params, "best_trial": best.user_attrs, "best_confirm": confirm,
        "tuning_gd_diff": diff,
        "top5": [{"params": t.params, "loss": t.value, **t.user_attrs}
                 for t in sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value)[:5]],
    }
    print(json.dumps(rep, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
