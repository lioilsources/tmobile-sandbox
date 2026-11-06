"""Walk-forward backtest by season (rules 1, 3, 4, 5, 7).

Seasons s0 < s1 < ... < sK (sorted per the whole dataset):
  * s0            train only
  * s1            VALIDATION: model trained on s0, threshold (and C) chosen here, then LOCKED
  * s2 .. sK      TEST: for each k train on s0..s(k-1), predict sk with the locked settings
Test rows never influence the threshold; tests/test_backtest.py checks that.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest import metrics, staking
from models.baseline_elo import Elo
from models.wave_logreg import WaveLogReg

SEL = ["H", "D", "A"]
THRESHOLDS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15)
C_GRID = (0.1, 1.0)
MIN_VAL_BETS = 30


@dataclass
class Config:
    stake: float = 1.0
    max_stake: float = 5.0          # units; e.g. 1 unit = 100 CZK -> cap 500 CZK (rule 4)
    kelly_fraction: float = 0.25
    kelly_bankroll: float = 100.0
    slippage: float = 0.0
    thresholds: tuple = THRESHOLDS
    c_grid: tuple = C_GRID
    n_boot: int = 10_000
    seed: int = 0


@dataclass
class Result:
    bets: pd.DataFrame
    test: pd.DataFrame                     # all test rows with model probabilities
    threshold: float
    C: float
    seasons: dict = field(default_factory=dict)
    coefficients: pd.DataFrame | None = None
    validation: list = field(default_factory=list)


def select_bets(df: pd.DataFrame, probs: np.ndarray, threshold: float) -> pd.DataFrame:
    """At most one bet per game: selection with max EV, only if p * odds > 1 + threshold."""
    odds = df[["o_H", "o_D", "o_A"]].to_numpy(float)
    ev = probs * odds - 1.0
    j = ev.argmax(axis=1)
    take = ev[np.arange(len(df)), j] > threshold
    b = df.loc[take].copy()
    jj = j[take]
    b["selection"] = [SEL[x] for x in jj]
    b["p_model"] = probs[take, jj]
    b["odds"] = odds[take, jj]
    b["ev"] = ev[take, jj]
    b["won"] = b["outcome"].to_numpy() == jj
    b["p_close_sel"] = df.loc[take, ["p_close_H", "p_close_D", "p_close_A"]].to_numpy(float)[np.arange(len(jj)), jj]
    b["clv"] = metrics.clv(b["odds"], b["p_close_sel"])
    return b


def _flat_roi(b: pd.DataFrame, cfg: Config) -> float:
    s, p = staking.flat(b["odds"], b["won"], cfg.stake, cfg.max_stake, cfg.slippage)
    return metrics.roi(s, p)


def run(df: pd.DataFrame, cfg: Config | None = None) -> Result:
    cfg = cfg or Config()
    seasons = sorted(df["season"].unique())
    if len(seasons) < 3:
        raise ValueError(f"need >= 3 seasons (train, validation, test), got {seasons}")
    s_train0, s_val, s_tests = seasons[0], seasons[1], seasons[2:]

    # --- validation: choose C and threshold on s1 only, then lock ---------------------------
    tr, va = df[df["season"] == s_train0], df[df["season"] == s_val]
    best = None
    grid_log = []
    for C in cfg.c_grid:
        pv = WaveLogReg(C=C).fit(tr).predict_proba(va)
        for thr in cfg.thresholds:
            b = select_bets(va, pv, thr)
            r = _flat_roi(b, cfg) if len(b) >= MIN_VAL_BETS else -np.inf
            grid_log.append({"C": C, "threshold": thr, "bets": len(b), "roi": r})
            if best is None or r > best[0]:
                best = (r, C, thr)
    _, C, thr = best

    # --- test: expanding window, locked settings -------------------------------------------
    tests, coefs = [], None
    for s in s_tests:
        train = df[df["season"] < s]
        test = df[df["season"] == s].copy()
        m = WaveLogReg(C=C).fit(train)
        p = m.predict_proba(test)
        for j, k in enumerate(SEL):
            test[f"p_wave_{k}"] = p[:, j]
        tests.append(test)
        coefs = m.coefficients()
    test = pd.concat(tests).sort_values(["bet_time", "game_id"])

    # Elo baseline over the whole timeline (online, point-in-time); draw rate fit on non-test seasons
    elo_p = Elo().run(df, train_mask=df["season"].isin([s_train0, s_val]).to_numpy())
    elo = pd.DataFrame(elo_p, columns=[f"p_elo_{k}" for k in SEL], index=df.index)
    test = test.join(elo)

    probs = test[[f"p_wave_{k}" for k in SEL]].to_numpy()
    bets = select_bets(test, probs, thr).sort_values(["bet_time", "game_id"]).reset_index(drop=True)
    return Result(bets=bets, test=test, threshold=thr, C=C, coefficients=coefs, validation=grid_log,
                  seasons={"train0": s_train0, "validation": s_val, "test": list(s_tests)})


def summarize(res: Result, cfg: Config | None = None) -> dict:
    """Headline numbers for the verdict and the report."""
    cfg = cfg or Config()
    b = res.bets
    fs, fp = staking.flat(b["odds"], b["won"], cfg.stake, cfg.max_stake, cfg.slippage)
    ks, kp, path = staking.kelly(b["p_model"], b["odds"], b["won"], cfg.kelly_fraction, cfg.kelly_bankroll,
                                 cfg.max_stake, cfg.slippage)
    lo, hi = metrics.bootstrap_roi_ci(fp, fs, n_boot=cfg.n_boot, seed=cfg.seed)
    t = res.test
    y = t["outcome"].to_numpy(int)
    scores = {}
    for name, cols in {"no-vig closing (A)": "p_close", "no-vig bet-time": "p_bet", "Elo (B)": "p_elo",
                       "wave logreg": "p_wave"}.items():
        pr = t[[f"{cols}_{k}" for k in SEL]].to_numpy(float)
        scores[name] = {"log_loss": metrics.log_loss(pr, y), "brier": metrics.brier(pr, y)}
    return {
        "n_bets": int(len(b)),
        "n_test_games": int(len(t)),
        "roi_flat": metrics.roi(fs, fp),
        "profit_flat": float(np.sum(fp)),
        "roi_kelly": metrics.roi(ks, kp),
        "kelly_final_bankroll": float(path[-1]) if len(path) else cfg.kelly_bankroll,
        "clv_mean": float(b["clv"].mean()) if len(b) else float("nan"),
        "clv_pos_share": float((b["clv"] > 0).mean()) if len(b) else float("nan"),
        "ci_low": lo,
        "ci_high": hi,
        "max_drawdown_flat": metrics.max_drawdown(fp),
        "scores": scores,
        "flat_profits": fp,
    }
