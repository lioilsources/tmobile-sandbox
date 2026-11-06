"""CLV, ROI, bootstrap CI (rule 6), drawdown, probabilistic scores."""
from __future__ import annotations

import numpy as np


def roi(stakes, profits) -> float:
    s = float(np.sum(stakes))
    return float(np.sum(profits)) / s if s > 0 else float("nan")


def clv(bet_odds, p_close_novig) -> np.ndarray:
    """Closing line value per bet: bet price vs. fair (no-vig) closing price, o_bet * p_close - 1."""
    return np.asarray(bet_odds, float) * np.asarray(p_close_novig, float) - 1.0


def bootstrap_roi_ci(profits, stakes=None, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0,
                     chunk: int = 500) -> tuple[float, float]:
    """Percentile CI of ROI from n_boot resamples of bets (with replacement)."""
    p = np.asarray(profits, float)
    s = np.ones_like(p) if stakes is None else np.asarray(stakes, float)
    n = len(p)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        m = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(m, n))
        out[start:start + m] = p[idx].sum(axis=1) / np.maximum(s[idx].sum(axis=1), 1e-12)
    lo, hi = np.quantile(out, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def max_drawdown(profits) -> float:
    """Largest peak-to-trough drop of the cumulative profit curve (in units, >= 0)."""
    c = np.concatenate([[0.0], np.cumsum(np.asarray(profits, float))])
    return float(np.max(np.maximum.accumulate(c) - c))


def log_loss(probs, y) -> float:
    p = np.clip(np.asarray(probs, float)[np.arange(len(y)), np.asarray(y, int)], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def brier(probs, y) -> float:
    pr = np.asarray(probs, float)
    onehot = np.eye(pr.shape[1])[np.asarray(y, int)]
    return float(np.mean(np.sum((pr - onehot) ** 2, axis=1)))
