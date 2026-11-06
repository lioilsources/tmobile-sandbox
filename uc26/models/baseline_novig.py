"""Baseline A (plan 2.1): implied probability from odds with the bookmaker margin removed.

From the CLOSING line of a sharp book this is the best public estimate; if the model does not
beat it (log-loss / Brier on the test seasons), there is no edge.
"""
from __future__ import annotations

import numpy as np


def overround(odds) -> np.ndarray:
    """Sum of implied probabilities minus 1 (the margin), per row."""
    o = np.atleast_2d(np.asarray(odds, dtype=float))
    return (1.0 / o).sum(axis=1) - 1.0


def remove_margin(odds, method: str = "proportional", tol: float = 1e-10) -> np.ndarray:
    """No-vig probabilities for each row of decimal odds (shape [n, k] or [k]).

    proportional: p_i = (1/o_i) / sum_j (1/o_j)
    power:        p_i = (1/o_i)^k with k chosen so that sum p_i = 1 (shifts margin to longshots)
    """
    o = np.asarray(odds, dtype=float)
    single = o.ndim == 1
    o = np.atleast_2d(o)
    if np.any(o <= 1.0):
        raise ValueError("decimal odds must be > 1")
    q = 1.0 / o
    if method == "proportional":
        p = q / q.sum(axis=1, keepdims=True)
    elif method == "power":
        p = np.empty_like(q)
        for i, row in enumerate(q):
            lo, hi = 0.5, 3.0
            for _ in range(200):
                k = 0.5 * (lo + hi)
                s = np.sum(row ** k)
                if abs(s - 1.0) < tol:
                    break
                lo, hi = (k, hi) if s > 1.0 else (lo, k)
            p[i] = row ** k
            p[i] /= p[i].sum()
    else:
        raise ValueError(method)
    return p[0] if single else p


def predict(odds_1x2) -> np.ndarray:
    """Baseline A probabilities [p_H, p_D, p_A] from 1X2 odds rows."""
    return remove_margin(odds_1x2, "proportional")
