"""Staking (rule 5): flat 1 unit and fractional (1/4) Kelly, both capped (rule 4: low limits).

Profit is always computed from the bet-time price at the book we would actually use (rule 3),
optionally haircut by `slippage` to model our own price impact in thin markets.
"""
from __future__ import annotations

import numpy as np


def effective_odds(odds, slippage: float = 0.0) -> np.ndarray:
    o = np.asarray(odds, float)
    return 1.0 + (o - 1.0) * (1.0 - slippage)


def flat(odds, won, stake: float = 1.0, max_stake: float = np.inf, slippage: float = 0.0):
    """(stakes, profits) for flat staking."""
    o = effective_odds(odds, slippage)
    s = np.full(len(o), min(stake, max_stake))
    p = np.where(np.asarray(won, bool), s * (o - 1.0), -s)
    return s, p


def kelly(prob, odds, won, fraction: float = 0.25, bankroll: float = 100.0, max_stake: float = 5.0,
          slippage: float = 0.0):
    """(stakes, profits, bankroll_path) for sequential fractional Kelly; rows must be in bet order.

    f* = (p*o - 1) / (o - 1); stake = min(fraction * f* * bankroll, max_stake), never negative.
    """
    o = effective_odds(odds, slippage)
    p = np.asarray(prob, float)
    w = np.asarray(won, bool)
    stakes, profits, path = np.zeros(len(o)), np.zeros(len(o)), np.zeros(len(o))
    bank = bankroll
    for i in range(len(o)):
        f = max(0.0, (p[i] * o[i] - 1.0) / (o[i] - 1.0))
        s = min(fraction * f * bank, max_stake, max(bank, 0.0))
        pr = s * (o[i] - 1.0) if w[i] else -s
        bank += pr
        stakes[i], profits[i], path[i] = s, pr, bank
    return stakes, profits, path
