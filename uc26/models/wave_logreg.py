""""Wave" model (plan 2.3): multinomial logistic regression over point-in-time features plus
the no-vig probability at bet time (log). Interpretable, hard to overfit. Gradient boosting
only after this shows a signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features.build import FEATURES

MARKET_COLS = ["logp_H", "logp_D", "logp_A"]


def design(df: pd.DataFrame, features=FEATURES) -> np.ndarray:
    """Feature matrix: wave features + log no-vig bet-time probabilities (p_bet_H/D/A)."""
    m = np.log(np.clip(df[["p_bet_H", "p_bet_D", "p_bet_A"]].to_numpy(float), 1e-6, 1.0))
    return np.column_stack([df[list(features)].to_numpy(float), m])


class WaveLogReg:
    def __init__(self, C: float = 1.0, features=FEATURES):
        self.C, self.features = C, list(features)
        self.pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))

    def fit(self, df: pd.DataFrame) -> "WaveLogReg":
        self.pipe.fit(design(df, self.features), df["outcome"].astype(int).to_numpy())
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        p = self.pipe.predict_proba(design(df, self.features))
        out = np.zeros((len(df), 3))
        for j, c in enumerate(self.pipe.classes_):
            out[:, int(c)] = p[:, j]
        return out

    def coefficients(self) -> pd.DataFrame:
        """Standardised coefficients per class (H, D, A) for the report (sign of momentum etc.)."""
        lr = self.pipe[-1]
        names = self.features + MARKET_COLS
        return pd.DataFrame(lr.coef_.T, index=names, columns=[["H", "D", "A"][int(c)] for c in lr.classes_])
