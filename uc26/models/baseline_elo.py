"""Baseline B (plan 2.1): Elo with home advantage, 3-way via a fitted draw rate.

Online and point-in-time by construction: events are processed in time order, a prediction
at bet_time sees only results whose available_at < bet_time (ties: result after prediction).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


class Elo:
    def __init__(self, k: float = 20.0, home_adv: float = 50.0, season_regress: float = 1 / 3, base: float = 1500.0):
        self.k, self.home_adv, self.season_regress, self.base = k, home_adv, season_regress, base
        self.draw_rate = 0.2

    def expected_home(self, r_home: float, r_away: float) -> float:
        return 1.0 / (1.0 + 10 ** (-(r_home + self.home_adv - r_away) / 400.0))

    def run(self, games: pd.DataFrame, train_mask=None) -> np.ndarray:
        """Return [n, 3] probs (H, D, A) for every row of `games`.

        Columns needed: game_id, league, season, home_team, away_team, bet_time,
        result_available_at, outcome (0=H, 1=D, 2=A; NaN if not final).
        `train_mask` selects rows used to fit the draw rate (never test rows).
        """
        g = games.reset_index(drop=True)
        if train_mask is not None:
            tm = np.asarray(train_mask, bool)
            y = g.loc[tm, "outcome"].dropna()
            if len(y):
                self.draw_rate = float((y == 1).mean())
        events = []  # (time, order, kind, idx); order 0 = predict before update at same timestamp
        for i, r in g.iterrows():
            events.append((r["bet_time"], 0, "p", i))
            if pd.notna(r["outcome"]) and pd.notna(r["result_available_at"]):
                events.append((r["result_available_at"], 1, "u", i))
        events.sort(key=lambda e: (e[0], e[1], e[3]))
        rating: dict[tuple[str, str], float] = defaultdict(lambda: self.base)
        season_of: dict[tuple[str, str], str] = {}
        out = np.zeros((len(g), 3))
        for _, _, kind, i in events:
            r = g.loc[i]
            kh, ka = (r["league"], r["home_team"]), (r["league"], r["away_team"])
            for key in (kh, ka):
                if season_of.get(key) not in (None, r["season"]):
                    rating[key] = self.base + (1 - self.season_regress) * (rating[key] - self.base)
                season_of[key] = r["season"]
            e = self.expected_home(rating[kh], rating[ka])
            if kind == "p":
                pd_ = self.draw_rate
                out[i] = [(1 - pd_) * e, pd_, (1 - pd_) * (1 - e)]
            else:
                s = {0: 1.0, 1: 0.5, 2: 0.0}[int(r["outcome"])]
                rating[kh] += self.k * (s - e)
                rating[ka] -= self.k * (s - e)
        return out
