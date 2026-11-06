"""Point-in-time "wave" features (plan section 2.2).

Every number is computed from a `PointInTimeView` (features/pit.py) taken at `bet_time`,
so nothing published at or after bet_time can leak in. This module must not import sqlite3
or execute SQL (enforced by tests/test_pit_leakage.py).

Features per game (home minus away where it makes sense):
  form_*        exponentially weighted points (W=1, D=0.5, L=0, regulation) of the last N games,
                home team at home / away team away, plus overall
  rest_*, b2b_* days since previous game, back-to-back flag
  momentum_*    mean (goals - shots * league shooting%) over last N: "luck" proxy for xG.
                Sign is NOT assumed (plan section 9: regression to the mean is a trap)
  abs_value_*   sum of points/game (last 20) of regular players missing from today's lineup
  abs_toi_*     number of missing regulars with above-average TOI (bench depth)
  goalie_*      shrunk save% of today's starter minus save% of the team's number one
  lineup_known  1 if both confirmed lineups were available at bet_time
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from db import parse_ts
from features.pit import PointInTimeReader, PointInTimeView

FORM_N = 10
FORM_HALFLIFE = 5.0
PLAYER_WINDOW = 20
REGULAR_SHARE = 0.5
LEAGUE_SV = 0.905
SV_PRIOR_SHOTS = 300.0
LEAGUE_SH_PCT = 0.095

FEATURES = [
    "form_home_at_home", "form_away_at_away", "form_diff",
    "rest_diff", "b2b_home", "b2b_away",
    "momentum_diff",
    "abs_value_home", "abs_value_away", "abs_value_diff",
    "abs_toi_diff",
    "goalie_home", "goalie_away", "goalie_diff",
    "lineup_known",
]


def _team_history(v: PointInTimeView, team: str, before_start: str, n: int) -> list[dict]:
    """Last n games of `team` that started before `before_start` and whose result is already published.

    `games.status` is deliberately not used: it is updated after the fact, the presence of a
    `results` row with available_at < bet_time is the point-in-time truth.
    """
    cols = ["game_id", "start_time", "home_team", "away_team"]
    games = []
    for side in ("home_team", "away_team"):
        games += v.records("games", [(side, "=", team), ("start_time", "<", before_start)],
                           columns=cols, order_by="start_time", desc=True, limit=n)
    games = sorted(games, key=lambda g: g["start_time"], reverse=True)[:n]
    if not games:
        return []
    res = {r["game_id"]: r for r in v.records(
        "results", [("game_id", "in", [g["game_id"] for g in games])],
        columns=["game_id", "home_goals", "away_goals", "home_shots", "away_shots"])}
    out = []
    for g in games:
        r = res.get(g["game_id"])
        if r is None:  # result not yet published at bet_time
            continue
        home = g["home_team"] == team
        gf, ga = (r["home_goals"], r["away_goals"]) if home else (r["away_goals"], r["home_goals"])
        sf = (r["home_shots"] if home else r["away_shots"]) or 0
        out.append({"game_id": g["game_id"], "start_time": g["start_time"], "is_home": home, "gf": gf, "ga": ga,
                    "sf": sf, "pts": 1.0 if gf > ga else (0.5 if gf == ga else 0.0)})
    return out


def _ew_mean(values: list[float], default: float) -> float:
    """Exponentially weighted mean, values[0] = most recent."""
    if not values:
        return default
    w = 0.5 ** (np.arange(len(values)) / FORM_HALFLIFE)
    return float(np.sum(w * np.asarray(values, float)) / np.sum(w))


def _days_between(a: str, b: str) -> float:
    return (parse_ts(b) - parse_ts(a)).total_seconds() / 86400.0


def _absences(v: PointInTimeView, game_id: str, team: str, hist_ids: list[str]) -> tuple[float, float, float, bool]:
    """(abs_value, abs_toi, goalie_delta, lineup_known) for one team."""
    lu = v.records("lineups", [("game_id", "=", game_id), ("team", "=", team)],
                   columns=["player_id", "is_starting_goalie"])
    pg = v.records("player_games", [("game_id", "in", hist_ids), ("team", "=", team)],
                   columns=["game_id", "player_id", "position", "toi_sec", "goals", "assists", "saves",
                            "shots_against"])
    if not lu or not pg:
        return 0.0, 0.0, 0.0, False
    n_games = len({r["game_id"] for r in pg})
    sk: dict[str, list] = defaultdict(lambda: [0, 0.0, 0.0])  # gp, points, toi sum
    gk: dict[str, list] = defaultdict(lambda: [0, 0.0, 0.0])  # starts, saves, shots against
    for r in pg:
        if r["position"] == "G":
            a = gk[r["player_id"]]
            a[0] += int((r["toi_sec"] or 0) >= 1800)
            a[1] += r["saves"] or 0
            a[2] += r["shots_against"] or 0
        else:
            a = sk[r["player_id"]]
            a[0] += 1
            a[1] += (r["goals"] or 0) + (r["assists"] or 0)
            a[2] += r["toi_sec"] or 0
    regulars = {pid: (a[1] / a[0], a[2] / a[0]) for pid, a in sk.items() if a[0] >= REGULAR_SHARE * n_games}
    dressed = {r["player_id"] for r in lu}
    missing = [x for pid, x in regulars.items() if pid not in dressed]
    abs_value = float(sum(m[0] for m in missing))
    mean_toi = float(np.mean([x[1] for x in regulars.values()])) if regulars else 0.0
    abs_toi = float(sum(1 for m in missing if m[1] > mean_toi))

    starter = [r["player_id"] for r in lu if r["is_starting_goalie"] == 1]
    goalie_delta = 0.0
    if gk and starter:
        def sv(pid: str) -> float:
            a = gk.get(pid, [0, 0.0, 0.0])
            return (a[1] + LEAGUE_SV * SV_PRIOR_SHOTS) / (a[2] + SV_PRIOR_SHOTS)

        number_one = max(gk.items(), key=lambda kv: (kv[1][0], kv[0]))[0]
        goalie_delta = sv(starter[0]) - sv(number_one)
    return abs_value, abs_toi, goalie_delta, True


def game_features(v: PointInTimeView, game: dict) -> dict:
    """Features for one fixture `game` (dict with game_id, start_time, home_team, away_team)."""
    gid, start = game["game_id"], game["start_time"]
    out: dict[str, float] = {}
    side = {}
    for role in ("home", "away"):
        team = game[f"{role}_team"]
        h = _team_history(v, team, start, max(FORM_N, PLAYER_WINDOW))
        recent = h[:FORM_N]
        form_venue = _ew_mean([g["pts"] for g in recent if g["is_home"] == (role == "home")], 0.5)
        form_all = _ew_mean([g["pts"] for g in recent], 0.5)
        rest = min(_days_between(h[0]["start_time"], start), 7.0) if h else 7.0
        mom = float(np.mean([g["gf"] - g["sf"] * LEAGUE_SH_PCT for g in recent])) if recent else 0.0
        abs_value, abs_toi, gdelta, known = _absences(v, gid, team, [g["game_id"] for g in h])
        side[role] = dict(form_venue=form_venue, form_all=form_all, rest=rest, mom=mom,
                          abs_value=abs_value, abs_toi=abs_toi, goalie=gdelta, known=known)
    hs, as_ = side["home"], side["away"]
    out["form_home_at_home"] = hs["form_venue"]
    out["form_away_at_away"] = as_["form_venue"]
    out["form_diff"] = hs["form_all"] - as_["form_all"]
    out["rest_diff"] = hs["rest"] - as_["rest"]
    out["b2b_home"] = float(hs["rest"] <= 1.2)
    out["b2b_away"] = float(as_["rest"] <= 1.2)
    out["momentum_diff"] = hs["mom"] - as_["mom"]
    out["abs_value_home"] = hs["abs_value"]
    out["abs_value_away"] = as_["abs_value"]
    out["abs_value_diff"] = hs["abs_value"] - as_["abs_value"]
    out["abs_toi_diff"] = hs["abs_toi"] - as_["abs_toi"]
    out["goalie_home"] = hs["goalie"]
    out["goalie_away"] = as_["goalie"]
    out["goalie_diff"] = hs["goalie"] - as_["goalie"]
    out["lineup_known"] = float(hs["known"] and as_["known"])
    for k, val in out.items():
        if not math.isfinite(val):
            out[k] = 0.0
    return out


def build_features(reader: PointInTimeReader, fixtures: pd.DataFrame, bet_offset_min: float = 60.0) -> pd.DataFrame:
    """One row per fixture; bet_time = start_time - bet_offset_min (realistic T-1h, plan section 9)."""
    rows = []
    for g in fixtures.to_dict("records"):
        bet_time = g.get("bet_time") or (parse_ts(g["start_time"]) - pd.Timedelta(minutes=bet_offset_min))
        v = reader.at(bet_time)
        f = game_features(v, g)
        f["game_id"] = g["game_id"]
        f["bet_time"] = v.bet_time
        rows.append(f)
    return pd.DataFrame(rows, columns=["game_id", "bet_time"] + FEATURES)
