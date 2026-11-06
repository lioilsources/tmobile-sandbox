"""Assemble the per-game backtest table from the store.

Three kinds of columns, kept strictly apart:
  * features + bet-time odds  -> point-in-time (features/pit.py, available_at < bet_time)
  * outcome                   -> label, future by definition (only for training/scoring)
  * closing odds              -> future by definition, used ONLY for CLV and Baseline A
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from features.build import build_features
from features.pit import PointInTimeReader
from models.baseline_novig import remove_margin

SEL = ["H", "D", "A"]


def bet_time_odds(reader: PointInTimeReader, game_id: str, bet_time: str, book: str, market: str = "1X2") -> dict | None:
    """Latest price per selection that existed before bet_time at `book` (never a later one)."""
    v = reader.at(bet_time)
    snaps = v.records("odds_snapshots", [("game_id", "=", game_id), ("book", "=", book), ("market", "=", market)],
                      columns=["selection", "odds"], order_by="available_at")
    last = {r["selection"]: r for r in snaps}  # ordered ascending -> last write wins = latest price
    if not all(k in last for k in SEL):
        return None
    return {k: float(last[k]["odds"]) for k in SEL} | {"ts": max(last[k]["available_at"] for k in SEL)}


def load_dataset(conn: sqlite3.Connection, bet_book: str, close_book: str, bet_offset_min: float = 60.0,
                 leagues: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Return (dataset, exclusions). One row per final game that had a bet-time price."""
    games = pd.read_sql_query("SELECT game_id, league, season, start_time, home_team, away_team, status FROM games "
                              "ORDER BY start_time, game_id", conn)
    if leagues:
        games = games[games["league"].isin(leagues)]
    excl = {"not_final": int((games["status"] != "final").sum()), "no_bet_odds": 0, "no_closing": 0, "no_result": 0}
    # Postponed / cancelled games are excluded explicitly and counted (survivorship trap).
    games = games[games["status"] == "final"].copy()
    games["bet_time"] = (pd.to_datetime(games["start_time"]) - pd.Timedelta(minutes=bet_offset_min)).dt.strftime(
        "%Y-%m-%d %H:%M:%S")

    res = pd.read_sql_query("SELECT game_id, home_goals, away_goals, available_at AS result_available_at FROM results",
                            conn)
    clo = pd.read_sql_query("SELECT game_id, selection, odds FROM closing_odds WHERE book = ? AND market = '1X2'",
                            conn, params=[close_book])
    clo = clo.pivot_table(index="game_id", columns="selection", values="odds")

    reader = PointInTimeReader(conn)
    rows = []
    for g in games.to_dict("records"):
        o = bet_time_odds(reader, g["game_id"], g["bet_time"], bet_book)
        if o is None:
            excl["no_bet_odds"] += 1
            continue
        if g["game_id"] not in clo.index or clo.loc[g["game_id"]][SEL].isna().any():
            excl["no_closing"] += 1
            continue
        c = clo.loc[g["game_id"]]
        rows.append(g | {"o_H": o["H"], "o_D": o["D"], "o_A": o["A"], "odds_ts": o["ts"],
                         "c_H": float(c["H"]), "c_D": float(c["D"]), "c_A": float(c["A"])})
    df = pd.DataFrame(rows)
    if not len(df):
        return df, excl
    df = df.merge(res, on="game_id", how="left")
    excl["no_result"] = int(df["home_goals"].isna().sum())
    df = df[df["home_goals"].notna()].copy()
    df["outcome"] = np.select([df["home_goals"] > df["away_goals"], df["home_goals"] == df["away_goals"]], [0, 1], 2)

    pb = remove_margin(df[["o_H", "o_D", "o_A"]].to_numpy())
    pc = remove_margin(df[["c_H", "c_D", "c_A"]].to_numpy())
    for j, k in enumerate(SEL):
        df[f"p_bet_{k}"] = pb[:, j]
        df[f"p_close_{k}"] = pc[:, j]
    df["market"] = "1X2"

    feats = build_features(reader, df[["game_id", "start_time", "home_team", "away_team", "bet_time"]],
                           bet_offset_min)
    df = df.merge(feats.drop(columns="bet_time"), on="game_id", how="left")
    return df.sort_values(["bet_time", "game_id"]).reset_index(drop=True), excl
