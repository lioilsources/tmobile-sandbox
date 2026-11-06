"""UC26 SQLite helpers: open/create the point-in-time store and format timestamps.

All timestamps are UTC TEXT 'YYYY-MM-DD HH:MM:SS' so string comparison == time comparison.
"""
from __future__ import annotations

import datetime as dt
import os
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = os.path.join(HERE, "schema.sql")
TS_FMT = "%Y-%m-%d %H:%M:%S"
TABLES = ("games", "results", "lineups", "player_games", "odds_snapshots", "closing_odds")


def ts(value) -> str:
    """Normalise datetime / pandas Timestamp / str to the canonical UTC string."""
    if isinstance(value, str):
        # validate and canonicalise (accepts 'T' separator and trailing 'Z')
        value = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if value.tzinfo is not None:
        value = value.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return value.strftime(TS_FMT)


def parse_ts(value: str) -> dt.datetime:
    return dt.datetime.strptime(value, TS_FMT)


def connect(path: str, create: bool = True) -> sqlite3.Connection:
    if not create and not os.path.exists(path):
        raise FileNotFoundError(path)
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    if create:
        with open(SCHEMA) as f:
            conn.executescript(f.read())
    return conn
