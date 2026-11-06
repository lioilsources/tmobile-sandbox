"""The ONLY way the feature pipeline touches the database.

`PointInTimeReader(conn).at(bet_time)` returns a view whose every query is forced to
`available_at < bet_time`. Feature code never sees the connection, never writes SQL against
raw tables and never chooses the time filter itself. tests/test_pit_leakage.py checks that
(a) no row with available_at >= bet_time ever leaves this module, (b) features are invariant
to deleting all future rows, and (c) no other features/ module executes SQL.
"""
from __future__ import annotations

import re
import sqlite3
from typing import Iterable, Sequence

import pandas as pd

from db import TABLES, ts

_COL = re.compile(r"^[a-z_][a-z0-9_]*$")
_OPS = {"=", "<", "<=", ">", ">=", "!=", "in"}


class LeakageError(AssertionError):
    pass


class PointInTimeView:
    """Read-only, time-filtered access to the store as it looked just before `bet_time`."""

    def __init__(self, conn: sqlite3.Connection, bet_time):
        self._conn = conn
        self.bet_time = ts(bet_time)

    def rows(self, table: str, where: Sequence[tuple[str, str, object]] = (), columns: Iterable[str] | None = None,
             order_by: str | None = None, desc: bool = False, limit: int | None = None) -> pd.DataFrame:
        """Same as `records`, as a DataFrame."""
        recs = self.records(table, where, columns, order_by, desc, limit)
        cols = None if columns is None or recs else list(dict.fromkeys([*columns, "available_at"]))
        return pd.DataFrame(recs, columns=cols)

    def records(
        self,
        table: str,
        where: Sequence[tuple[str, str, object]] = (),
        columns: Iterable[str] | None = None,
        order_by: str | None = None,
        desc: bool = False,
        limit: int | None = None,
    ) -> list[dict]:
        """SELECT from `table` with structured filters [(column, op, value)] AND available_at < bet_time.

        Filters are structured (not raw SQL) so callers cannot bypass the time filter with OR.
        """
        if table not in TABLES:
            raise ValueError(f"unknown table {table!r}")
        cols = "*"
        if columns is not None:
            cols_l = list(columns)
            for c in cols_l:
                _check_col(c)
            if "available_at" not in cols_l:
                cols_l.append("available_at")
            cols = ", ".join(cols_l)
        sql = [f"SELECT {cols} FROM {table} WHERE available_at < ?"]
        params: list[object] = [self.bet_time]
        for col, op, val in where:
            _check_col(col)
            if op not in _OPS:
                raise ValueError(f"bad operator {op!r}")
            if op == "in":
                vals = list(val)  # type: ignore[arg-type]
                if not vals:
                    return []
                sql.append(f"AND {col} IN ({','.join('?' * len(vals))})")
                params.extend(vals)
            else:
                sql.append(f"AND {col} {op} ?")
                params.append(val)
        if order_by is not None:
            _check_col(order_by)
            sql.append(f"ORDER BY {order_by} {'DESC' if desc else 'ASC'}")
        if limit is not None:
            sql.append(f"LIMIT {int(limit)}")
        cur = self._conn.execute(" ".join(sql), params)
        names = [d[0] for d in cur.description]
        out = [dict(zip(names, r)) for r in cur.fetchall()]
        # Belt and braces: the guarantee is checked on the way out, not just assumed.
        for r in out:
            if r["available_at"] >= self.bet_time:
                raise LeakageError(f"{table}: row with available_at >= {self.bet_time}")
        return out


class PointInTimeReader:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def at(self, bet_time) -> PointInTimeView:
        return PointInTimeView(self._conn, bet_time)


def _check_col(name: str) -> None:
    if not _COL.match(name):
        raise ValueError(f"bad column name {name!r}")
