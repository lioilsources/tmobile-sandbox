"""Write ingested Rows into the SQLite store. Local only, no network."""
from __future__ import annotations

import sqlite3
from typing import Iterable

from db import TABLES, ts
from ingest.base import Row


def upsert(conn: sqlite3.Connection, rows: Iterable[Row]) -> int:
    """INSERT OR REPLACE rows; canonicalises every *_at / ts / start_time timestamp. Returns count."""
    n = 0
    for r in rows:
        if r.table not in TABLES:
            raise ValueError(f"unknown table {r.table!r}")
        vals = dict(r.values)
        for k in ("available_at", "ts", "start_time"):
            if vals.get(k) is not None:
                vals[k] = ts(vals[k])
        cols = list(vals)
        conn.execute(f"INSERT OR REPLACE INTO {r.table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                     [vals[c] for c in cols])
        n += 1
    conn.commit()
    return n
