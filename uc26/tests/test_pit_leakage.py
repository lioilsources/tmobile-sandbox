"""Rule 2: no feature may read data with available_at >= bet_time.

  cd uc26 && .venv/bin/python -m pytest -q tests
"""
import ast
import glob
import os
import shutil
import sqlite3

import pandas as pd
import pytest

from db import TABLES, connect
from features.build import FEATURES, build_features
from features.pit import LeakageError, PointInTimeReader, PointInTimeView

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fixtures(conn, n=25):
    g = pd.read_sql_query("SELECT game_id, start_time, home_team, away_team FROM games WHERE status='final' "
                          "ORDER BY start_time", conn)
    # spread over the timeline, skip the very first rounds (no history)
    return g.iloc[len(g) // 6::max(1, len(g) // n)].head(n).reset_index(drop=True)


def _bet_time(start):
    return (pd.Timestamp(start) - pd.Timedelta(minutes=60)).strftime("%Y-%m-%d %H:%M:%S")


def test_view_never_returns_future_rows(small_db):
    conn = connect(small_db, create=False)
    for start in _fixtures(conn, 8)["start_time"]:
        bt = _bet_time(start)
        v = PointInTimeReader(conn).at(bt)
        for t in TABLES:
            rows = v.records(t)
            assert all(r["available_at"] < bt for r in rows)
            expected = conn.execute(f"SELECT COUNT(*) FROM {t} WHERE available_at < ?", [bt]).fetchone()[0]
            assert len(rows) == expected


def test_boundary_is_strict(small_db, tmp_path):
    db = str(tmp_path / "b.sqlite")
    shutil.copy(small_db, db)
    conn = connect(db, create=False)
    gid, start = conn.execute("SELECT game_id, start_time FROM games WHERE status='final' LIMIT 1 OFFSET 200").fetchone()
    bt = _bet_time(start)
    conn.execute("UPDATE lineups SET available_at = ? WHERE game_id = ?", [bt, gid])
    v = PointInTimeReader(conn).at(bt)
    assert v.records("lineups", [("game_id", "=", gid)]) == []  # available_at == bet_time is NOT available


def test_features_invariant_to_deleting_the_future(small_db, tmp_path):
    """Features at bet_time on the full DB == features on a DB truncated to rows known before bet_time."""
    conn = connect(small_db, create=False)
    fx = _fixtures(conn, 12)
    full = build_features(PointInTimeReader(conn), fx)
    for i, g in fx.iterrows():
        bt = _bet_time(g["start_time"])
        db = str(tmp_path / f"t{i}.sqlite")
        shutil.copy(small_db, db)
        c2 = sqlite3.connect(db)
        c2.execute("PRAGMA foreign_keys = OFF")
        for t in TABLES:
            if t != "games":  # fixture rows are needed for the target game itself
                c2.execute(f"DELETE FROM {t} WHERE available_at >= ?", [bt])
        c2.commit()
        trunc = build_features(PointInTimeReader(c2), fx.iloc[[i]])
        pd.testing.assert_frame_equal(full.iloc[[i]][FEATURES].reset_index(drop=True),
                                      trunc[FEATURES].reset_index(drop=True))
        c2.close()
        os.remove(db)


def test_features_invariant_to_poisoned_future(small_db, tmp_path):
    """Corrupt every future row; features must not move."""
    db = str(tmp_path / "p.sqlite")
    shutil.copy(small_db, db)
    conn = connect(db, create=False)
    fx = _fixtures(conn, 10)
    before = build_features(PointInTimeReader(conn), fx)
    bt_max = max(_bet_time(s) for s in fx["start_time"])
    # rows not yet available at the LATEST bet time are future for every fixture -> poison them
    assert conn.execute("SELECT COUNT(*) FROM results WHERE available_at >= ?", [bt_max]).fetchone()[0] > 0
    conn.execute("UPDATE results SET home_goals = 99, away_goals = 0, home_shots = 1 WHERE available_at >= ?", [bt_max])
    conn.execute("UPDATE player_games SET goals = 50, saves = 0, toi_sec = 9999 WHERE available_at >= ?", [bt_max])
    conn.commit()
    after = build_features(PointInTimeReader(conn), fx)
    pd.testing.assert_frame_equal(before[FEATURES], after[FEATURES])


def test_lineup_after_bet_time_is_invisible(small_db, tmp_path):
    db = str(tmp_path / "l.sqlite")
    shutil.copy(small_db, db)
    conn = connect(db, create=False)
    fx = _fixtures(conn, 3)
    assert build_features(PointInTimeReader(conn), fx)["lineup_known"].eq(1).all()
    for gid, start in fx[["game_id", "start_time"]].itertuples(index=False):
        conn.execute("UPDATE lineups SET available_at = ? WHERE game_id = ?", [start, gid])  # confirmed at puck drop
    conn.commit()
    assert build_features(PointInTimeReader(conn), fx)["lineup_known"].eq(0).all()


def test_exit_check_catches_a_leaky_query(small_db):
    """If the SQL filter were ever broken, the exit check must still raise."""
    conn = connect(small_db, create=False)

    class Broken(PointInTimeView):
        def records(self, table, *a, **k):
            cur = self._conn.execute(f"SELECT * FROM {table}")
            names = [d[0] for d in cur.description]
            rows = [dict(zip(names, r)) for r in cur.fetchall()]
            for r in rows:
                if r["available_at"] >= self.bet_time:
                    raise LeakageError("future row")
            return rows

    with pytest.raises(LeakageError):
        Broken(conn, "2002-01-01 00:00:00").records("results")


def test_filters_are_structured():
    conn = sqlite3.connect(":memory:")
    v = PointInTimeView(conn, "2001-01-01 00:00:00")
    with pytest.raises(ValueError):
        v.records("results; DROP TABLE games")
    with pytest.raises(ValueError):
        v.records("results", [("1=1 OR available_at", ">", "")])
    with pytest.raises(ValueError):
        v.records("results", [("game_id", "OR", "x")])


def test_only_pit_module_touches_the_database():
    """features/*.py except pit.py must not import sqlite3/db access or execute SQL."""
    banned_modules = {"sqlite3", "sqlalchemy", "psycopg", "psycopg2"}
    banned_calls = {"execute", "executemany", "executescript", "read_sql", "read_sql_query", "connect", "cursor"}
    for path in glob.glob(os.path.join(HERE, "features", "*.py")):
        name = os.path.basename(path)
        if name == "pit.py":
            continue
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not {a.name.split(".")[0] for a in node.names} & banned_modules, name
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in banned_modules | {"db"} or \
                    {a.name for a in node.names} <= {"parse_ts", "ts"}, name
            elif isinstance(node, ast.Call):
                fn = node.func
                called = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                assert called not in banned_calls, f"{name} calls {called}(); go through features.pit"
