"""Ingest is a skeleton: stubs must refuse to fetch and nothing may touch the network."""
import socket

import pytest

from db import connect
from ingest import book_snapshots, eliteprospects, flashscore, oddsportal
from ingest.base import RateLimiter, Row
from ingest.store import upsert

SOURCES = [eliteprospects.EliteProspects, flashscore.Flashscore, oddsportal.OddsPortal, *book_snapshots.SOURCES]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)


@pytest.mark.parametrize("cls", SOURCES)
def test_stubs_refuse(cls):
    s = cls()
    assert s.robots_txt.endswith("/robots.txt") and s.min_interval_s >= 5
    with pytest.raises(NotImplementedError):
        list(s.fetch("X", "2025-26"))
    with pytest.raises(NotImplementedError):
        list(s.parse(b""))


def test_row_requires_available_at():
    with pytest.raises(ValueError):
        Row("games", {"game_id": "g"})


def test_rate_limiter():
    t = [0.0]
    slept = []
    rl = RateLimiter(5.0, clock=lambda: t[0], sleep=lambda d: (slept.append(d), t.__setitem__(0, t[0] + d)))
    rl.wait()
    t[0] += 2
    rl.wait()
    assert slept == [3.0]


def test_upsert_canonicalises(tmp_path):
    conn = connect(str(tmp_path / "i.sqlite"))
    upsert(conn, [Row("games", {"game_id": "g1", "league": "X", "season": "2025-26", "start_time": "2025-10-01T17:00:00Z",
                                "home_team": "A", "away_team": "B", "available_at": "2025-09-01T00:00:00+02:00"})])
    r = conn.execute("SELECT start_time, available_at FROM games").fetchone()
    assert tuple(r) == ("2025-10-01 17:00:00", "2025-08-31 22:00:00")
