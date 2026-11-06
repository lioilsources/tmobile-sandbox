"""Ingest interface (phase 1). Runs OFF the T-Mobile VM (on the Mac); the VM only gets the
finished SQLite file for batch compute.

Every source turns raw pages / API payloads into rows for schema.sql tables, and every row
carries `available_at` = when the information was publicly known:
  * results / box scores  -> final whistle (or the publish time, if later)
  * lineups               -> time the confirmed lineup appeared (marginal leagues: ~T-1h)
  * odds snapshot         -> the snapshot timestamp itself (never back-dated)
  * historic odds we only learn about later (e.g. OddsPortal opening price): available_at is the
    timestamp the source claims for that price, and the row keeps `source` so it can be audited.

Etiquette (hard rules, not suggestions):
  * read and obey robots.txt of each site before writing a fetcher; if it disallows a path, don't.
  * respect the site's terms of use; prefer official APIs / Apify Actors where they exist.
  * rate limit: at least `min_interval_s` between requests per host (default 5 s), one
    connection per host, back off exponentially on 429/5xx, identify with an honest User-Agent.
  * cache raw responses under data/raw/ (gitignored) so re-parsing never re-downloads.
  * nothing scraped (pages, odds, names) is ever committed: this repo is public.

Nothing in this package performs network I/O yet: fetchers are TODO stubs.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Iterator, Protocol


@dataclass(frozen=True)
class Row:
    table: str          # one of db.TABLES
    values: dict        # column -> value; must include available_at

    def __post_init__(self):
        if "available_at" not in self.values or not self.values["available_at"]:
            raise ValueError(f"{self.table}: row without available_at")


class Source(Protocol):
    """A data source. `fetch` yields raw payloads (cached), `parse` turns them into Rows."""

    name: str
    robots_txt: str         # where robots.txt lives; checked manually before enabling the fetcher
    min_interval_s: float

    def fetch(self, league: str, season: str) -> Iterator[bytes]: ...

    def parse(self, payload: bytes) -> Iterable[Row]: ...


class RateLimiter:
    """Minimum interval between calls (per instance = per host). Pure local logic, no I/O."""

    def __init__(self, min_interval_s: float = 5.0, clock=time.monotonic, sleep=time.sleep):
        self.min_interval_s, self._clock, self._sleep = min_interval_s, clock, sleep
        self._last: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            delta = self.min_interval_s - (now - self._last)
            if delta > 0:
                self._sleep(delta)
                now = self._clock()
        self._last = now


class StubSource:
    """Base for not-yet-implemented sources: declares metadata, refuses to fetch."""

    name = "stub"
    robots_txt = ""
    min_interval_s = 5.0

    def fetch(self, league: str, season: str) -> Iterator[bytes]:
        raise NotImplementedError(f"{self.name}: fetcher not implemented (see TODO in module)")

    def parse(self, payload: bytes) -> Iterable[Row]:
        raise NotImplementedError(f"{self.name}: parser not implemented (see TODO in module)")
