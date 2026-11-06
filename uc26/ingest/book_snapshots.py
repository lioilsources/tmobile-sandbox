"""Own odds snapshots of CZ books (Tipsport, Fortuna) from now on (plan 1.2).

These are the prices we would actually bet at (rule 3/4), so collection should start early.

TODO:
  * check robots.txt / terms of each book; prefer a public feed or an Apify Actor
  * schedule (cron / launchd on the Mac, NOT on the T-Mobile VM): snapshots at T-24h, T-2h, T-1h
    and the last one before start for the target leagues
  * every snapshot -> odds_snapshots rows with ts = available_at = snapshot time (UTC)
  * last pre-start snapshot also -> closing_odds for that book
  * keep raw payloads under data/raw/<book>/ (gitignored); never commit prices
"""
from __future__ import annotations

from ingest.base import StubSource


class Tipsport(StubSource):
    name = "tipsport"
    robots_txt = "https://www.tipsport.cz/robots.txt"
    min_interval_s = 30.0


class Fortuna(StubSource):
    name = "fortuna"
    robots_txt = "https://www.ifortuna.cz/robots.txt"
    min_interval_s = 30.0


SOURCES = (Tipsport, Fortuna)
