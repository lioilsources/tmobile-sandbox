"""flashscore / livesport: results by period, start times, shots, power plays (plan 1.1).

TODO:
  * check robots.txt and terms of use first; the site is JS-heavy -> an Apify Actor is the
    likely route (Go client from ApifySandbox), not a home-made headless scraper
  * map to rows: games (available_at = schedule publish time, or first time we saw the fixture),
    results (available_at = final whistle), status postponed / cancelled kept, not dropped
  * rate limit with ingest.base.RateLimiter, cache raw payloads under data/raw/flashscore/
"""
from __future__ import annotations

from ingest.base import StubSource


class Flashscore(StubSource):
    name = "flashscore"
    robots_txt = "https://www.flashscore.com/robots.txt"
    min_interval_s = 10.0
