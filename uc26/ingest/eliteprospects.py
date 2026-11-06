"""eliteprospects.com: rosters, lineups / absences, player stats (plan 1.1).

TODO:
  * check robots.txt and terms of use first; prefer an existing Apify Actor or an official API
  * fetch per (league, season): game list, per-game lineups (who dressed, starting goalie), box scores
  * map to rows: lineups (available_at = time the lineup was confirmed, ~T-1h; if unknown,
    use start_time and flag source), player_games (available_at = final whistle)
  * stable player_id (source id), team names normalised across sources (alias table)
  * rate limit with ingest.base.RateLimiter, cache raw payloads under data/raw/eliteprospects/
"""
from __future__ import annotations

from ingest.base import StubSource


class EliteProspects(StubSource):
    name = "eliteprospects"
    robots_txt = "https://www.eliteprospects.com/robots.txt"
    min_interval_s = 10.0
