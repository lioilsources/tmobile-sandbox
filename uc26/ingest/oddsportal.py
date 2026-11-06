"""OddsPortal: historic opening + closing odds per bookmaker (plan 1.2, free source first).

TODO:
  * check robots.txt and terms of use first; respect rate limits (slow, cached, resumable)
  * per game: opening and closing price per book for 1X2 / handicap / total, with timestamps
    -> odds_snapshots (opening, available_at = the opening timestamp the site shows) and
       closing_odds (available_at = start_time)
  * survivorship: OddsPortal sometimes lacks postponed / cancelled games -> join with games and
    mark them, never silently drop (plan section 9)
  * the closing line of a sharp book (Pinnacle where listed) is the CLV reference
  * paid sources (The Odds API, Pinnacle resellers) are decided later; same Row interface
"""
from __future__ import annotations

from ingest.base import StubSource


class OddsPortal(StubSource):
    name = "oddsportal"
    robots_txt = "https://www.oddsportal.com/robots.txt"
    min_interval_s = 15.0
