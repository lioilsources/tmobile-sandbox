"""UC26 backtest CLI: DB -> walk-forward -> verdict -> reports/backtest_<date>.md

  cd uc26 && .venv/bin/python -m backtest.run --db data/synth.sqlite --bet-book softbook --close-book sharpbook
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from backtest import report  # noqa: E402
from backtest.dataset import load_dataset  # noqa: E402
from backtest.verdict import decide  # noqa: E402
from backtest.walkforward import Config, run, summarize  # noqa: E402
from db import connect  # noqa: E402


def backtest(db_path: str, bet_book: str, close_book: str, out_dir: str, date: str | None = None,
             cfg: Config | None = None, leagues: list[str] | None = None, bet_offset_min: float = 60.0,
             title: str | None = None) -> dict:
    cfg = cfg or Config()
    conn = connect(db_path, create=False)
    df, excl = load_dataset(conn, bet_book, close_book, bet_offset_min, leagues)
    conn.close()
    res = run(df, cfg)
    s = summarize(res, cfg)
    v = decide(s["n_bets"], s["roi_flat"], s["clv_mean"], s["clv_pos_share"], s["ci_low"], s["ci_high"])
    date = date or dt.date.today().isoformat()
    meta = {"date": date, "db_name": os.path.basename(db_path), "bet_book": bet_book, "close_book": close_book,
            "exclusions": excl, "title": title or "BestBet validace"}
    path = report.write(report.render(res, s, v, cfg, meta), out_dir, date)
    s.pop("flat_profits")
    return {"verdict": v.label, "report": path, "threshold": res.threshold, "C": res.C, **s}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--bet-book", required=True, help="book where we would actually bet (profit uses its price)")
    ap.add_argument("--close-book", required=True, help="sharp book for closing line (CLV only)")
    ap.add_argument("--out", default=os.path.join(HERE, "reports"))
    ap.add_argument("--date", default=None)
    ap.add_argument("--league", action="append", help="restrict to league(s); repeatable")
    ap.add_argument("--bet-offset-min", type=float, default=60.0, help="bet_time = start - N minutes")
    ap.add_argument("--max-stake", type=float, default=5.0)
    ap.add_argument("--slippage", type=float, default=0.0)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    cfg = Config(max_stake=a.max_stake, slippage=a.slippage, n_boot=a.n_boot, seed=a.seed)
    out = backtest(a.db, a.bet_book, a.close_book, a.out, a.date, cfg, a.league, a.bet_offset_min)
    out.pop("scores", None)
    print(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
