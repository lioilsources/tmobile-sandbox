"""Markdown report reports/backtest_<date>.md (rule 8). Text in Czech (user-facing document)."""
from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

from backtest import metrics, staking
from backtest.verdict import Verdict
from backtest.walkforward import Config, Result

SMALL_GROUP = 200  # a group with fewer bets than this is noise on its own


def _pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.2%}"


def _group_table(b: pd.DataFrame, by: str, cfg: Config) -> str:
    lines = [f"| {by} | sázky | ROI flat | CLV průměr | CLV > 0 | zisk (j.) |", "|---|---|---|---|---|---|"]
    for key, g in b.groupby(by):
        s, p = staking.flat(g["odds"], g["won"], cfg.stake, cfg.max_stake, cfg.slippage)
        flag = " ⚠ málo" if len(g) < SMALL_GROUP else ""
        lines.append(f"| {key}{flag} | {len(g)} | {_pct(metrics.roi(s, p))} | {_pct(g['clv'].mean())} | "
                     f"{(g['clv'] > 0).mean():.1%} | {p.sum():+.1f} |")
    return "\n".join(lines)


def render(res: Result, summary: dict, verdict: Verdict, cfg: Config, meta: dict) -> str:
    b = res.bets
    s = summary
    out = [f"# UC26 backtest — {meta.get('title', 'BestBet validace')}", "",
           f"Datum: {meta.get('date')} · DB: `{meta.get('db_name', '?')}` · sázková kancelář: `{meta.get('bet_book')}` · "
           f"closing: `{meta.get('close_book')}`", "",
           f"## Verdikt: **{verdict.label}**", "",
           "| Kritérium (sekce 0) | Hodnota | Práh | OK |", "|---|---|---|---|"]
    names = {"clv_mean": "CLV průměr", "clv_pos_share": "CLV kladné u sázek", "roi_flat": "ROI flat",
             "n_bets": "Počet sázek", "bootstrap_ci": "95% bootstrap CI ROI"}
    for k, (val, thr, ok) in verdict.checks.items():
        if k == "bootstrap_ci":
            vs = f"[{_pct(val[0])}, {_pct(val[1])}]"
        elif k == "n_bets":
            vs = str(val)
        elif k == "clv_pos_share":
            vs = f"{val:.1%}"
        else:
            vs = _pct(val)
        out.append(f"| {names[k]} | {vs} | {thr} | {'✅' if ok else '❌'} |")
    out += ["", "Kritéria se počítají z flat stakingu; edge, který existuje jen s Kellym, neexistuje.", "",
            "## Souhrn", "",
            f"- Sezóny: trénink `{res.seasons['train0']}`, validace `{res.seasons['validation']}` "
            f"(threshold **{res.threshold:.2f}**, C = {res.C}, zamčeno), test {', '.join(f'`{x}`' for x in res.seasons['test'])}",
            f"- Testovací zápasy: {s['n_test_games']}, sázky: {s['n_bets']}",
            f"- ROI flat: {_pct(s['roi_flat'])} (zisk {s['profit_flat']:+.1f} j.), 95% CI [{_pct(s['ci_low'])}, {_pct(s['ci_high'])}] "
            f"({cfg.n_boot} převzorkování)",
            f"- ROI 1/4 Kelly (cap {cfg.max_stake:g} j., bank {cfg.kelly_bankroll:g}): {_pct(s['roi_kelly'])}, "
            f"konečný bank {s['kelly_final_bankroll']:.1f}",
            f"- CLV: průměr {_pct(s['clv_mean'])}, kladné u {s['clv_pos_share']:.1%} sázek" if s['n_bets'] else "- CLV: n/a",
            f"- Max. drawdown (flat): {s['max_drawdown_flat']:.1f} j.",
            f"- Vyřazeno: {meta.get('exclusions', {})}", "",
            "## Modely na testovacích sezónách (nižší = lepší)", "",
            "| Model | log-loss | Brier |", "|---|---|---|"]
    for name, sc in s["scores"].items():
        out.append(f"| {name} | {sc['log_loss']:.4f} | {sc['brier']:.4f} |")
    if len(b):
        out += ["", "## Po ligách", "", _group_table(b, "league", cfg), "",
                "## Po trzích", "", _group_table(b, "market", cfg), "",
                "## Po sezónách", "", _group_table(b, "season", cfg)]
        by_league = b.groupby("league").apply(lambda g: (g["won"] * g["odds"] - 1).sum(), include_groups=False)
        if s["profit_flat"] > 0 and len(by_league) > 1 and by_league.max() > s["profit_flat"]:
            out += ["", f"⚠ Zisk stojí na jedné lize (`{by_league.idxmax()}`); ostatní dohromady prodělávají."]
    if res.coefficients is not None:
        out += ["", "## Koeficienty wave logreg (poslední fold, standardizované)", "",
                "| feature | H | D | A |", "|---|---|---|---|"]
        for f, r in res.coefficients.iterrows():
            out.append(f"| {f} | {r.get('H', np.nan):+.3f} | {r.get('D', np.nan):+.3f} | {r.get('A', np.nan):+.3f} |")
        out += ["", "Záporný koeficient momenta = trh je napřed (sekce 9)."]
    out += ["", "## Validační mřížka", "", "| C | threshold | sázky | ROI |", "|---|---|---|---|"]
    for g in res.validation:
        out.append(f"| {g['C']} | {g['threshold']:.2f} | {g['bets']} | {_pct(g['roi']) if np.isfinite(g['roi']) else 'n/a'} |")
    return "\n".join(out) + "\n"


def write(text: str, out_dir: str, date: str | None = None) -> str:
    date = date or dt.date.today().isoformat()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"backtest_{date}.md")
    with open(path, "w") as f:
        f.write(text)
    return path
