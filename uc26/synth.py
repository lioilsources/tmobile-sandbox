"""UC26 synthetic data: fake hockey leagues, lineups, box scores and odds with a KNOWN edge.

Used to prove the pipeline does not lie: with --edge 0 the backtest must say NO-GO, with a
strong injected edge it should say GO. Nothing here is real: league/team/player names are
generated, the schema is the real one (schema.sql), every row carries available_at.

Generative model (regulation time, 1X2):
  d        = strength_home - strength_away - ALPHA * (missing_value_home - missing_value_away)
  shots_h ~ Poisson(SHOTS * exp(+d/2 + HFA)), shots_a ~ Poisson(SHOTS * exp(-d/2))
  goals   ~ Binomial(shots, 1 - sv_true(starting goalie of the other team))
  => goals ~ Poisson(lambda) with lambda = E[shots] * (1 - sv), true 1X2 probs are exact.

Markets:
  softbook  (where we "bet"): snapshots T-24h and T-2h, margin --soft-margin. It knows team
            strength (+noise) but misses the fraction `edge` of today's lineup information
            (absences and who starts in goal). edge = 0: knows everything, edge = 1: knows nothing.
  sharpbook closing line at puck drop: knows everything (+small noise), margin --sharp-margin.
Lineups are confirmed at T-75min, bet_time is T-60min (plan section 9).

  python synth.py --out data/synth.sqlite --edge 0.0 --seed 1
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from db import connect, ts  # noqa: E402

SHOTS = 30.0
HFA = 0.04
ALPHA = 0.06            # strength lost per unit of missing player value
N_SKATERS, N_DRESSED = 20, 18
P_INJURE, P_RECOVER = 0.05, 0.25
P_BACKUP_START = 0.3
MAXG = 16


def outcome_probs(lam_h: float, lam_a: float) -> np.ndarray:
    """Exact regulation 1X2 probabilities for independent Poisson goals."""
    k = np.arange(MAXG)
    ph = np.exp(-lam_h + k * math.log(lam_h) - np.array([math.lgamma(x + 1) for x in k]))
    pa = np.exp(-lam_a + k * math.log(lam_a) - np.array([math.lgamma(x + 1) for x in k]))
    m = np.outer(ph, pa)
    p = np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])
    return p / p.sum()


def lambdas(d: float, sv_h_goalie: float, sv_a_goalie: float) -> tuple[float, float]:
    sh = SHOTS * math.exp(d / 2 + HFA)
    sa = SHOTS * math.exp(-d / 2)
    return sh * (1 - sv_a_goalie), sa * (1 - sv_h_goalie)


def to_odds(p: np.ndarray, margin: float) -> np.ndarray:
    return np.maximum(1.01, 1.0 / (p * (1.0 + margin)))


def schedule(teams: list[str], rng) -> list[list[tuple[str, str]]]:
    """Quadruple round robin (each pair twice at each venue), circle method."""
    n = len(teams)
    t = list(teams)
    rounds = []
    for _ in range(n - 1):
        rounds.append([(t[i], t[n - 1 - i]) for i in range(n // 2)])
        t = [t[0]] + [t[-1]] + t[1:-1]
    second = [[(b, a) for a, b in r] for r in rounds]
    all_rounds = rounds + second
    all_rounds = all_rounds + [list(r) for r in all_rounds]
    rng.shuffle(all_rounds)
    return all_rounds


def generate(out: str, edge: float = 0.0, seed: int = 0, n_leagues: int = 3, n_seasons: int = 6,
             n_teams: int = 14, soft_margin: float = 0.05, sharp_margin: float = 0.025,
             mkt_noise: float = 0.03, close_noise: float = 0.01, p_postponed: float = 0.01) -> dict:
    if os.path.exists(out):
        os.remove(out)
    rng = np.random.default_rng(seed)
    conn = connect(out)
    G, R, L, PG, OS, CO = [], [], [], [], [], []
    for li in range(n_leagues):
        league = f"SYN{li + 1}"
        teams = [f"{league}-T{t + 1:02d}" for t in range(n_teams)]
        strength = {t: rng.normal(0, 0.15) for t in teams}
        players, goalies = {}, {}
        for t in teams:
            players[t] = [{"id": f"{t}-P{i + 1:02d}", "pos": "F" if i < 12 else "D",
                           "v": float(rng.gamma(2.0, 0.5)), "inj": False} for i in range(N_SKATERS)]
            goalies[t] = [{"id": f"{t}-G1", "sv": float(rng.normal(0.915, 0.005))},
                          {"id": f"{t}-G2", "sv": float(rng.normal(0.895, 0.005))}]
        for si in range(n_seasons):
            season = f"{2001 + si}-{(2002 + si) % 100:02d}"
            day = dt.datetime(2001 + si, 9, 15, 17 + li, 0)
            published = day - dt.timedelta(days=30)
            if si:
                strength = {t: 0.7 * s + rng.normal(0, 0.1) for t, s in strength.items()}
            for rnd, games in enumerate(schedule(teams, rng)):
                day += dt.timedelta(days=int(rng.choice([1, 2, 2, 3, 3, 4])))
                for gi, (h, a) in enumerate(games):
                    gid = f"{league}-{season}-R{rnd + 1:02d}-{gi + 1}"
                    start = day
                    postponed = rng.random() < p_postponed
                    G.append((gid, league, season, ts(start), h, a, "postponed" if postponed else "final", "synth",
                              ts(published)))
                    # injuries evolve per team per game (Markov), lineup = best healthy skaters
                    lineup, missing_val, starter = {}, {}, {}
                    for t in (h, a):
                        for p in players[t]:
                            p["inj"] = (rng.random() > P_RECOVER) if p["inj"] else (rng.random() < P_INJURE)
                        healthy = sorted([p for p in players[t] if not p["inj"]], key=lambda p: -p["v"])
                        dressed = healthy[:N_DRESSED]
                        top = sorted(players[t], key=lambda p: -p["v"])[:N_DRESSED]
                        ids = {p["id"] for p in dressed}
                        missing_val[t] = sum(p["v"] for p in top if p["id"] not in ids) - \
                            sum(p["v"] for p in dressed if p not in top)
                        lineup[t] = dressed
                        starter[t] = goalies[t][1] if rng.random() < P_BACKUP_START else goalies[t][0]
                    base_d = strength[h] - strength[a]
                    lineup_d = -ALPHA * (missing_val[h] - missing_val[a])
                    sv_h, sv_a = starter[h]["sv"], starter[a]["sv"]
                    p_true = outcome_probs(*lambdas(base_d + lineup_d, sv_h, sv_a))
                    # softbook sees (1-edge) of today's lineup info; default goalies = number ones
                    sv_h_m = (1 - edge) * sv_h + edge * goalies[h][0]["sv"]
                    sv_a_m = (1 - edge) * sv_a + edge * goalies[a][0]["sv"]
                    for hours, noise in ((24, 1.5 * mkt_noise), (2, mkt_noise)):
                        d_m = base_d + (1 - edge) * lineup_d + rng.normal(0, noise)
                        o = to_odds(outcome_probs(*lambdas(d_m, sv_h_m, sv_a_m)), soft_margin)
                        snap = ts(start - dt.timedelta(hours=hours))
                        for sel, x in zip("HDA", o):
                            OS.append((gid, "softbook", "1X2", sel, 0.0, snap, round(float(x), 2), "synth", snap))
                    if postponed:
                        continue
                    d_c = base_d + lineup_d + rng.normal(0, close_noise)
                    pc = outcome_probs(*lambdas(d_c, sv_h, sv_a))
                    for book, margin in (("sharpbook", sharp_margin), ("softbook", soft_margin)):
                        for sel, x in zip("HDA", to_odds(pc, margin)):
                            CO.append((gid, book, "1X2", sel, 0.0, round(float(x), 2), "synth", ts(start)))
                    lu_at = ts(start - dt.timedelta(minutes=75))
                    for t in (h, a):
                        for p in lineup[t]:
                            L.append((gid, t, p["id"], p["pos"], 0, "synth", lu_at))
                        for gk in goalies[t]:
                            L.append((gid, t, gk["id"], "G", int(gk is starter[t]), "synth", lu_at))
                    # play the game
                    sh = rng.poisson(SHOTS * math.exp((base_d + lineup_d) / 2 + HFA))
                    sa = rng.poisson(SHOTS * math.exp(-(base_d + lineup_d) / 2))
                    gh, ga = rng.binomial(sh, 1 - sv_a), rng.binomial(sa, 1 - sv_h)
                    per = lambda g: np.bincount(rng.integers(0, 3, g), minlength=3)  # noqa: E731
                    ph_, pa_ = per(gh), per(ga)
                    res_at = ts(start + dt.timedelta(hours=2, minutes=40))
                    R.append((gid, int(gh), int(ga), *[int(x) for pair in zip(ph_, pa_) for x in pair],
                              int(sh), int(sa), int(rng.poisson(3)), int(rng.poisson(3)), "REG", "synth", res_at))
                    for t, gf, shots_against, sv_g in ((h, gh, sa, (ga, sa)), (a, ga, sh, (gh, sh))):
                        dressed = lineup[t]
                        v = np.array([p["v"] for p in dressed])
                        pts = {p["id"]: [0, 0] for p in dressed}
                        for _ in range(gf):
                            sc = rng.choice(len(dressed), p=v / v.sum())
                            pts[dressed[sc]["id"]][0] += 1
                            for asst in rng.choice(len(dressed), size=2, replace=False, p=v / v.sum()):
                                if asst != sc:
                                    pts[dressed[asst]["id"]][1] += 1
                        for p in dressed:
                            toi = int(max(300, rng.normal(600 + 300 * p["v"], 90)))
                            PG.append((gid, t, p["id"], p["pos"], toi, pts[p["id"]][0], pts[p["id"]][1], None, None,
                                       "synth", res_at))
                        goals_against, sa_ = sv_g
                        st = starter[t]
                        PG.append((gid, t, st["id"], "G", 3600, 0, 0, int(sa_), int(sa_ - goals_against), "synth",
                                   res_at))
    conn.executemany("INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?)", G)
    conn.executemany("INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", R)
    conn.executemany("INSERT INTO lineups VALUES (?,?,?,?,?,?,?)", L)
    conn.executemany("INSERT INTO player_games VALUES (?,?,?,?,?,?,?,?,?,?,?)", PG)
    conn.executemany("INSERT INTO odds_snapshots VALUES (?,?,?,?,?,?,?,?,?)", OS)
    conn.executemany("INSERT INTO closing_odds VALUES (?,?,?,?,?,?,?,?)", CO)
    conn.commit()
    conn.close()
    return {"games": len(G), "results": len(R), "lineups": len(L), "player_games": len(PG),
            "odds_snapshots": len(OS), "closing_odds": len(CO)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "synth.sqlite"))
    ap.add_argument("--edge", type=float, default=0.0, help="share of lineup info the bet-time market misses (0..1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--leagues", type=int, default=3)
    ap.add_argument("--seasons", type=int, default=6)
    ap.add_argument("--teams", type=int, default=14)
    ap.add_argument("--soft-margin", type=float, default=0.05)
    a = ap.parse_args(argv)
    print(generate(a.out, a.edge, a.seed, a.leagues, a.seasons, a.teams, a.soft_margin))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
