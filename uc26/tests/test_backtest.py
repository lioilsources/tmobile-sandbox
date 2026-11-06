import numpy as np
import pandas as pd
import pytest

from backtest import metrics, staking, walkforward
from backtest.dataset import load_dataset
from backtest.verdict import MIN_BETS, decide
from db import connect


@pytest.fixture(scope="module")
def dataset(small_db):
    conn = connect(small_db, create=False)
    df, excl = load_dataset(conn, "softbook", "sharpbook")
    return df, excl


def test_dataset_excludes_non_final_and_uses_bet_time_odds(dataset, small_db):
    df, excl = dataset
    conn = connect(small_db, create=False)
    n_post = conn.execute("SELECT COUNT(*) FROM games WHERE status != 'final'").fetchone()[0]
    assert excl["not_final"] == n_post
    assert (df["odds_ts"] < df["bet_time"]).all()       # rule 3: price existed at bet time
    assert not np.allclose(df["o_H"], df["c_H"])         # profit price != closing price


def test_walkforward_seasons_and_locked_threshold(dataset):
    df, _ = dataset
    cfg = walkforward.Config(n_boot=200)
    r1 = walkforward.run(df, cfg)
    seasons = sorted(df["season"].unique())
    assert r1.seasons == {"train0": seasons[0], "validation": seasons[1], "test": seasons[2:]}
    assert set(r1.bets["season"]) <= set(seasons[2:])
    # rule 7: scrambling TEST outcomes must not change the threshold / C chosen on validation
    df2 = df.copy()
    test = df2["season"].isin(seasons[2:])
    df2.loc[test, "outcome"] = np.random.default_rng(0).permutation(df2.loc[test, "outcome"].to_numpy())
    r2 = walkforward.run(df2, cfg)
    assert (r1.threshold, r1.C) == (r2.threshold, r2.C)


def test_walkforward_needs_three_seasons(dataset):
    df, _ = dataset
    with pytest.raises(ValueError):
        walkforward.run(df[df["season"] != sorted(df["season"].unique())[-1]])


def test_flat_and_kelly_cap():
    s, p = staking.flat([2.0, 3.0], [True, False])
    np.testing.assert_allclose(s, [1, 1])
    np.testing.assert_allclose(p, [1, -1])
    ks, kp, path = staking.kelly([0.9, 0.9], [2.0, 2.0], [True, True], fraction=0.25, bankroll=1000, max_stake=5)
    assert ks.max() == 5  # capped (rule 4)
    ks, _, _ = staking.kelly([0.4], [2.0], [True])
    assert ks[0] == 0     # negative EV -> no stake


def test_slippage_reduces_profit():
    _, p0 = staking.flat([2.0], [True])
    _, p1 = staking.flat([2.0], [True], slippage=0.1)
    assert p1[0] < p0[0]


def test_bootstrap_ci():
    rng = np.random.default_rng(1)
    profits = np.where(rng.random(2000) < 0.55, 1.0, -1.0)
    lo, hi = metrics.bootstrap_roi_ci(profits, n_boot=10_000, seed=3)
    assert lo < profits.mean() < hi
    assert (lo, hi) == metrics.bootstrap_roi_ci(profits, n_boot=10_000, seed=3)


def test_clv_and_drawdown():
    assert metrics.clv([2.1], [0.5])[0] == pytest.approx(0.05)
    assert metrics.max_drawdown([1, 1, -3, 1]) == pytest.approx(3)


def test_verdict_section0():
    good = dict(n_bets=MIN_BETS, roi_flat=0.031, clv_mean=0.021, clv_pos_share=0.56, ci_low=0.001, ci_high=0.06)
    assert decide(**good).go
    for k, bad in [("n_bets", MIN_BETS - 1), ("roi_flat", 0.03), ("clv_mean", 0.02), ("clv_pos_share", 0.55),
                   ("ci_low", -0.001), ("roi_flat", float("nan"))]:
        assert not decide(**(good | {k: bad})).go, k
