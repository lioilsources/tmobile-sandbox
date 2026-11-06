"""Smoke test of the whole pipeline on synthetic data with a KNOWN edge:
no edge -> NO-GO, strong injected edge -> GO. If either fails, the backtest lies."""
import os

import pytest

import synth
from backtest.run import backtest

CFG = dict(n_leagues=3, n_seasons=5, n_teams=14)


@pytest.mark.parametrize("edge,expected", [(0.0, "NO-GO"), (1.0, "GO")])
def test_verdict_tracks_injected_edge(tmp_path, edge, expected):
    db = str(tmp_path / f"synth_{edge}.sqlite")
    synth.generate(db, edge=edge, seed=11, **CFG)
    out = backtest(db, "softbook", "sharpbook", str(tmp_path / "reports"), date="smoke")
    print(f"\nedge={edge}: verdict={out['verdict']} bets={out['n_bets']} roi_flat={out['roi_flat']:+.4f} "
          f"roi_kelly={out['roi_kelly']:+.4f} clv={out['clv_mean']:+.4f} clv>0={out['clv_pos_share']:.3f} "
          f"ci=[{out['ci_low']:+.4f}, {out['ci_high']:+.4f}] thr={out['threshold']} C={out['C']}")
    assert out["verdict"] == expected
    assert os.path.exists(out["report"]) and out["report"].endswith("backtest_smoke.md")
