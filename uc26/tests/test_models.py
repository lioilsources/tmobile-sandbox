import numpy as np
import pandas as pd
import pytest

from models.baseline_elo import Elo
from models.baseline_novig import overround, remove_margin


def test_novig_proportional():
    o = np.array([[2.0, 4.0, 4.0], [1.5, 4.5, 6.0]])
    p = remove_margin(o)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    np.testing.assert_allclose(p[0], [0.5, 0.25, 0.25])
    assert overround(o)[0] == pytest.approx(0.0)
    assert overround(o)[1] > 0


def test_novig_power_shifts_margin_to_longshots():
    o = np.array([1.40, 5.0, 8.0])
    pp, pw = remove_margin(o, "proportional"), remove_margin(o, "power")
    assert pw.sum() == pytest.approx(1.0)
    assert pw[0] > pp[0] and pw[2] < pp[2]


def test_novig_rejects_bad_odds():
    with pytest.raises(ValueError):
        remove_margin([1.0, 2.0, 3.0])


def _games(outcomes, times):
    rows = []
    for i, (y, t) in enumerate(zip(outcomes, times)):
        rows.append({"game_id": f"g{i}", "league": "L", "season": "S1", "home_team": "A", "away_team": "B",
                     "bet_time": t[0], "result_available_at": t[1], "outcome": y})
    return pd.DataFrame(rows)


def test_elo_home_advantage_and_draws():
    g = _games([0], [("2001-01-01 17:00:00", "2001-01-01 20:00:00")])
    p = Elo(home_adv=50).run(g)
    assert p[0, 0] > p[0, 2]
    assert p[0].sum() == pytest.approx(1.0)


def test_elo_is_point_in_time():
    """A game's own result, or one published after its bet_time, must not move its prediction."""
    t = [("2001-01-01 17:00:00", "2001-01-01 20:00:00"),   # g0 result known at 20:00
         ("2001-01-01 19:00:00", "2001-01-01 22:00:00"),   # g1 bets at 19:00 -> must not see g0
         ("2001-01-02 17:00:00", "2001-01-02 20:00:00")]   # g2 sees g0 and g1
    a = Elo().run(_games([0, 0, 0], t))
    b = Elo().run(_games([2, 2, 0], t))
    np.testing.assert_allclose(a[:2], b[:2])
    assert not np.allclose(a[2], b[2])
