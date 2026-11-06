"""GO / NO-GO against plan section 0. Thresholds are constants on purpose: written before the code,
never tuned. Changing them requires changing the plan first."""
from __future__ import annotations

from dataclasses import dataclass, field

CLV_MEAN_MIN = 0.02        # mean CLV > +2 % per bet
CLV_POS_SHARE_MIN = 0.55   # positive CLV on > 55 % of bets
ROI_MIN = 0.03             # backtest ROI > +3 % (flat, walk-forward)
MIN_BETS = 1000            # on at least 1 000 bets
CI_LEVEL = 0.95            # bootstrap CI of ROI must not contain 0


@dataclass
class Verdict:
    go: bool
    checks: dict = field(default_factory=dict)  # name -> (value, threshold, passed)

    @property
    def label(self) -> str:
        return "GO" if self.go else "NO-GO"


def decide(n_bets: int, roi_flat: float, clv_mean: float, clv_pos_share: float, ci_low: float, ci_high: float) -> Verdict:
    """All criteria use FLAT staking: an edge that only exists with Kelly does not exist (rule 5)."""
    def ok(x):
        return bool(x == x and x)  # NaN-safe

    checks = {
        "clv_mean": (clv_mean, f"> {CLV_MEAN_MIN:+.0%}", ok(clv_mean > CLV_MEAN_MIN)),
        "clv_pos_share": (clv_pos_share, f"> {CLV_POS_SHARE_MIN:.0%}", ok(clv_pos_share > CLV_POS_SHARE_MIN)),
        "roi_flat": (roi_flat, f"> {ROI_MIN:+.0%}", ok(roi_flat > ROI_MIN)),
        "n_bets": (n_bets, f">= {MIN_BETS}", n_bets >= MIN_BETS),
        # a CI entirely below 0 also "excludes 0" but is a loss, so the check is ci_low > 0
        "bootstrap_ci": ((ci_low, ci_high), "excludes 0 (low > 0)", ok(ci_low > 0)),
    }
    return Verdict(go=all(c[2] for c in checks.values()), checks=checks)
