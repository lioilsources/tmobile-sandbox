"""UC12: BattleOdds (lib/core/ai/battle_odds.dart) v numpy, float64 jako Dart.

Pravděpodobnost výhry útočníka z času do zničení (time-to-kill) obou lodí po upgradech,
sigmoida se strmostí 3, oříznuto na 0,1–0,9. Tabulka 6×6 se počítá jednou na dvojici flotil
a do JAX jde jako konstanta.
"""
import math

import numpy as np

# unit_base_stats.dart: (maxHp, damage, attackIntervalMs, defenseRating, zbraň)
BASE = [
    (40, 8, 400, 0.0, "rapidFire"),       # pawn
    (120, 30, 1200, 0.25, "heavyCannon"),  # rook
    (80, 20, 800, 0.10, "standard"),      # knight
    (70, 25, 900, 0.05, "sniper"),        # bishop
    (150, 40, 1000, 0.20, "heavyCannon"),  # queen
    (180, 35, 1400, 0.35, "heavyCannon"),  # king
]
SPEED = {"rapidFire": 1.0 / 300, "standard": 1.0 / 500, "sniper": 1.0 / 250, "heavyCannon": 1.0 / 700}
HP_MUL = {1: 1.20, 2: 1.40, 3: 1.60}
DMG_MUL = {1: 1.20, 2: 1.40, 3: 1.60}
SPD_MUL = {1: 0.90, 2: 0.85, 3: 0.80}
NEVER = 1 << 30


def dart_round(x):
    """Dart `double.round()`: polovina od nuly (Python round je bankéřské)."""
    r = math.floor(abs(x))
    if abs(x) - r >= 0.5:
        r += 1
    return int(math.copysign(r, x))


def stats(ptype, level):
    hp, dmg, interval, defense, weapon = BASE[ptype]
    if level > 0:
        lv = min(max(level, 1), 3)
        hp, dmg = dart_round(hp * HP_MUL[lv]), dart_round(dmg * DMG_MUL[lv])
        interval = dart_round(interval * SPD_MUL[lv])
    return hp, dmg, interval, defense, weapon


def time_to_kill(shooter, target):
    net = shooter[1] - dart_round(shooter[1] * target[3])
    if net <= 0:
        return NEVER
    shots = math.ceil(target[0] / net)
    return shots * shooter[2] + dart_round(1 / SPEED[shooter[4]])


def win_probability(att_type, def_type, att_levels, def_levels):
    a = stats(att_type, att_levels[att_type])
    d = stats(def_type, def_levels[def_type])
    at, dt = time_to_kill(a, d), time_to_kill(d, a)
    adv = (dt - at) / ((dt + at) / 2)
    p = 1 / (1 + math.exp(-3 * adv))
    return min(max(p, 0.1), 0.9)


def table(att_levels=(0,) * 6, def_levels=(0,) * 6):
    """[útočník, obránce] → P(útočník vyhraje), typy v pořadí PieceType."""
    return np.array([[win_probability(a, d, att_levels, def_levels) for d in range(6)]
                     for a in range(6)], np.float64)


def odds_pair(white_levels=(0,) * 6, black_levels=(0,) * 6):
    """odds[barva útočníka] pro rules.step_random: [2, 6, 6]."""
    return np.stack([table(white_levels, black_levels), table(black_levels, white_levels)])
