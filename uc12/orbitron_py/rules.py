"""UC12: pravidla OrbitronTactics v JAX (port lib/core/game_logic/, repo lioilsources/OrbitronTactics).

Čistě funkční, pgx-style: stav = pár tenzorů, `legal_mask` i `step` jdou pod `jax.jit` a `jax.vmap`.

Deska: pole 64 buněk, index = řádek * 8 + sloupec, řádek 0 = zadní řada bílého (jako `Position`).
Kód buňky (stejný jako `cellCode` v tools/engine_cli.dart): 0 prázdno, jinak
1 + typ + 6 * barva + 12, je-li figura Last Warrior. Typy v pořadí `PieceType`:
pawn 0, rook 1, knight 2, bishop 3, queen 4, king 5; barva white 0, black 1.

Akce = (pole, posun): 64 polí × 40 posunů = 2560 (viz DISPLACEMENTS). Posun pokrývá všechno, co umí
kterákoli figura kterékoli barvy: 4 rovné směry × 1–4, 4 diagonály × 1–4, 8 skoků jezdce.
Pro srovnání s Dartem `action_to_move_id` → from * 64 + to (to je id v engine_cli).

Bitva při braní je náhodný uzel: `step(state, action, attacker_wins)` je deterministický; náhodu
dodá `step_random` (los podle BattleOdds) nebo MCTS (viz uc12/az.py). Konec partie jako v AI self-play
testu repa: vítězství, strana na tahu nemá tah, nebo MAX_PLIES půltahů (obojí bez vítěze).
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

PAWN, ROOK, KNIGHT, BISHOP, QUEEN, KING = range(6)
WHITE, BLACK = 0, 1
MAX_PLIES = 300
N_SQ = 64
HIST = 4  # kolik desek (včetně aktuální) vidí síť

# Pole silových polí, 8 variant z PowerFieldGenerator.variants (doslovně, i s asymetriemi originálu).
PF_VARIANTS = [
    [(3, 3), (3, 4), (4, 3), (4, 4), (3, 5)],
    [(2, 3), (3, 2), (3, 5), (4, 4), (5, 4)],
    [(1, 1), (2, 2), (5, 5), (6, 6), (3, 4)],
    [(2, 0), (2, 7), (5, 0), (5, 7), (3, 4)],
    [(2, 3), (2, 4), (5, 3), (5, 4), (4, 3)],
    [(1, 2), (1, 5), (6, 2), (6, 5), (4, 3)],
    [(1, 3), (3, 1), (4, 6), (6, 4), (3, 4)],
    [(1, 4), (3, 3), (4, 4), (6, 3), (4, 3)],
]

# --- Akční prostor ------------------------------------------------------------------------------

STRAIGHT = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
KNIGHT_JUMPS = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
DISPLACEMENTS = ([(dr * k, dc * k) for dr, dc in STRAIGHT for k in range(1, 5)]
                 + [(dr * k, dc * k) for dr, dc in DIAGONAL for k in range(1, 5)]
                 + KNIGHT_JUMPS)
N_DISP = len(DISPLACEMENTS)          # 40
N_ACTIONS = N_SQ * N_DISP            # 2560
D_INDEX = {d: i for i, d in enumerate(DISPLACEMENTS)}

# Pravidla posunu pro jednu figuru: kam smí a za jaké podmínky.
NONE, SLIDE, JUMP, QUIET, KNIGHT_FWD = range(5)
# SLIDE: cesta mezi poli prázdná, cíl ne vlastní (PathChecker.getReachableSquares)
# JUMP: cíl ne vlastní (jezdec, snipe střelce, Last Warrior, diagonála pěšce vpřed)
# QUIET: cíl prázdný (pěšec do strany a dozadu — nebere)
# KNIGHT_FWD: vysoké L vpřed — jako JUMP, ale ne s věží (libovolné barvy) přímo před jezdcem


def _rules_for(ptype, color, last_warrior):
    f = 1 if color == WHITE else -1   # PlayerColorX.forwardDir
    r = {}
    if last_warrior:  # LastWarriorValidator: +1 všemi 8 směry, jakýkoli typ
        for dr, dc in STRAIGHT + DIAGONAL:
            r[(dr, dc)] = JUMP
        return r

    def slide(dirs, reach):
        for dr, dc in dirs:
            for k in range(1, reach + 1):
                r[(dr * k, dc * k)] = SLIDE

    if ptype == PAWN:
        r[(0, 1)] = r[(0, -1)] = QUIET
        r[(-f, 0)] = QUIET
        r[(f, 1)] = r[(f, -1)] = JUMP
    elif ptype == ROOK:
        slide(STRAIGHT, 4)
    elif ptype == KNIGHT:
        r[(2 * f, 1)] = r[(2 * f, -1)] = KNIGHT_FWD
        for dc in (2, -2):
            r[(f, dc)] = JUMP
            r[(-f, dc)] = JUMP
    elif ptype == BISHOP:
        slide(DIAGONAL, 3)
        r[(3 * f, 0)] = JUMP  # snipe: přesně 3 vpřed, přes figury
    elif ptype == QUEEN:
        slide(DIAGONAL, 4)
        slide(STRAIGHT, 2)
    elif ptype == KING:
        slide(STRAIGHT, 4)
        slide(DIAGONAL, 2)
    return r


def _tables():
    rule = np.zeros((25, N_DISP), np.int8)          # [kód buňky, posun]
    for code in range(1, 25):
        v = code - 1
        for d, kind in _rules_for(v % 6, (v // 6) % 2, v >= 12).items():
            rule[code, D_INDEX[d]] = kind
    target = np.full((N_SQ, N_DISP), -1, np.int32)
    path = np.full((N_SQ, N_DISP, 3), N_SQ, np.int32)  # N_SQ = vždy prázdná pomocná buňka
    for s in range(N_SQ):
        r0, c0 = divmod(s, 8)
        for i, (dr, dc) in enumerate(DISPLACEMENTS):
            r1, c1 = r0 + dr, c0 + dc
            if not (0 <= r1 < 8 and 0 <= c1 < 8):
                continue
            target[s, i] = r1 * 8 + c1
            if i < 32:  # přímky a diagonály: mezilehlá pole
                k = max(abs(dr), abs(dc))
                sr, sc = dr // k, dc // k
                for j in range(1, k):
                    path[s, i, j - 1] = (r0 + sr * j) * 8 + c0 + sc * j
    front = np.full((N_SQ, 2), N_SQ, np.int32)     # pole přímo před figurou dané barvy
    for s in range(N_SQ):
        r0, c0 = divmod(s, 8)
        for color, f in ((WHITE, 1), (BLACK, -1)):
            if 0 <= r0 + f < 8:
                front[s, color] = (r0 + f) * 8 + c0
    move_id = np.where(target >= 0, np.arange(N_SQ)[:, None] * 64 + target, -1).reshape(-1)
    # zrcadlení o 180° (Position.mirrored): pole s → 63 - s, posun (dr, dc) → (-dr, -dc)
    neg = np.array([D_INDEX[(-dr, -dc)] for dr, dc in DISPLACEMENTS], np.int32)
    mirror_action = ((N_SQ - 1 - np.arange(N_SQ))[:, None] * N_DISP + neg[None, :]).reshape(-1)
    pf = np.zeros((len(PF_VARIANTS), N_SQ), bool)
    for v, cells in enumerate(PF_VARIANTS):
        for r0, c0 in cells:
            pf[v, r0 * 8 + c0] = True
    return rule, target, path, front, move_id, mirror_action, pf


RULE_NP, TARGET_NP, PATH_NP, FRONT_NP, MOVE_ID_NP, MIRROR_ACTION_NP, PF_NP = _tables()
RULE, TARGET, PATH, FRONT = map(jnp.asarray, (RULE_NP, TARGET_NP, PATH_NP, FRONT_NP))
MIRROR_ACTION, PF = jnp.asarray(MIRROR_ACTION_NP), jnp.asarray(PF_NP)


def action_to_move_id(a):
    """Akce 0..2559 → from * 64 + to (id z engine_cli.dart), −1 mimo desku."""
    return MOVE_ID_NP[a]


MOVE_ID_TO_ACTION = {int(m): a for a, m in enumerate(MOVE_ID_NP) if m >= 0}

# --- Stav -----------------------------------------------------------------------------------


class State(NamedTuple):
    cells: jnp.ndarray       # int8[64]
    turn: jnp.ndarray        # int8, 0 white / 1 black
    finished: jnp.ndarray    # bool, GamePhase.finished
    winner: jnp.ndarray      # int8, 0 nikdo, 1 white, 2 black
    victory: jnp.ndarray     # int8, 0 nic, 1 power fields, 2 royal elimination, 3 infiltration
    move_count: jnp.ndarray  # int32, GameState.moveCount
    royals: jnp.ndarray      # bool[4], hasKing/hasQueen bílého, pak černého (Player)
    plies: jnp.ndarray       # int32, odehrané půltahy (limit MAX_PLIES)
    pf: jnp.ndarray          # bool[64], silová pole
    hist: jnp.ndarray        # int8[HIST-1, 64], předchozí desky (nejnovější první)


def new_state(cells, variant, turn=WHITE, royals=(True, True, True, True)):
    cells = jnp.asarray(cells, jnp.int8)
    return State(
        cells=cells, turn=jnp.int8(turn), finished=jnp.bool_(False), winner=jnp.int8(0),
        victory=jnp.int8(0), move_count=jnp.int32(0), royals=jnp.asarray(royals, bool),
        plies=jnp.int32(0), pf=PF[variant], hist=jnp.zeros((HIST - 1, N_SQ), jnp.int8))


DEFAULT_BACK = [ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK]


def start_cells(white=None, black=None):
    """Deska po rozestavení: white = 16 typů pro řádky 0–1, black pro řádky 6–7 (jako příkaz `new`)."""
    if white is None:
        white = DEFAULT_BACK + [PAWN] * 8
    if black is None:
        black = [PAWN] * 8 + DEFAULT_BACK
    cells = np.zeros(N_SQ, np.int8)
    cells[0:16] = np.asarray(white) + 1
    cells[48:64] = np.asarray(black) + 7
    return cells


def _ptype(code):
    return (code.astype(jnp.int32) - 1) % 6


def _pcolor(code):
    return ((code.astype(jnp.int32) - 1) // 6) % 2


# --- Legální tahy ---------------------------------------------------------------------------


def legal_mask(s: State):
    """bool[2560]: legální akce strany na tahu (MoveGenerator / MoveValidator.getLegalMoves)."""
    board = jnp.concatenate([s.cells, jnp.zeros(1, jnp.int8)])   # [65], buňka 64 je vždy prázdná
    occ = board > 0
    mine = occ & (_pcolor(board) == s.turn)
    rule = RULE[s.cells.astype(jnp.int32)]                       # [64, 40]
    rule = jnp.where(mine[:N_SQ, None], rule, NONE)
    valid = TARGET >= 0
    tgt = jnp.where(valid, TARGET, N_SQ)
    friendly = mine[tgt]
    empty = ~occ[tgt]
    clear = ~jnp.any(occ[PATH], axis=-1)
    front = board[FRONT[:, s.turn]]                              # [64]
    rook_front = (front > 0) & (_ptype(front) == ROOK)
    ok = jnp.select(
        [rule == SLIDE, rule == JUMP, rule == QUIET, rule == KNIGHT_FWD],
        [clear & ~friendly, ~friendly, empty, ~friendly & ~rook_front[:, None]],
        False)
    return (ok & valid & ~s.finished).reshape(-1)


# --- Tah ------------------------------------------------------------------------------------


def _find_first(cells, ptype, color):
    """Index prvního pole (po řádcích) s figurou daného typu a barvy, nebo −1 (BoardState.findPieces)."""
    hit = (cells > 0) & (_ptype(cells) == ptype) & (_pcolor(cells) == color)
    return jnp.where(jnp.any(hit), jnp.argmax(hit), -1)


def step(s: State, action, attacker_wins):
    """Zahraje legální akci. Při braní rozhoduje `attacker_wins` (výsledek bitvy v aréně).

    Útočník vyhraje → GameEngine.applyMove: tah, hasKing/hasQueen poraženého, Last Warrior, kontrola
    vítězství pro táhnoucího. Obránce vyhraje → GameEngine.resolveBattle: útočník zmizí, tah přejde,
    bez kontroly vítězství, bez Last Warrior a bez změny hasKing/hasQueen (tak to dělá originál).
    """
    action = jnp.asarray(action, jnp.int32)
    frm = action // N_DISP
    to = TARGET.reshape(-1)[action]
    piece = s.cells[frm]
    captured = s.cells[to]
    is_capture = captured > 0
    lost = is_capture & ~jnp.asarray(attacker_wins, bool)
    mover = s.turn.astype(jnp.int32)
    opp = 1 - mover

    # --- útočník vyhrál nebo tichý tah
    moved = s.cells.at[frm].set(0).at[to].set(piece)
    ctype, ccolor = _ptype(captured), _pcolor(captured)
    royal = is_capture & ((ctype == KING) | (ctype == QUEEN))
    royals = s.royals.at[2 * ccolor + jnp.where(ctype == KING, 0, 1)].set(
        jnp.where(royal, False, s.royals[2 * ccolor + jnp.where(ctype == KING, 0, 1)]))
    survivor = _find_first(moved, jnp.where(ctype == KING, QUEEN, KING), ccolor)
    lw_code = jnp.where(moved[survivor] > 12, moved[survivor], moved[survivor] + 12)
    moved = jnp.where(royal & (survivor >= 0), moved.at[survivor].set(lw_code), moved)

    # VictoryChecker.checkVictory pro táhnoucího, v tomto pořadí
    t, c = _ptype(moved), _pcolor(moved)
    occ = moved > 0
    opp_royals = jnp.any(occ & (c == opp) & ((t == KING) | (t == QUEEN)))
    rank = jnp.where(mover == WHITE, 7, 0)
    infiltr = jnp.any((occ & (c == mover) & (t == PAWN)).reshape(8, 8)[rank])
    domination = jnp.all(jnp.where(s.pf, occ & (c == mover), True))
    victory = jnp.select([~opp_royals, infiltr, domination], [2, 3, 1], 0).astype(jnp.int8)
    won = victory > 0

    # --- obránce vyhrál
    removed = s.cells.at[frm].set(0)

    cells = jnp.where(lost, removed, moved)
    victory = jnp.where(lost, jnp.int8(0), victory)
    won = won & ~lost
    return State(
        cells=cells,
        turn=jnp.where(won, s.turn, (1 - s.turn).astype(jnp.int8)),
        finished=won,
        winner=jnp.where(won, (mover + 1).astype(jnp.int8), jnp.int8(0)),
        victory=victory,
        move_count=s.move_count + 1,
        royals=jnp.where(lost, s.royals, royals),
        plies=s.plies + 1,
        pf=s.pf,
        hist=jnp.concatenate([s.cells[None], s.hist[:-1]], axis=0),
    )


def terminal(s: State, mask=None):
    """Konec partie: vítězství, žádný legální tah, nebo limit půltahů."""
    if mask is None:
        mask = legal_mask(s)
    return s.finished | ~jnp.any(mask) | (s.plies >= MAX_PLIES)


def capture_odds(s: State, action, odds):
    """Pravděpodobnost, že útočník vyhraje bitvu; 1 u tichého tahu. odds[barva] = tabulka 6×6."""
    frm = action // N_DISP
    to = TARGET.reshape(-1)[action]
    a, d = s.cells[frm], s.cells[to]
    p = odds[s.turn.astype(jnp.int32), _ptype(a), _ptype(d)]
    return jnp.where(d > 0, p, 1.0)


def step_random(s: State, action, key, odds):
    """Tah s bitvou losovanou podle BattleOdds (odds[barva útočníka] = tabulka 6×6)."""
    p = capture_odds(s, action, odds)
    return step(s, action, jax.random.uniform(key) < p)
