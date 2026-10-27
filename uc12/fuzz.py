"""UC12: fuzz ekvivalence Dart engine (tools/engine_cli.dart) vs JAX port (orbitron_py).

Tři zkoušky:
1. replay — Dart hraje náhodné partie (`fuzz`), JAX je dávkově (vmap) přehrává podle zaznamenaných akcí
   a výsledků bitev a v každé pozici porovná FNV-1a hash stavu + seřazených legálních tahů
   (stejné bajty jako `stateHash` v CLI), konec partie, vítěze a typ vítězství. Režim `start` začíná
   z náhodných platných formací (čtvrtina výchozích), `random` z náhodných pozic (Last Warrior, věže
   před jezdci, pěšci na poslední řadě…).
2. interactive — Python vybírá tahy sám a posílá je CLI po jednom (`move`), porovná desku i tahy.
3. odds — tabulky BattleOdds pro náhodné úrovně upgradů.

Použití:
  python uc12/fuzz.py --cli OrbitronTactics/build/engine_cli --games 100000 --out bench/results/UC12_fuzz.json
"""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from orbitron_py import odds as O  # noqa: E402
from orbitron_py import rules as R  # noqa: E402

FNV_OFFSET, FNV_PRIME = 0x811C9DC5, 0x01000193
ORDER = np.argsort(R.MOVE_ID_NP, kind="stable")  # akce seřazené podle id (neplatné −1 vpředu)
SORTED_IDS = R.MOVE_ID_NP[ORDER]

legal_b = jax.jit(jax.vmap(R.legal_mask))
step_b = jax.jit(jax.vmap(R.step))


def run_cli(cli, cmd):
    """Jeden příkaz s víceřádkovou odpovědí (fuzz/oddsFuzz) → seznam záznamů."""
    p = subprocess.run([cli], input=json.dumps(cmd) + "\n", capture_output=True, text=True, check=True)
    out = [json.loads(line) for line in p.stdout.splitlines() if line]
    assert "done" in out[-1], out[-1]
    return out[:-1]


def dart_games(cli, seed, n, mode, workers, max_plies=R.MAX_PLIES):
    chunk = (n + workers - 1) // workers
    with cf.ThreadPoolExecutor(workers) as ex:
        parts = ex.map(lambda i: run_cli(cli, {"cmd": "fuzz", "seed": seed + i * chunk,
                                               "games": min(chunk, n - i * chunk), "mode": mode,
                                               "maxPlies": max_plies}),
                       [i for i in range(workers) if i * chunk < n])
        return [g for part in parts for g in part]


def state_hashes(st, mask):
    """FNV-1a 32 bit přes bajty stavu a seřazené legální tahy, dávkově (numpy)."""
    cells = np.asarray(st.cells, np.int64)
    mask = np.asarray(mask)[:, ORDER]
    b = cells.shape[0]
    n = mask.sum(1)
    idx = np.argsort(~mask, axis=1, kind="stable")
    ids = SORTED_IDS[idx]
    mc = np.asarray(st.move_count, np.int64)
    fixed = np.concatenate([
        cells,
        np.asarray(st.turn, np.int64)[:, None],
        np.asarray(st.finished, np.int64)[:, None],
        np.asarray(st.winner, np.int64)[:, None],
        np.asarray(st.victory, np.int64)[:, None],
        (mc & 0xFF)[:, None], ((mc >> 8) & 0xFF)[:, None],
        np.asarray(st.royals, np.int64),
        (n & 0xFF)[:, None], ((n >> 8) & 0xFF)[:, None],
    ], axis=1)
    h = np.full(b, FNV_OFFSET, np.uint64)
    for j in range(fixed.shape[1]):
        h = ((h ^ fixed[:, j].astype(np.uint64)) * FNV_PRIME) & 0xFFFFFFFF
    maxn = int(n.max()) if b else 0
    for j in range(maxn):
        live = j < n
        v = ids[:, j].astype(np.int64)
        for byte in (v & 0xFF, (v >> 8) & 0xFF):
            nh = ((h ^ byte.astype(np.uint64)) * FNV_PRIME) & 0xFFFFFFFF
            h = np.where(live, nh, h)
    return h.astype(np.int64), n


def stack_states(games):
    sts = [R.new_state(np.asarray(g["cells"], np.int8), g["variant"], g["turn"], g["royals"]) for g in games]
    return jax.tree.map(lambda *xs: jnp.stack(xs), *sts)


def replay(games, stats):
    """Přehraje dávku Dart partií v JAX; vrátí seznam rozdílů."""
    diffs = []
    b = len(games)
    st = stack_states(games)
    T = max(len(g["hashes"]) for g in games)
    acts = np.zeros((b, T), np.int32)
    for i, g in enumerate(games):
        for t, a in enumerate(g["actions"]):
            acts[i, t] = R.MOVE_ID_TO_ACTION[a % 4096]
    alive = np.ones(b, bool)
    for t in range(T):
        mask = legal_b(st)
        h, n = state_hashes(st, mask)
        mask_np = np.asarray(mask)
        term = np.asarray(jax.vmap(R.terminal)(st, mask))
        for i in np.nonzero(alive)[0]:
            g = games[i]
            last = t == len(g["hashes"]) - 1
            if h[i] != g["hashes"][t]:
                diffs.append({"seed": g["seed"], "mode": g.get("mode"), "ply": t, "what": "hash"})
                alive[i] = False
                continue
            stats["positions"] += 1
            stats["legal_moves"] += int(n[i])
            stats["max_legal"] = max(stats["max_legal"], int(n[i]))
            if bool(term[i]) != last:
                diffs.append({"seed": g["seed"], "mode": g.get("mode"), "ply": t, "what": "terminal"})
                alive[i] = False
                continue
            if last:
                ok = (int(st.winner[i]) == (0 if g["winner"] is None else g["winner"] + 1)
                      and int(st.victory[i]) == g["victory"])
                if not ok:
                    diffs.append({"seed": g["seed"], "mode": g.get("mode"), "ply": t, "what": "result"})
                stats["end_" + g["end"]] += 1
                if g["end"] == "victory":
                    stats["victory_" + ["", "powerFields", "royalElimination", "infiltration"][g["victory"]]] += 1
                alive[i] = False
                continue
            a = g["actions"][t]
            if not mask_np[i, acts[i, t]]:
                diffs.append({"seed": g["seed"], "mode": g.get("mode"), "ply": t, "what": "dart move not legal in JAX"})
                alive[i] = False
                continue
            stats["plies"] += 1
            if int(st.cells[i, R.TARGET_NP.reshape(-1)[acts[i, t]]]) > 0:
                stats["captures"] += 1
                stats["attacker_won" if a >= 4096 else "defender_won"] += 1
        if not alive.any():
            break
        wins_t = np.array([(games[i]["actions"][t] >= 4096) if t < len(games[i]["actions"]) else True
                           for i in range(b)])
        before = np.asarray(st.cells)
        st = step_b(st, jnp.asarray(acts[:, t]), jnp.asarray(wins_t))
        after = np.asarray(st.cells)
        stats["last_warrior_made"] += int(np.maximum((after > 12).sum(1) - (before > 12).sum(1), 0).sum())
    return diffs


def interactive(cli, games, seed):
    """Python vybírá tahy, CLI je hraje po jednom; porovná desku, tahy a hash v každé pozici."""
    rng = np.random.default_rng(seed)
    odds = O.odds_pair()
    p = subprocess.Popen([cli], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    legal1 = jax.jit(R.legal_mask)
    step1 = jax.jit(R.step)

    def ask(cmd):
        p.stdin.write(json.dumps(cmd) + "\n")
        return json.loads(p.stdout.readline())

    diffs, plies = [], 0
    for g in range(games):
        variant = int(rng.integers(8))
        white = [int(x) for x in rng.permutation(R.start_cells()[:16] - 1)]
        black = [int(x) for x in rng.permutation(R.start_cells()[48:] - 7)]
        names = ["pawn", "rook", "knight", "bishop", "queen", "king"]
        d = ask({"cmd": "new", "variant": variant, "white": [names[x] for x in white],
                 "black": [names[x] for x in black]})
        st = R.new_state(R.start_cells(white, black), variant)
        while True:
            mask = legal1(st)
            ours = sorted(int(R.MOVE_ID_NP[a]) for a in np.nonzero(np.asarray(mask))[0])
            theirs = sorted(m["id"] for m in d["legal"])
            h, _ = state_hashes(jax.tree.map(lambda x: x[None], st), np.asarray(mask)[None])
            if d["cells"] != [int(x) for x in st.cells] or ours != theirs or int(h[0]) != d["hash"]:
                diffs.append({"game": g, "ply": int(st.plies), "what": "interactive"})
                break
            if bool(R.terminal(st, mask)) != d["over"]:
                diffs.append({"game": g, "ply": int(st.plies), "what": "interactive terminal"})
                break
            if d["over"]:
                break
            a = int(rng.choice(np.nonzero(np.asarray(mask))[0]))
            pw = float(R.capture_odds(st, a, jnp.asarray(odds)))
            won = bool(rng.random() < pw)
            mid = int(R.MOVE_ID_NP[a])
            d = ask({"cmd": "move", "from": divmod(mid // 64, 8), "to": divmod(mid % 64, 8),
                     "battle": "attacker" if won else "defender"})
            st = step1(st, a, won)
            plies += 1
    p.stdin.close()
    p.wait()
    return {"games": games, "plies": plies, "diffs": diffs}


def odds_check(cli, count, seed):
    rows = run_cli(cli, {"cmd": "oddsFuzz", "seed": seed, "count": count})
    names = ["pawn", "rook", "knight", "bishop", "queen", "king"]
    worst = 0.0
    bad = 0
    for r in rows:
        ours = O.table([r["attacker"][n] for n in names], [r["defender"][n] for n in names])
        err = float(np.abs(ours - np.asarray(r["odds"])).max())
        worst = max(worst, err)
        bad += err > 1e-12
    return {"tables": count, "pairs": count * 36, "max_abs_diff": worst, "tables_over_1e-12": int(bad)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--games", type=int, default=1000, help="partie z formací")
    ap.add_argument("--random-games", type=int, default=None, help="partie z náhodných pozic (výchozí games/5)")
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--interactive", type=int, default=200)
    ap.add_argument("--odds", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out")
    args = ap.parse_args()
    random_games = args.games // 5 if args.random_games is None else args.random_games

    t0 = time.time()
    keys = ["positions", "plies", "legal_moves", "max_legal", "captures", "attacker_won", "defender_won",
            "last_warrior_made", "end_victory", "end_cap", "end_noMoves", "victory_powerFields",
            "victory_royalElimination", "victory_infiltration"]
    report = {}
    all_diffs = []
    for mode, total, seed0 in (("start", args.games, args.seed), ("random", random_games, args.seed + 10**8)):
        stats = dict.fromkeys(keys, 0)
        stats["games"] = total
        done = 0
        with cf.ThreadPoolExecutor(1) as pre:
            nxt = pre.submit(dart_games, args.cli, seed0, min(args.batch, total), mode, args.workers) if total else None
            while done < total:
                n = min(args.batch, total - done)
                games = nxt.result()
                done += n
                if done < total:
                    nxt = pre.submit(dart_games, args.cli, seed0 + done, min(args.batch, total - done), mode,
                                     args.workers)
                for g in games:
                    g["mode"] = mode
                d = replay(games, stats)
                all_diffs += d
                print(f"[{mode}] {done}/{total} her, {stats['plies']} tahů, rozdílů {len(all_diffs)}, "
                      f"{time.time() - t0:.0f} s", flush=True)
        report[mode] = stats
    inter = interactive(args.cli, args.interactive, args.seed) if args.interactive else None
    odds = odds_check(args.cli, args.odds, args.seed) if args.odds else None
    total_games = sum(report[m]["games"] for m in report)
    result = {
        "uc": "UC12",
        "what": "Dart engine (OrbitronTactics tools/engine_cli.dart) vs JAX port uc12/orbitron_py",
        "games": total_games,
        "plies": sum(report[m]["plies"] for m in report),
        "positions_compared": sum(report[m]["positions"] for m in report),
        "legal_moves_compared": sum(report[m]["legal_moves"] for m in report),
        "differences": len(all_diffs) + (len(inter["diffs"]) if inter else 0),
        "by_mode": report,
        "interactive": inter,
        "odds": odds,
        "diff_samples": all_diffs[:50],
        "action_space": {"squares": R.N_SQ, "displacements": R.N_DISP, "actions": R.N_ACTIONS},
        "seconds": round(time.time() - t0, 1),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
    }
    print(json.dumps({k: v for k, v in result.items() if k != "diff_samples"}, indent=1))
    if all_diffs:
        print("první rozdíly:", all_diffs[:10])
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
