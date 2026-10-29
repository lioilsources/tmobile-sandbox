"""UC12: Elo liga — AZ checkpointy proti Dart AI (Easy / Medium / Hard) přes tools/engine_cli.

Rozhodčí je Dart engine (originální pravidla i BattleOdds): každá partie má vlastní proces engine_cli,
tahy Dart AI dává příkaz `ai`, bitvy losuje Dart (`move` se `seed`). AZ hráč drží zrcadlový stav
v JAX portu a po každém tahu ověří, že deska sedí s Dartem (jinak partie skončí chybou `desync`).
AZ tahy se počítají dávkově přes všechny rozehrané partie (jeden jit pro checkpoint), jen na CPU,
takže liga může běžet vedle tréninku.

Výsledky se přidávají do `--results` (jsonl, partie po partii) — opakované spuštění dohraje jen to,
co k počtu `--games` chybí, a Elo se počítá ze všech uložených partií: Bradley–Terry (remíza = ½),
Easy = 0 (lze změnit `--anchor`), slabý prior σ = 1000 Elo, 95% CI bootstrapem po partiích.

  python uc12/league.py --cli /srv/work/w2/uc12/bin/engine_cli --ckpts '/cold/w2/uc12/ckpt/nets/net_*.msgpack' \\
      --max-ckpts 6 --games 100 --anchor-games 100 --sims 32 --out /cold/w2/uc12/league
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import concurrent.futures as cf  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import az  # noqa: E402
from orbitron_py import rules as R  # noqa: E402

TYPE_NAMES = ["pawn", "rook", "knight", "bishop", "queen", "king"]
DART = ("easy", "medium", "hard")


# --- Dart rozhodčí --------------------------------------------------------------------------

class Engine:
    def __init__(self, cli):
        self.p = subprocess.Popen([cli], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                                  start_new_session=True)

    def __call__(self, **cmd):
        self.p.stdin.write(json.dumps(cmd) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError(f"engine_cli skončil (kód {self.p.poll()})")
        r = json.loads(line)
        if not r.get("ok", False):
            raise RuntimeError(f"engine_cli: {r.get('error')} pro {cmd}")
        return r

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(5)
        except Exception:  # noqa: BLE001
            self.p.kill()


def game_setup(seed):
    """Deterministický start partie ze seedu: varianta, formace (½ výchozí, ½ náhodná permutace)."""
    rng = np.random.default_rng(seed)
    variant = int(rng.integers(len(R.PF_VARIANTS)))
    army = [R.PAWN] * 8 + [R.ROOK] * 2 + [R.KNIGHT] * 2 + [R.BISHOP] * 2 + [R.QUEEN, R.KING]
    w = None if rng.random() < 0.5 else list(rng.permutation(army))
    b = None if rng.random() < 0.5 else list(rng.permutation(army))
    return variant, w, b


def move_seed(seed, ply):
    return int.from_bytes(hashlib.blake2b(f"{seed}:{ply}".encode(), digest_size=4).digest(), "little") & 0x7FFFFFFF


# --- AZ hráč --------------------------------------------------------------------------------

class AzPlayer:
    def __init__(self, path, sims, batch, chance):
        self.path, self.sims, self.batch = path, sims, batch
        self.net, self.variables, self.meta = az.load_net(path)
        self.name = "az:" + os.path.basename(path)[:-len(".msgpack")]
        search = az.Search(self.net, chance)

        @jax.jit
        def act(variables, states, key):
            if sims == 0:
                logits, _, _ = az.evaluate(self.net, variables, states)
                return jnp.argmax(logits, -1)
            out, *_ = az.run_search(search, variables, states, key, sims, gumbel_scale=0.0)
            return out.action

        self._act = act
        self.key = jax.random.PRNGKey(0)

    def actions(self, states):
        """states: seznam R.State → seznam akcí (dávka doplněná na pevnou velikost → jeden jit)."""
        out = []
        for i in range(0, len(states), self.batch):
            part = states[i:i + self.batch]
            pad = part + [part[0]] * (self.batch - len(part))
            st = jax.tree.map(lambda *x: jnp.stack(x), *pad)
            self.key, k = jax.random.split(self.key)
            out += [int(a) for a in np.asarray(self._act(self.variables, st, k))[:len(part)]]
        return out


_step1 = jax.jit(R.step)


class Game:
    def __init__(self, cli, white, black, seed, max_plies):
        self.white, self.black, self.seed = white, black, seed
        self.eng = Engine(cli)
        variant, w, b = game_setup(seed)
        r = self.eng(cmd="new", variant=variant, maxPlies=max_plies,
                     white=None if w is None else [TYPE_NAMES[t] for t in w],
                     black=None if b is None else [TYPE_NAMES[t] for t in b])
        self.state = R.new_state(R.start_cells(w, b), variant)
        assert list(np.asarray(self.state.cells)) == r["cells"], "start nesedí s Dartem"
        self.last = r
        self.plies = 0
        self.t0 = time.time()
        self.error = None

    @property
    def to_move(self):
        return self.white if self.last["turn"] == "white" else self.black

    def dart_move(self, level):
        m = self.eng(cmd="ai", level=level, seed=move_seed(self.seed, self.plies))["move"]
        return m["from"], m["to"]

    def play(self, frm, to):
        r = self.eng(cmd="move", **{"from": frm, "to": to}, seed=move_seed(self.seed, self.plies + 10_000))
        a = R.MOVE_ID_TO_ACTION[(frm[0] * 8 + frm[1]) * 64 + to[0] * 8 + to[1]]
        self.state = _step1(self.state, a, r["battle"] != "defender")
        self.plies += 1
        self.last = r
        if list(np.asarray(self.state.cells)) != r["cells"]:
            self.error = "desync"

    def result(self):
        w = self.last["winner"]
        return 1.0 if w == "white" else 0.0 if w == "black" else 0.5


def action_to_move(a):
    mid = int(R.action_to_move_id(a))
    f, t = divmod(mid, 64)
    return [f // 8, f % 8], [t // 8, t % 8]


def play_batch(cli, pairings, az_players, workers, max_plies, log):
    """pairings: [(white, black, seed)] → záznamy partií. Všechny partie běží souběžně."""
    games = []
    with cf.ThreadPoolExecutor(workers) as ex:
        games = list(ex.map(lambda p: Game(cli, p[0], p[1], p[2], max_plies), pairings))
        active = list(games)
        while active:
            # AZ tahy dávkově po hráčích
            moves = {}
            by_az = {}
            for g in active:
                if g.to_move not in DART:
                    by_az.setdefault(g.to_move, []).append(g)
            for name, gs in by_az.items():
                for g, a in zip(gs, az_players[name].actions([g.state for g in gs])):
                    if not bool(R.legal_mask(g.state)[a]):
                        g.error = "illegal_az"
                    moves[id(g)] = action_to_move(a)
            dart = [g for g in active if g.to_move in DART]
            for g, mv in zip(dart, ex.map(lambda g: g.dart_move(g.to_move), dart)):
                moves[id(g)] = mv
            list(ex.map(lambda g: g.play(*moves[id(g)]) if g.error is None else None, active))
            done = [g for g in active if g.last["over"] or g.error]
            for g in done:
                rec = {"white": g.white, "black": g.black, "seed": g.seed, "score_white": g.result(),
                       "winner": g.last["winner"], "victory": g.last["victory"], "plies": g.plies,
                       "sec": round(time.time() - g.t0, 1), "error": g.error, "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
                g.eng.close()
                log(rec)
            active = [g for g in active if not (g.last["over"] or g.error)]
    return games


# --- Elo ------------------------------------------------------------------------------------

def fit_elo(players, recs, anchor, prior_sigma=1000.0, iters=3000):
    idx = {p: i for i, p in enumerate(players)}
    if not recs:
        return np.zeros(len(players))
    w = np.array([idx[r["white"]] for r in recs])
    b = np.array([idx[r["black"]] for r in recs])
    s = np.array([r["score_white"] for r in recs], float)
    k = math.log(10) / 400
    x = np.zeros(len(players))
    lr = 50.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-k * (x[w] - x[b])))
        g = np.zeros_like(x)
        np.add.at(g, w, k * (s - p))
        np.add.at(g, b, -k * (s - p))
        g -= x / prior_sigma ** 2
        h = np.zeros_like(x)
        np.add.at(h, w, k * k * p * (1 - p))
        np.add.at(h, b, k * k * p * (1 - p))
        h += 1 / prior_sigma ** 2
        step = g / h
        x += np.clip(step, -lr, lr)
        x -= x[idx[anchor]]
        if np.abs(step).max() < 1e-4:
            break
    return x


def elo_table(recs, anchor, boot, meta):
    recs = [r for r in recs if not r.get("error")]
    players = sorted({r["white"] for r in recs} | {r["black"] for r in recs},
                     key=lambda p: (p not in DART, DART.index(p) if p in DART else 0, p))
    if anchor not in players:
        anchor = players[0]
    elo = fit_elo(players, recs, anchor)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(boot):
        sample = [recs[i] for i in rng.integers(0, len(recs), len(recs))]
        bs.append(fit_elo(players, sample, anchor, iters=500))
    bs = np.array(bs) if bs else np.zeros((1, len(players)))
    rows = []
    for i, p in enumerate(players):
        mine = [r for r in recs if p in (r["white"], r["black"])]
        score = sum(r["score_white"] if r["white"] == p else 1 - r["score_white"] for r in mine)
        vs = {}
        for r in mine:
            o = r["black"] if r["white"] == p else r["white"]
            sc = r["score_white"] if r["white"] == p else 1 - r["score_white"]
            a = vs.setdefault(o, [0.0, 0])
            a[0] += sc
            a[1] += 1
        rows.append({"player": p, "elo": round(float(elo[i])), "ci95": [round(float(np.percentile(bs[:, i], 2.5))),
                                                                        round(float(np.percentile(bs[:, i], 97.5)))],
                     "games": len(mine), "score": round(score / max(len(mine), 1), 3),
                     "vs": {o: f"{a[0]:g}/{a[1]}" for o, a in vs.items()}, **meta.get(p, {})})
    return {"anchor": anchor, "games": len(recs), "players": rows}


# --- CLI ------------------------------------------------------------------------------------

def pick_ckpts(pattern, max_n):
    files = sorted(glob.glob(pattern))
    if max_n and len(files) > max_n:
        sel = np.unique(np.linspace(0, len(files) - 1, max_n).round().astype(int))
        files = [files[i] for i in sel]
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cli", required=True, help="zkompilovaný tools/engine_cli")
    ap.add_argument("--ckpts", default="/cold/w2/uc12/ckpt/nets/net_*.msgpack", help="glob AZ sítí")
    ap.add_argument("--max-ckpts", type=int, default=6, help="rovnoměrně vybrat max N checkpointů (vč. prvního a posledního)")
    ap.add_argument("--opponents", default="easy,medium,hard")
    ap.add_argument("--games", type=int, default=100, help="partií na dvojici AZ × Dart úroveň")
    ap.add_argument("--anchor-games", type=int, default=100, help="partií na dvojici Dart × Dart (kalibrace)")
    ap.add_argument("--sims", type=int, default=32, help="simulací AZ na tah (0 = jen policy síť)")
    ap.add_argument("--chance", choices=["expect", "sample"], default="expect")
    ap.add_argument("--parallel", type=int, default=32, help="souběžných partií v jedné dávce")
    ap.add_argument("--workers", type=int, default=4, help="vláken pro Dart volání")
    ap.add_argument("--max-plies", type=int, default=R.MAX_PLIES)
    ap.add_argument("--anchor", default="easy")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--out", default="/cold/w2/uc12/league")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--elo-only", action="store_true", help="jen přepočítat Elo z uložených partií")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    res_path = os.path.join(args.out, "games.jsonl")
    recs = [json.loads(line) for line in open(res_path)] if os.path.exists(res_path) else []
    opps = [o for o in args.opponents.split(",") if o]
    ckpts = [] if args.elo_only else pick_ckpts(args.ckpts, args.max_ckpts)
    meta = {}
    az_players = {}
    for path in ckpts:
        pl = AzPlayer(path, args.sims, args.parallel, args.chance)
        az_players[pl.name] = pl
        meta[pl.name] = {k: pl.meta.get(k) for k in ("iter", "gpu_hours", "train_steps")} | {"train_games": pl.meta.get("games")}
    for r in recs:  # meta i pro starší checkpointy, které tentokrát nehrají
        for p in (r["white"], r["black"]):
            if p.startswith("az:") and p not in meta:
                j = os.path.join(os.path.dirname(args.ckpts), p[3:] + ".json")
                if os.path.exists(j):
                    m = json.load(open(j))
                    meta[p] = {k: m.get(k) for k in ("iter", "gpu_hours", "train_steps")} | {"train_games": m.get("games")}

    # rozpis: každá dvojice N partií, barvy střídavě, seed partie = seed páru (obě barvy stejný start)
    played = {}
    for r in recs:
        key = tuple(sorted((r["white"], r["black"])))
        played[key] = played.get(key, 0) + 1
    todo = []
    pairs = [(a, b, args.anchor_games) for i, a in enumerate(opps) for b in opps[i + 1:]]
    pairs += [(p, o, args.games) for p in az_players for o in opps]
    if args.elo_only:
        pairs = []
    for a, b, n in pairs:
        have = played.get(tuple(sorted((a, b))), 0)
        for j in range(have, n):
            seed = args.seed * 1_000_003 + j // 2
            todo.append((a, b, seed) if j % 2 == 0 else (b, a, seed))
    print(f"liga: {len(az_players)} AZ sítí, soupeři {opps}, k odehrání {len(todo)} partií "
          f"(uloženo {len(recs)})", flush=True)

    t0 = time.time()
    n_done = [0]
    with open(res_path, "a") as f:
        def log(rec):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            recs.append(rec)
            n_done[0] += 1
            if rec["error"]:
                print(f"CHYBA partie {rec}", file=sys.stderr, flush=True)
            if n_done[0] % 10 == 0 or n_done[0] == len(todo):
                el = time.time() - t0
                print(f"  {n_done[0]}/{len(todo)} partií, {el / 60:.1f} min, {n_done[0] / el * 60:.1f} partií/min",
                      flush=True)

        for i in range(0, len(todo), args.parallel):
            play_batch(args.cli, todo[i:i + args.parallel], az_players, args.workers, args.max_plies, log)

    table = elo_table(recs, args.anchor, args.boot, meta)
    table.update(time=time.strftime("%Y-%m-%dT%H:%M:%S"), sims=args.sims, minutes=round((time.time() - t0) / 60, 1),
                 errors=sum(1 for r in recs if r.get("error")))
    json.dump(table, open(os.path.join(args.out, "elo.json"), "w"), indent=1)
    print(f"{'hráč':40s} {'Elo':>6s}  {'95% CI':>13s}  partie  skóre  GPU-h")
    for r in table["players"]:
        print(f"{r['player']:40s} {r['elo']:6d}  [{r['ci95'][0]:5d},{r['ci95'][1]:5d}]  {r['games']:6d}  "
              f"{r['score']:.3f}  {r.get('gpu_hours', '')}")
    print(f"→ {os.path.join(args.out, 'elo.json')}")


if __name__ == "__main__":
    main()
