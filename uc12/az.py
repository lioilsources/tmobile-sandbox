"""UC12: AlphaZero (Gumbel MuZero s perfektním simulátorem) pro OrbitronTactics — JAX + mctx + flax.

- Prostředí: uc12/orbitron_py/rules.py (fuzz-ekvivalentní s Dart enginem, 0 rozdílů / 120k partií).
- Hledání: `mctx.gumbel_muzero_policy`, model = pravidla. Bitva při braní je náhodný uzel podle
  BattleOdds. Režim `--chance expect` (výchozí): v `recurrent_fn` se vyhodnotí OBA výsledky bitvy
  (výhra / prohra útočníka) a hodnota hrany je jejich očekávaná hodnota; do hlubšího hledání jde
  jeden vylosovaný výsledek a jeho zpřesnění se přičte jako korekce (reward = E[q] + v_vylosovaný),
  takže Q = E[q] při expanzi a hlubší backup opravuje jen vylosovanou větev. `--chance sample` =
  jen losování (poloviční cena sítě v hledání, větší rozptyl).
  (`mctx.stochastic_muzero_policy` má přesné chance uzly, ale jen s PUCT, ne Gumbel → nepoužito.)
- Síť: ResNet (6 bloků × 128 kanálů) na kanonickém vstupu 8×8×58 (strana na tahu vždy „dole“),
  policy hlava 8×8×40 = 2560 akcí (pole × posun), value hlava tanh. Na GPU výpočet v bf16.
- Self-play: B paralelních partií (vmap) na GPU, `--steps` půltahů na jedno volání (lax.scan),
  auto-reset po konci partie. Cíl hodnoty = výsledek partie; vzorky čekají v hostitelském okně,
  dokud jejich partie neskončí (žádné useknuté partie).
- Replay: kruhový buffer v RAM, kompaktní záznam (stav 324 B + top-K politika) ≈ 452 B/pozice
  při K=32 → 4 M pozic ≈ 1,8 GB.
- Checkpointy každých `--ckpt-min` minut do `--ckpt-dir`: `state.msgpack` (síť + optimizér +
  čítače, přepisuje se atomicky), `replay.npz` (volitelně), `nets/net_<iter>.msgpack` + `.json`
  (jen síť, pro Elo ligu). Start s existujícím `state.msgpack` = resume (vypnout `--fresh`).
- Metriky: `--metrics` jsonl, jeden řádek na iteraci (partie/min, ztráty, entropie, GPU util z CSV
  `bench/gpu_log.sh` přes `--gpu-log`, RSS).
- `--mem-fraction` (výchozí 0.9 = GPU sama) se nastaví do XLA_PYTHON_CLIENT_MEM_FRACTION před
  importem JAX; proměnná prostředí má přednost, pokud flag není zadán.

Smoke (CPU):
  JAX_PLATFORMS=cpu python uc12/az.py --batch 16 --sims 8 --steps 8 --blocks 2 --ch 32 \\
      --iters 6 --min-replay 256 --train-batch 64 --ckpt-dir /tmp/az --ckpt-min 0
"""
import os
import sys

if __name__ == "__main__":  # před importem JAX
    _a = sys.argv
    if "--mem-fraction" in _a:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = _a[_a.index("--mem-fraction") + 1]
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

import argparse  # noqa: E402
import functools  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import resource  # noqa: E402
import signal  # noqa: E402
import time  # noqa: E402
from typing import NamedTuple  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import flax.linen as nn  # noqa: E402
import flax.serialization as fser  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mctx  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from orbitron_py import odds as O  # noqa: E402
from orbitron_py import rules as R  # noqa: E402

ODDS = jnp.asarray(O.odds_pair(), jnp.float32)  # flotily bez upgradů
N_PLANES = 14 * R.HIST + 2                      # 58
BASE_ARMY = jnp.asarray([R.PAWN] * 8 + [R.ROOK] * 2 + [R.KNIGHT] * 2 + [R.BISHOP] * 2 + [R.QUEEN, R.KING],
                        jnp.int8)
DEFAULT_W = jnp.asarray(R.start_cells()[:16] - 1, jnp.int8)
DEFAULT_B = jnp.asarray(R.start_cells()[48:] - 7, jnp.int8)
NEG = -1e9


# --- Prostředí ------------------------------------------------------------------------------

def init_state(key):
    """Nová partie: náhodná varianta silových polí, každá strana z půlky výchozí, z půlky náhodná formace
    (každá permutace 16 figur je platná formace, viz FormationValidator)."""
    kv, kw, kb, kw2, kb2 = jax.random.split(key, 5)
    white = jnp.where(jax.random.bernoulli(kw2), DEFAULT_W, jax.random.permutation(kw, BASE_ARMY))
    black = jnp.where(jax.random.bernoulli(kb2), DEFAULT_B, jax.random.permutation(kb, BASE_ARMY))
    cells = jnp.zeros(64, jnp.int8).at[:16].set(white + 1).at[48:].set(black + 7)
    variant = jax.random.randint(kv, (), 0, len(R.PF_VARIANTS))
    return R.new_state(cells, variant)


def observe_fields(cells, hist, turn, pf, plies):
    """Kanonický vstup [8, 8, 58]: pro každou ze 4 desek (aktuální + historie) 6 typů vlastních, 6 cizích,
    vlastní a cizí Last Warrior; pak silová pole a postup k limitu půltahů."""
    black = turn == R.BLACK
    boards = jnp.concatenate([cells[None], hist], axis=0).astype(jnp.int32)  # [4, 64]
    boards = jnp.where(black, boards[:, ::-1], boards)                        # otočení o 180°
    occ = boards > 0
    t = (boards - 1) % 6
    color = ((boards - 1) // 6) % 2
    own = occ & (color == turn)
    opp = occ & (color != turn)
    lw = boards > 12
    types = jnp.arange(6)[None, None, :]
    planes = jnp.concatenate([
        (own[..., None] & (t[..., None] == types)),
        (opp[..., None] & (t[..., None] == types)),
        (own & lw)[..., None], (opp & lw)[..., None]], axis=-1)               # [4, 64, 14]
    planes = planes.transpose(1, 0, 2).reshape(64, 14 * R.HIST)
    pf = jnp.where(black, pf[::-1], pf)
    extra = jnp.stack([pf.astype(jnp.float32), jnp.full(64, plies / R.MAX_PLIES, jnp.float32)], axis=-1)
    return jnp.concatenate([planes.astype(jnp.float32), extra], -1).reshape(8, 8, N_PLANES)


def observe(s: R.State):
    return observe_fields(s.cells, s.hist, s.turn, s.pf, s.plies)


def to_real(s_turn, canon):
    """Logity / váhy z kanonického rámce (černý otočený) do skutečných akcí; zobrazení je involuce."""
    return jnp.where(s_turn == R.BLACK, canon[..., R.MIRROR_ACTION], canon)


# --- Síť ------------------------------------------------------------------------------------

class Block(nn.Module):
    ch: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, train):
        bn = functools.partial(nn.BatchNorm, use_running_average=not train, dtype=self.dtype)
        y = nn.Conv(self.ch, (3, 3), use_bias=False, dtype=self.dtype)(x)
        y = nn.relu(bn()(y))
        y = nn.Conv(self.ch, (3, 3), use_bias=False, dtype=self.dtype)(y)
        y = bn()(y)
        return nn.relu(x + y)


class AZNet(nn.Module):
    blocks: int = 6
    ch: int = 128
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, train=False):
        bn = functools.partial(nn.BatchNorm, use_running_average=not train, dtype=self.dtype)
        x = x.astype(self.dtype)
        x = nn.relu(bn()(nn.Conv(self.ch, (3, 3), use_bias=False, dtype=self.dtype)(x)))
        for _ in range(self.blocks):
            x = Block(self.ch, self.dtype)(x, train)
        p = nn.relu(bn()(nn.Conv(64, (1, 1), use_bias=False, dtype=self.dtype)(x)))
        logits = nn.Conv(R.N_DISP, (1, 1), dtype=self.dtype)(p).reshape(x.shape[0], R.N_ACTIONS)
        v = nn.relu(bn()(nn.Conv(4, (1, 1), use_bias=False, dtype=self.dtype)(x)))
        v = nn.relu(nn.Dense(128, dtype=self.dtype)(v.reshape(x.shape[0], -1)))
        v = nn.Dense(1, dtype=self.dtype)(v)
        return logits.astype(jnp.float32), jnp.tanh(v.astype(jnp.float32))[:, 0]


class Search(NamedTuple):
    """Vše, co potřebuje hledání: síť (statická) a jak zacházet s bitvou."""
    net: AZNet
    chance: str = "expect"


def evaluate(net, variables, states):
    obs = jax.vmap(observe)(states)
    canon, value = net.apply(variables, obs, train=False)
    logits = jax.vmap(to_real)(states.turn, canon)
    mask = jax.vmap(R.legal_mask)(states)
    return jnp.where(mask, logits, NEG), value, mask


# --- MCTS -----------------------------------------------------------------------------------

def _pick(c, a, b):
    return jax.tree.map(lambda x, y: jnp.where(c.reshape((-1,) + (1,) * (x.ndim - 1)), x, y), a, b)


def make_recurrent_fn(search: Search):
    net = search.net

    def recurrent_fn(variables, rng, action, states):
        B = action.shape[0]
        mover = states.turn
        p = jax.vmap(R.capture_odds, in_axes=(0, 0, None))(states, action, ODDS)
        win = jax.random.uniform(rng, (B,)) < p
        if search.chance == "sample":
            nxt = jax.vmap(R.step)(states, action, win)
            logits, value, mask = evaluate(net, variables, nxt)
            done = jax.vmap(R.terminal)(nxt, mask)
            reward = jnp.where(nxt.finished, jnp.where(nxt.winner == mover + 1, 1.0, -1.0), 0.0)
            value = jnp.where(done, 0.0, value)
        else:  # expect: oba výsledky bitvy v jedné dávce 2B
            nw = jax.vmap(R.step)(states, action, jnp.ones(B, bool))
            nl = jax.vmap(R.step)(states, action, jnp.zeros(B, bool))
            both = jax.tree.map(lambda a, b: jnp.concatenate([a, b]), nw, nl)
            logits2, value2, mask2 = evaluate(net, variables, both)
            done2 = jax.vmap(R.terminal)(both, mask2)
            mover2 = jnp.concatenate([mover, mover])
            r2 = jnp.where(both.finished, jnp.where(both.winner == mover2 + 1, 1.0, -1.0), 0.0)
            v2 = jnp.where(done2, 0.0, value2)
            q2 = r2 - v2                                   # z pohledu táhnoucího (discount −1)
            q_exp = p * q2[:B] + (1 - p) * q2[B:]
            nxt = _pick(win, nw, nl)
            logits = jnp.where(win[:, None], logits2[:B], logits2[B:])
            done = jnp.where(win, done2[:B], done2[B:])
            value = jnp.where(win, v2[:B], v2[B:])
            reward = q_exp + value                         # r + d·v = E[q] při d = −1; při d = 0 je v = 0
        out = mctx.RecurrentFnOutput(
            reward=reward,
            discount=jnp.where(done, 0.0, -1.0),           # hráči se střídají
            prior_logits=logits,
            value=value)
        return out, nxt

    return recurrent_fn


def run_search(search: Search, variables, states, key, sims, gumbel_scale=1.0, considered=16):
    logits, value, mask = evaluate(search.net, variables, states)
    root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=states)
    out = mctx.gumbel_muzero_policy(variables, key, root, make_recurrent_fn(search), num_simulations=sims,
                                    invalid_actions=~mask, max_num_considered_actions=considered,
                                    gumbel_scale=gumbel_scale)
    return out, logits, value, mask


class Sample(NamedTuple):
    """Kompaktní záznam pozice (hostitel / replay)."""
    cells: jnp.ndarray   # int8[64]
    hist: jnp.ndarray    # int8[3, 64]
    pf: jnp.ndarray      # bool[64]
    turn: jnp.ndarray    # int8
    plies: jnp.ndarray   # int16
    pidx: jnp.ndarray    # int16[K] kanonické akce (top-K cílové politiky)
    pw: jnp.ndarray      # float16[K]
    done: jnp.ndarray    # bool — partie skončila tímto tahem
    winner: jnp.ndarray  # int8 — vítěz po tahu (0 nikdo)
    victory: jnp.ndarray  # int8
    plies_after: jnp.ndarray  # int16
    prior_ent: jnp.ndarray  # float32 — entropie priora sítě (legální tahy)
    target_ent: jnp.ndarray  # float32 — entropie cíle z MCTS
    root_v: jnp.ndarray  # float32


def _bin_entropy(v):
    """Průměrná entropie hodnoty jako Bernoulli p = (v + 1) / 2, v nats (ln 2 = nejistá, 0 = jistá)."""
    p = np.clip((np.asarray(v, np.float64) + 1) / 2, 1e-6, 1 - 1e-6)
    return float(np.mean(-(p * np.log(p) + (1 - p) * np.log(1 - p))))


def _entropy(logits, mask):
    lp = jax.nn.log_softmax(jnp.where(mask, logits, NEG))
    return -jnp.sum(jnp.where(mask, jnp.exp(lp) * lp, 0.0), -1)


def make_selfplay(search: Search, sims, steps, topk, considered):
    @jax.jit
    def selfplay(variables, states, key):
        def body(states, key):
            k1, k2, k3 = jax.random.split(key, 3)
            out, logits, value, mask = run_search(search, variables, states, k1, sims, 1.0, considered)
            w = out.action_weights
            canon_w = jax.vmap(to_real)(states.turn, w)    # involuce: real → kanonický
            pw, pidx = jax.lax.top_k(canon_w, topk)
            pw = pw / jnp.maximum(pw.sum(-1, keepdims=True), 1e-9)
            t_ent = -jnp.sum(jnp.where(w > 0, w * jnp.log(jnp.maximum(w, 1e-12)), 0.0), -1)
            keys = jax.random.split(k2, states.turn.shape[0])
            nxt = jax.vmap(R.step_random, in_axes=(0, 0, 0, None))(states, out.action, keys, ODDS)
            done = jax.vmap(R.terminal)(nxt)
            smp = Sample(states.cells, states.hist, states.pf, states.turn, states.plies.astype(jnp.int16),
                         pidx.astype(jnp.int16), pw.astype(jnp.float16), done, nxt.winner, nxt.victory,
                         nxt.plies.astype(jnp.int16), _entropy(logits, mask), t_ent, value)
            fresh = jax.vmap(init_state)(jax.random.split(k3, states.turn.shape[0]))
            nxt = _pick(done, fresh, nxt)
            return nxt, smp

        return jax.lax.scan(body, states, jax.random.split(key, steps))

    return selfplay


# --- Hostitel: dokončení partií a replay ---------------------------------------------------------

STORE = ("cells", "hist", "pf", "turn", "plies", "pidx", "pw")


class Assembler:
    """Drží vzorky rozehraných partií, dokud neskončí; pak jim přiřadí z a pošle je do replaye."""

    def __init__(self):
        self.win = None      # dict polí [T, B, ...]
        self.emitted = None  # bool[T, B]

    def add(self, chunk: dict):
        if self.win is None:
            win = chunk
            emitted = np.zeros(chunk["done"].shape, bool)
        else:
            win = {k: np.concatenate([self.win[k], chunk[k]]) for k in chunk}
            emitted = np.concatenate([self.emitted, np.zeros(chunk["done"].shape, bool)])
        T, B = win["done"].shape
        z = np.zeros((T, B), np.int8)
        seen = np.zeros((T, B), bool)
        wc = np.zeros(B, np.int8)
        sc = np.zeros(B, bool)
        for i in range(T - 1, -1, -1):
            d = win["done"][i]
            wc = np.where(d, win["winner"][i], wc)
            sc = sc | d
            z[i] = np.where(wc == 0, 0, np.where(wc == win["turn"][i] + 1, 1, -1))
            seen[i] = sc
        out = seen & ~emitted
        ready = {k: win[k][out] for k in STORE}
        ready["z"] = z[out]
        unseen_rows = (~seen).any(1)
        if unseen_rows.any():
            t0 = int(np.argmax(unseen_rows))
            self.win = {k: v[t0:] for k, v in win.items()}
            self.emitted = (seen | emitted)[t0:]
        else:
            self.win = self.emitted = None
        return ready


class Replay:
    def __init__(self, cap, topk):
        self.cap, self.n, self.pos = cap, 0, 0
        self.a = {"cells": np.zeros((cap, 64), np.int8), "hist": np.zeros((cap, R.HIST - 1, 64), np.int8),
                  "pf": np.zeros((cap, 64), bool), "turn": np.zeros(cap, np.int8),
                  "plies": np.zeros(cap, np.int16), "pidx": np.zeros((cap, topk), np.int16),
                  "pw": np.zeros((cap, topk), np.float16), "z": np.zeros(cap, np.int8)}

    def nbytes(self):
        return sum(v.nbytes for v in self.a.values())

    def add(self, d):
        m = len(d["z"])
        if m == 0:
            return
        if m > self.cap:
            d = {k: v[-self.cap:] for k, v in d.items()}
            m = self.cap
        idx = (self.pos + np.arange(m)) % self.cap
        for k, v in self.a.items():
            v[idx] = d[k]
        self.pos = (self.pos + m) % self.cap
        self.n = min(self.n + m, self.cap)

    def sample(self, rng, shape):
        idx = rng.integers(0, self.n, size=shape)
        return {k: v[idx] for k, v in self.a.items()}

    def save(self, path):
        tmp = path + ".tmp.npz"
        np.savez(tmp, n=self.n, pos=self.pos, **{k: v[:self.n] for k, v in self.a.items()})
        os.replace(tmp, path)

    def load(self, path):
        with np.load(path) as f:
            n = int(f["n"])
            keep = min(n, self.cap)
            for k, v in self.a.items():
                if f[k].shape[1:] != v.shape[1:]:
                    print(f"replay: {k} má jiný tvar, nenačteno", flush=True)
                    return
            for k, v in self.a.items():
                v[:keep] = f[k][:keep]
            self.n, self.pos = keep, (int(f["pos"]) if n <= self.cap else 0) % self.cap


# --- Trénink --------------------------------------------------------------------------------

def make_train(net, opt, steps):
    def loss_fn(params, batch_stats, b):
        st = R.State(cells=b["cells"], turn=b["turn"], finished=jnp.zeros_like(b["turn"], bool),
                     winner=jnp.zeros_like(b["turn"]), victory=jnp.zeros_like(b["turn"]),
                     move_count=jnp.zeros(b["turn"].shape, jnp.int32),
                     royals=jnp.ones(b["turn"].shape + (4,), bool), plies=b["plies"].astype(jnp.int32),
                     pf=b["pf"], hist=b["hist"])
        obs = jax.vmap(observe)(st)
        mask = jax.vmap(R.legal_mask)(st)
        cmask = jax.vmap(to_real)(st.turn, mask)          # kanonický rámec (involuce)
        target = jax.vmap(lambda i, w: jnp.zeros(R.N_ACTIONS).at[i].add(w))(
            b["pidx"].astype(jnp.int32), b["pw"].astype(jnp.float32))
        (logits, v), upd = net.apply({"params": params, "batch_stats": batch_stats}, obs, train=True,
                                     mutable=["batch_stats"])
        logits = jnp.where(cmask, logits, NEG)
        pl = optax.softmax_cross_entropy(logits, target).mean()
        vl = jnp.mean((v - b["z"].astype(jnp.float32)) ** 2)
        return pl + vl, (upd["batch_stats"], pl, vl)

    @jax.jit
    def train(params, batch_stats, opt_state, batches):
        def body(carry, b):
            params, batch_stats, opt_state = carry
            (loss, (bs, pl, vl)), g = jax.value_and_grad(loss_fn, has_aux=True)(params, batch_stats, b)
            upd, opt_state = opt.update(g, opt_state, params)
            gn = optax.global_norm(g)
            return (optax.apply_updates(params, upd), bs, opt_state), jnp.stack([loss, pl, vl, gn])

        (params, batch_stats, opt_state), m = jax.lax.scan(body, (params, batch_stats, opt_state), batches)
        return params, batch_stats, opt_state, m.mean(0)

    return train


# --- Checkpointy ----------------------------------------------------------------------------

def net_config(args):
    return {"blocks": args.blocks, "ch": args.ch}


def build_net(cfg, dtype=jnp.float32):
    return AZNet(cfg["blocks"], cfg["ch"], dtype)


def init_variables(net, seed=0):
    return net.init(jax.random.PRNGKey(seed), jnp.zeros((1, 8, 8, N_PLANES)))


def load_net(path, dtype=jnp.float32):
    """Síť z `nets/net_*.msgpack` (+ `.json` s konfigurací) pro ligu nebo export."""
    meta = json.load(open(path[:-len(".msgpack")] + ".json"))
    net = build_net(meta["net"], dtype)
    tmpl = init_variables(build_net(meta["net"]))
    variables = fser.from_bytes(tmpl, open(path, "rb").read())
    return net, variables, meta


def _atomic_write(path, data: bytes):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def rss_mb():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r / (1024 * 1024 if sys.platform == "darwin" else 1024))


class GpuLog:
    """Čte přírůstky CSV z `nvidia-smi dmon -s pucvmt -o T` (bench/gpu_log.sh)."""

    def __init__(self, path):
        self.path, self.off, self.cols = path, 0, None

    def read(self):
        if not self.path or not os.path.exists(self.path):
            return {}
        sm, fb, pw = [], [], []
        with open(self.path) as f:
            if self.cols is None:
                first = f.readline()
                self.cols = first[1:].split() if first.startswith("#") else None
            f.seek(max(self.off, f.tell()))
            for line in f:
                if line.startswith("#") or not self.cols:
                    continue
                v = line.split()
                if len(v) != len(self.cols):
                    continue
                r = dict(zip(self.cols, v))
                try:
                    sm.append(float(r["sm"]))
                    fb.append(float(r["fb"]))
                    pw.append(float(r["pwr"]))
                except (KeyError, ValueError):
                    pass
            self.off = f.tell()
        if not sm:
            return {}
        return {"gpu_util": round(sum(sm) / len(sm), 1), "gpu_mem_mib": max(fb), "gpu_power_w": round(sum(pw) / len(pw))}


# --- Hlavní smyčka ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=0, help="délka běhu této relace (0 = jen --iters)")
    ap.add_argument("--iters", type=int, default=0, help="max iterací v této relaci (0 = bez limitu)")
    ap.add_argument("--batch", type=int, default=512, help="paralelní partie")
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--considered", type=int, default=16, help="max_num_considered_actions (Gumbel)")
    ap.add_argument("--chance", choices=["expect", "sample"], default="expect")
    ap.add_argument("--steps", type=int, default=16, help="půltahů na jedno volání self-play (scan)")
    ap.add_argument("--topk", type=int, default=32, help="kolik akcí cílové politiky ukládat")
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--dtype", choices=["auto", "f32", "bf16"], default="auto", help="auto = bf16 na GPU")
    ap.add_argument("--replay", type=int, default=4_000_000, help="kapacita replaye (pozic)")
    ap.add_argument("--min-replay", type=int, default=100_000, help="trénink až od tolika pozic")
    ap.add_argument("--train-batch", type=int, default=1024)
    ap.add_argument("--train-steps", type=int, default=0,
                    help="kroků SGD na iteraci (0 = replay_ratio × batch × steps / train_batch)")
    ap.add_argument("--replay-ratio", type=float, default=4.0, help="kolikrát se každá pozice v průměru použije")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--ckpt-dir", default="/cold/w2/uc12/ckpt")
    ap.add_argument("--ckpt-min", type=float, default=15, help="checkpoint každých N minut (0 = každou iteraci)")
    ap.add_argument("--no-save-replay", action="store_true", help="neukládat replay.npz (rychlejší checkpoint)")
    ap.add_argument("--fresh", action="store_true", help="ignorovat existující checkpoint")
    ap.add_argument("--metrics", help="jsonl s metrikami (výchozí <ckpt-dir>/metrics.jsonl)")
    ap.add_argument("--gpu-log", help="CSV z bench/gpu_log.sh pro GPU util v metrikách")
    ap.add_argument("--mem-fraction", default=None, help="XLA_PYTHON_CLIENT_MEM_FRACTION (výchozí 0.9)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.ckpt_dir, "nets"), exist_ok=True)
    metrics_path = args.metrics or os.path.join(args.ckpt_dir, "metrics.jsonl")
    backend = jax.default_backend()
    dtype = {"f32": jnp.float32, "bf16": jnp.bfloat16}.get(args.dtype,
                                                          jnp.bfloat16 if backend == "gpu" else jnp.float32)
    cfg = net_config(args)
    net = build_net(cfg, dtype)
    search = Search(net, args.chance)
    train_steps = args.train_steps or max(1, round(args.replay_ratio * args.batch * args.steps / args.train_batch))
    sched = optax.join_schedules([optax.linear_schedule(0.0, args.lr, args.warmup), optax.constant_schedule(args.lr)],
                                 [args.warmup])
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=args.wd))

    variables = init_variables(build_net(cfg), args.seed)
    params, batch_stats = variables["params"], variables["batch_stats"]
    opt_state = opt.init(params)
    ctr = {"iter": 0, "train_steps": 0, "games": 0, "positions": 0, "elapsed_s": 0.0, "sessions": 0}
    replay = Replay(args.replay, args.topk)
    state_path = os.path.join(args.ckpt_dir, "state.msgpack")
    replay_path = os.path.join(args.ckpt_dir, "replay.npz")
    meta_path = os.path.join(args.ckpt_dir, "state.json")
    if not args.fresh and os.path.exists(state_path):
        meta = json.load(open(meta_path))
        if meta["net"] != cfg:
            sys.exit(f"checkpoint má jinou síť {meta['net']} než {cfg}; použij --fresh nebo jiný --ckpt-dir")
        tmpl = {"params": params, "batch_stats": batch_stats, "opt_state": opt_state}
        restored = fser.from_bytes(tmpl, open(state_path, "rb").read())
        params, batch_stats, opt_state = restored["params"], restored["batch_stats"], restored["opt_state"]
        ctr.update(meta["ctr"])
        if os.path.exists(replay_path):
            t = time.time()
            replay.load(replay_path)
            print(f"replay načten: {replay.n} pozic za {time.time() - t:.1f} s", flush=True)
        print(f"RESUME z iterace {ctr['iter']} ({ctr['elapsed_s'] / 3600:.2f} h, {ctr['games']} partií)", flush=True)
    ctr["sessions"] += 1
    nparams = sum(x.size for x in jax.tree.leaves(params))
    print(json.dumps({"start": time.strftime("%Y-%m-%dT%H:%M:%S"), "backend": backend, "devices": str(jax.devices()),
                      "compute_dtype": str(np.dtype(dtype)), "params_M": round(nparams / 1e6, 3), "train_steps_per_iter": train_steps,
                      "replay_GB": round(replay.nbytes() / 1e9, 2), "mem_fraction": os.environ.get(
                          "XLA_PYTHON_CLIENT_MEM_FRACTION"), **vars(args)}), flush=True)

    selfplay = make_selfplay(search, args.sims, args.steps, args.topk, min(args.considered, R.N_ACTIONS))
    train = make_train(build_net(cfg, dtype), opt, train_steps)
    key = jax.random.PRNGKey(args.seed * 1000003 + ctr["iter"])
    rng = np.random.default_rng(args.seed * 7919 + ctr["iter"])
    key, k = jax.random.split(key)
    states = jax.vmap(init_state)(jax.random.split(k, args.batch))
    asm = Assembler()
    gpu = GpuLog(args.gpu_log)

    stop = {"flag": False}

    def on_signal(sig, _):
        print(f"signál {sig}: dokončím iteraci, uložím checkpoint a skončím", flush=True)
        stop["flag"] = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    def checkpoint(final=False):
        t = time.time()
        blob = fser.to_bytes({"params": params, "batch_stats": batch_stats, "opt_state": opt_state})
        _atomic_write(state_path, blob)
        if not args.no_save_replay:
            replay.save(replay_path)
        meta = {"net": cfg, "ctr": ctr, "args": vars(args), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        _atomic_write(meta_path, json.dumps(meta, indent=1).encode())
        name = os.path.join(args.ckpt_dir, "nets", f"net_{ctr['iter']:06d}")
        _atomic_write(name + ".msgpack", fser.to_bytes({"params": params, "batch_stats": batch_stats}))
        _atomic_write(name + ".json", json.dumps({"net": cfg, "iter": ctr["iter"], "train_steps": ctr["train_steps"],
                                                  "games": ctr["games"], "positions": ctr["positions"],
                                                  "gpu_hours": round(ctr["elapsed_s"] / 3600, 3),
                                                  "backend": backend, "final": final}).encode())
        dt = time.time() - t
        print(f"checkpoint iter {ctr['iter']} → {args.ckpt_dir} ({dt:.1f} s)", flush=True)
        return dt

    t_session = time.time()
    start_iter = ctr["iter"]
    t_last_ckpt = time.time()
    t_prev = time.time()
    first = True
    mf = open(metrics_path, "a")
    while not stop["flag"]:
        t0 = time.time()
        key, k = jax.random.split(key)
        variables = {"params": params, "batch_stats": batch_stats}
        states, smp = selfplay(variables, states, k)
        smp = jax.device_get(smp)
        t_play = time.time() - t0
        chunk = {f: np.asarray(getattr(smp, f)) for f in STORE + ("done", "winner")}
        ready = asm.add(chunk)
        replay.add(ready)
        d = smp.done
        games = int(d.sum())
        ctr["games"] += games
        ctr["positions"] += int(d.size)

        t1 = time.time()
        tm = None
        if replay.n >= args.min_replay:
            b = replay.sample(rng, (train_steps, args.train_batch))
            params, batch_stats, opt_state, tm = train(params, batch_stats, opt_state, b)
            tm = np.asarray(tm)
            ctr["train_steps"] += train_steps
        t_train = time.time() - t1
        now = time.time()
        ctr["elapsed_s"] += now - t_prev
        t_prev = now
        ctr["iter"] += 1
        it_s = now - t0
        win = smp.winner[d]
        vic = smp.victory[d]
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "iter": ctr["iter"], "session_min": round((now - t_session) / 60, 2),
               "gpu_hours": round(ctr["elapsed_s"] / 3600, 4), "iter_s": round(it_s, 2), "selfplay_s": round(t_play, 2),
               "train_s": round(t_train, 2), "compile_iter": first,
               "games": games, "games_total": ctr["games"], "games_per_min": round(games / it_s * 60, 1),
               "plies_per_s": round(d.size / it_s, 1),
               "game_len": round(float(smp.plies_after[d].mean()), 1) if games else None,
               "white_win": round(float((win == 1).mean()), 3) if games else None,
               "black_win": round(float((win == 2).mean()), 3) if games else None,
               "draw": round(float((win == 0).mean()), 3) if games else None,
               "vic_pf_royal_infil": [int((vic == i).sum()) for i in (1, 2, 3)],
               "prior_entropy": round(float(smp.prior_ent.mean()), 3), "target_entropy": round(float(smp.target_ent.mean()), 3),
               "root_value_abs": round(float(np.abs(smp.root_v).mean()), 3),
               "value_entropy": round(float(_bin_entropy(smp.root_v)), 3),
               "replay": replay.n, "train_steps": ctr["train_steps"], "rss_mb": rss_mb()}
        if tm is not None:
            rec.update(loss=round(float(tm[0]), 4), policy_ce=round(float(tm[1]), 4), value_mse=round(float(tm[2]), 4),
                       grad_norm=round(float(tm[3]), 3))
        rec.update(gpu.read())
        first = False
        stop_now = stop["flag"] or (args.iters and ctr["iter"] - start_iter >= args.iters) \
            or (args.hours and now - t_session >= args.hours * 3600)
        if stop_now or now - t_last_ckpt >= args.ckpt_min * 60:
            rec["ckpt_s"] = round(checkpoint(final=bool(stop_now)), 1)
            t_last_ckpt = time.time()
        mf.write(json.dumps(rec) + "\n")
        mf.flush()
        print(json.dumps(rec), flush=True)
        if stop_now:
            break
    mf.close()
    print(f"konec relace: {ctr['iter']} iterací celkem, {ctr['games']} partií, {ctr['elapsed_s'] / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
