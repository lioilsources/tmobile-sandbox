"""UC12: kostra AlphaZero smyčky pro OrbitronTactics (JAX + mctx + flax), podle pgx AlphaZero příkladu.

- Prostředí: uc12/orbitron_py/rules.py (fuzz-ekvivalentní s Dart enginem, viz uc12/fuzz.py).
- Hledání: `mctx.gumbel_muzero_policy` s pravidly jako modelem (perfektní simulátor).
  Bitva při braní je náhodný uzel: v `recurrent_fn` se výsledek losuje podle BattleOdds (vzorkovaný
  chance node). Přesná varianta s očekávanou hodnotou přes oba výsledky = `mctx.stochastic_muzero_policy`
  (afterstate = pozice + tah, chance_logits = log [p, 1−p]) — TODO pro plný trénink.
- Síť: ResNet (6 bloků × 128 kanálů) na kanonickém vstupu 8×8×58 (strana na tahu vždy „dole“:
  pro černého otočeno o 180° a prohozené barvy), policy hlava 8×8×40 = 2560 akcí (pole × posun),
  value hlava tanh.
- Self-play: B paralelních partií (vmap), auto-reset po konci partie, cíle hodnoty z výsledku partie.

Smoke test (CPU, pár minut):
  JAX_PLATFORMS=cpu python uc12/az.py --batch 32 --sims 16 --steps 64 --iters 2
"""
import argparse
import functools
import json
import os
import sys
import time
from typing import NamedTuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import flax.linen as nn  # noqa: E402
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


# --- Prostředí ------------------------------------------------------------------------------

def init_state(key):
    """Nová partie: náhodná varianta silových polí, každá strana z půlky výchozí, z půlky náhodná formace."""
    kv, kw, kb, kw2, kb2 = jax.random.split(key, 5)
    white = jnp.where(jax.random.bernoulli(kw2), DEFAULT_W, jax.random.permutation(kw, BASE_ARMY))
    black = jnp.where(jax.random.bernoulli(kb2), DEFAULT_B, jax.random.permutation(kb, BASE_ARMY))
    cells = jnp.zeros(64, jnp.int8).at[:16].set(white + 1).at[48:].set(black + 7)
    variant = jax.random.randint(kv, (), 0, len(R.PF_VARIANTS))
    return R.new_state(cells, variant)


def observe(s: R.State):
    """Kanonický vstup [8, 8, 58]: pro každou ze 4 desek (aktuální + historie) 6 typů vlastních, 6 cizích,
    vlastní a cizí Last Warrior; pak silová pole a postup k limitu půltahů."""
    black = s.turn == R.BLACK
    boards = jnp.concatenate([s.cells[None], s.hist], axis=0).astype(jnp.int32)  # [4, 64]
    boards = jnp.where(black, boards[:, ::-1], boards)                            # otočení o 180°
    occ = boards > 0
    t = (boards - 1) % 6
    color = ((boards - 1) // 6) % 2
    own = occ & (color == s.turn)
    opp = occ & (color != s.turn)
    lw = boards > 12
    types = jnp.arange(6)[None, None, :]
    planes = jnp.concatenate([
        (own[..., None] & (t[..., None] == types)),
        (opp[..., None] & (t[..., None] == types)),
        (own & lw)[..., None], (opp & lw)[..., None]], axis=-1)                   # [4, 64, 14]
    planes = planes.transpose(1, 0, 2).reshape(64, 14 * R.HIST)
    pf = jnp.where(black, s.pf[::-1], s.pf)
    extra = jnp.stack([pf, jnp.full(64, s.plies / R.MAX_PLIES)], axis=-1)
    return jnp.concatenate([planes.astype(jnp.float32), extra.astype(jnp.float32)], -1).reshape(8, 8, N_PLANES)


def to_real(s, canon):
    """Logity / váhy z kanonického rámce (černý otočený) do skutečných akcí; zobrazení je involuce."""
    return jnp.where(s.turn == R.BLACK, canon[..., R.MIRROR_ACTION], canon)


# --- Síť ------------------------------------------------------------------------------------

class Block(nn.Module):
    ch: int

    @nn.compact
    def __call__(self, x, train):
        y = nn.Conv(self.ch, (3, 3), use_bias=False)(x)
        y = nn.relu(nn.BatchNorm(use_running_average=not train)(y))
        y = nn.Conv(self.ch, (3, 3), use_bias=False)(y)
        y = nn.BatchNorm(use_running_average=not train)(y)
        return nn.relu(x + y)


class AZNet(nn.Module):
    blocks: int = 6
    ch: int = 128

    @nn.compact
    def __call__(self, x, train=False):
        x = nn.relu(nn.BatchNorm(use_running_average=not train)(nn.Conv(self.ch, (3, 3), use_bias=False)(x)))
        for _ in range(self.blocks):
            x = Block(self.ch)(x, train)
        p = nn.relu(nn.BatchNorm(use_running_average=not train)(nn.Conv(64, (1, 1), use_bias=False)(x)))
        logits = nn.Conv(R.N_DISP, (1, 1))(p).reshape(x.shape[0], R.N_ACTIONS)    # pole × posun
        v = nn.relu(nn.BatchNorm(use_running_average=not train)(nn.Conv(4, (1, 1), use_bias=False)(x)))
        v = nn.relu(nn.Dense(128)(v.reshape(x.shape[0], -1)))
        return logits, jnp.tanh(nn.Dense(1)(v))[:, 0]


NET = AZNet()


def evaluate(variables, states):
    obs = jax.vmap(observe)(states)
    canon, value = NET.apply(variables, obs, train=False)
    logits = jax.vmap(to_real)(states, canon)
    mask = jax.vmap(R.legal_mask)(states)
    return jnp.where(mask, logits, -1e9), value, mask


# --- MCTS -----------------------------------------------------------------------------------

def recurrent_fn(variables, rng, action, states):
    keys = jax.random.split(rng, action.shape[0])
    mover = states.turn
    nxt = jax.vmap(R.step_random, in_axes=(0, 0, 0, None))(states, action, keys, ODDS)
    logits, value, mask = evaluate(variables, nxt)
    done = jax.vmap(R.terminal)(nxt, mask)
    reward = jnp.where(nxt.finished, jnp.where(nxt.winner == mover + 1, 1.0, -1.0), 0.0)
    out = mctx.RecurrentFnOutput(
        reward=reward,
        discount=jnp.where(done, 0.0, -1.0),   # hráči se střídají (vítězný tah tah nepředá, ale je konec)
        prior_logits=logits,
        value=jnp.where(done, 0.0, value))
    return out, nxt


class Sample(NamedTuple):
    obs: jnp.ndarray
    policy: jnp.ndarray     # kanonické cílové váhy z MCTS
    player: jnp.ndarray
    done: jnp.ndarray
    winner: jnp.ndarray
    mask: jnp.ndarray       # platný krok (ne po konci před resetem)


@functools.partial(jax.jit, static_argnames=("sims",))
def selfplay_step(variables, states, key, sims):
    k1, k2, k3 = jax.random.split(key, 3)
    logits, value, mask = evaluate(variables, states)
    root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=states)
    out = mctx.gumbel_muzero_policy(variables, k1, root, recurrent_fn, num_simulations=sims,
                                    invalid_actions=~mask, max_num_considered_actions=16)
    obs = jax.vmap(observe)(states)
    canon_w = jax.vmap(to_real)(states, out.action_weights)  # involuce: real → kanonický
    keys = jax.random.split(k2, states.turn.shape[0])
    nxt = jax.vmap(R.step_random, in_axes=(0, 0, 0, None))(states, out.action, keys, ODDS)
    done = jax.vmap(R.terminal)(nxt)
    sample = Sample(obs, canon_w, states.turn, done, nxt.winner, jnp.ones_like(done))
    fresh = jax.vmap(init_state)(jax.random.split(k3, states.turn.shape[0]))
    nxt = jax.tree.map(lambda f, n: jnp.where(done.reshape((-1,) + (1,) * (n.ndim - 1)), f, n), fresh, nxt)
    return nxt, sample


def value_targets(samples: Sample):
    """Výsledek partie z pohledu hráče na tahu; partie useknuté koncem dávky se nepočítají (mask 0)."""
    T = samples.done.shape[0]

    def body(carry, t):
        winner, seen = carry
        i = T - 1 - t
        winner = jnp.where(samples.done[i], samples.winner[i], winner)
        seen = seen | samples.done[i]
        z = jnp.where(winner == 0, 0.0, jnp.where(winner == samples.player[i] + 1, 1.0, -1.0))
        return (winner, seen), (z, seen)

    init = (jnp.zeros_like(samples.winner[0]), jnp.zeros_like(samples.done[0]))
    _, (z, seen) = jax.lax.scan(body, init, jnp.arange(T))
    return z[::-1], seen[::-1]


def loss_fn(params, batch_stats, obs, policy, z, w):
    (logits, v), upd = NET.apply({"params": params, "batch_stats": batch_stats}, obs, train=True,
                                 mutable=["batch_stats"])
    pl = optax.softmax_cross_entropy(logits, policy)
    vl = (v - z) ** 2
    # policy cíl platí vždy, hodnota jen u partií, které v dávce skončily
    vl = jnp.sum(vl * w) / jnp.maximum(w.sum(), 1)
    loss = pl.mean() + vl
    return loss, (upd["batch_stats"], pl.mean(), vl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32, help="paralelní partie")
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--steps", type=int, default=64, help="půltahů na iteraci")
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--out")
    args = ap.parse_args()

    global NET
    NET = AZNet(args.blocks, args.ch)
    key = jax.random.PRNGKey(0)
    key, k = jax.random.split(key)
    variables = NET.init(k, jnp.zeros((1, 8, 8, N_PLANES)))
    params, batch_stats = variables["params"], variables["batch_stats"]
    opt = optax.adamw(2e-4, weight_decay=1e-4)
    opt_state = opt.init(params)
    nparams = sum(x.size for x in jax.tree.leaves(params))
    print(f"síť: {args.blocks} bloků × {args.ch} kanálů, {nparams / 1e6:.2f} M parametrů, backend "
          f"{jax.default_backend()}", flush=True)

    @jax.jit
    def train_step(params, batch_stats, opt_state, obs, policy, z, w):
        (loss, (bs, pl, vl)), g = jax.value_and_grad(loss_fn, has_aux=True)(params, batch_stats, obs, policy, z, w)
        upd, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, upd), bs, opt_state, loss, pl, vl

    key, k = jax.random.split(key)
    states = jax.vmap(init_state)(jax.random.split(k, args.batch))
    log = []
    for it in range(args.iters):
        t0 = time.time()
        samples = []
        variables = {"params": params, "batch_stats": batch_stats}
        for _ in range(args.steps):
            key, k = jax.random.split(key)
            states, s = selfplay_step(variables, states, k, args.sims)
            samples.append(s)
        samples = jax.tree.map(lambda *x: jnp.stack(x), *samples)
        jax.block_until_ready(samples)
        t_play = time.time() - t0
        z, w = value_targets(samples)
        flat = lambda x: x.reshape((-1,) + x.shape[2:])  # noqa: E731
        obs, pol, zz, ww = flat(samples.obs), flat(samples.policy), flat(z), flat(w).astype(jnp.float32)
        t1 = time.time()
        params, batch_stats, opt_state, loss, pl, vl = train_step(params, batch_stats, opt_state, obs, pol, zz, ww)
        jax.block_until_ready(params)
        games = int(samples.done.sum())
        pos = args.batch * args.steps
        rec = {"iter": it, "positions": pos, "games_finished": games, "selfplay_s": round(t_play, 2),
               "positions_per_s": round(pos / t_play, 1), "train_s": round(time.time() - t1, 2),
               "loss": float(loss), "policy_ce": float(pl), "value_mse": float(vl)}
        log.append(rec)
        print(json.dumps(rec), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "params_M": nparams / 1e6, "backend": jax.default_backend(),
                       "log": log}, f, indent=1)


if __name__ == "__main__":
    main()
