"""UC14 — vektorizovaná fyzika volejového kopu FotballSimulatoru v JAXu.

Port čistých core skriptů Godot projektu (lioilsources/FotballSimulator, master):
  scripts/core/serve_generator.gd  → godot_serves()   (bitově shodný PCG32 + randf Godotu, numpy)
  scripts/core/ball_physics.gd     → tick()           (semi-implicit Euler 120 Hz, drag, Magnus, odskok, kutálení)
  scripts/core/contact_model.gd    → contact()        (nárt × míč → rychlost + spin, rating timingu)
  scripts/core/game_state.gd       → presim_ideal(), serve_window(), kick_eval()
                                     (pre-simulace ideálu, lokální rámec kopu, průlet rovinou brány)

Konstanty se NEopisují: parse_tuning() čte přímo scripts/util/tuning.gd (jediný zdroj
pravdy). Tuning je pytree jnp skalárů → přepsání konstanty (Optuna, D1 σ=90 ms) nevyvolá
rekompilaci.

Výsledek kopu (kód):
  0 GOAL   = GÓL (průlet rovinou brány uvnitř tyčí a pod břevnem)
  1 WIDE   = MIMO (průlet rovinou mimo bránu)
  2 SHORT  = NEDOLETĚL (míč se zastavil / nedoletěl do limitu ticků)
  3 WHIFF  = nárt minul (|timing_error| ≥ RATING_LATE_EARLY) → v game_state „MIMO“ po 2. dopadu
  4 LATE   = puštění až po 2. dopadu serve (šance pryč)

Přesnost: float32 (default, rychlé na GPU) nebo float64 (`VOLLEY_X64=1` před importem).
Godot počítá Vector3 ve float32, skaláry GDScriptu ve float64.
"""
from __future__ import annotations

import ast
import math
import os
import re
from functools import partial
from pathlib import Path
from typing import NamedTuple

if os.environ.get("VOLLEY_X64") == "1":
    os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

if os.environ.get("VOLLEY_X64") == "1":
    jax.config.update("jax_enable_x64", True)

DTYPE = jnp.float64 if os.environ.get("VOLLEY_X64") == "1" else jnp.float32
TICKS_PER_SEC = 120  # project.godot: physics/common/physics_ticks_per_second
DT = 1.0 / TICKS_PER_SEC
MAX_FLIGHT_TICKS = 120 * 12   # stejný strop jako tests/tools/kick_dump.gd
PRESIM_MAX_T = 8.0            # game_state._presim_ideal_time: while t < 8.0

GOAL, WIDE, SHORT, WHIFF, LATE = 0, 1, 2, 3, 4
OUTCOME_NAMES = ["GOAL", "WIDE", "SHORT", "WHIFF", "LATE"]
PERFECT, GOOD, EARLY, LATE_R, MISS = 0, 1, 2, 3, 4  # ContactModel.Rating

# ── tuning.gd parser ──────────────────────────────────────────

_CONST_RE = re.compile(r"^\s*const\s+([A-Z_][A-Z0-9_]*)\s*:?=\s*(.+?)\s*(#.*)?$")


def _eval_gd(expr: str, env: dict):
    """Bezpečné vyhodnocení konstantního výrazu GDScriptu (čísla, PI, Vector3, + - * /)."""
    node = ast.parse(expr.replace("true", "True").replace("false", "False"), mode="eval").body

    def ev(n):
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            if n.id in ("PI",):
                return math.pi
            if n.id == "TAU":
                return math.tau
            return env[n.id]
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            v = ev(n.operand)
            return tuple(-x for x in v) if isinstance(v, tuple) else -v
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            ops = {ast.Add: lambda x, y: x + y, ast.Sub: lambda x, y: x - y,
                   ast.Mult: lambda x, y: x * y, ast.Div: lambda x, y: x / y}
            return ops[type(n.op)](a, b)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("Vector3", "Vector2"):
            return tuple(float(ev(a)) for a in n.args)
        raise ValueError(f"nepodporovaný výraz v tuning.gd: {expr!r}")

    return ev(node)


def parse_tuning(path: str | Path) -> dict:
    """Přečte všechny `const X := …` z tuning.gd → {jméno: float | tuple | bool}."""
    env: dict = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = _CONST_RE.match(line)
        if not m:
            continue
        name, expr = m.group(1), m.group(2)
        env[name] = _eval_gd(expr, env)
    return env


# fyzikální konstanty, které jdou do JAX pytree (zbytek tuning.gd je UI/swipe)
PHYS_KEYS = [
    "BALL_RADIUS", "BALL_MASS", "BALL_AREA", "RHO", "CD", "S_COEFF", "SPIN_DECAY",
    "RESTITUTION", "GROUND_FRICTION", "BOUNCE_SPIN_TRANSFER", "BOUNCE_SPIN_RETAIN",
    "BOUNCE_MIN_VY", "ROLL_FRICTION", "STOP_SPEED", "TIMING_SIGMA", "TIMING_WINDOW",
    "FOOT_SPEED_MAX", "FOOT_MASS_EFF", "COR_KICK", "SWING_LIFT", "SPIN_TRANSFER",
    "CURVE_SPIN", "LATE_SHIFT", "RATING_PERFECT", "RATING_GOOD", "RATING_LATE_EARLY",
    "IDEAL_KICK_HEIGHT", "GOAL_DISTANCE", "GOAL_HALF_WIDTH", "GOAL_HEIGHT",
]


def tuning_pytree(tun: dict, overrides: dict | None = None, dtype=None) -> dict:
    """Fyzikální podmnožina tuningu jako dict jnp skalárů (+ GRAVITY_Y)."""
    dtype = dtype or DTYPE
    t = {k: tun[k] for k in PHYS_KEYS}
    t["GRAVITY_Y"] = tun["GRAVITY"][1]
    if overrides:
        for k, v in overrides.items():
            if k not in t:
                raise KeyError(k)
            t[k] = v
        if "BALL_RADIUS" in overrides and "BALL_AREA" not in overrides:
            t["BALL_AREA"] = math.pi * t["BALL_RADIUS"] ** 2
    return {k: jnp.asarray(v, dtype) for k, v in t.items()}


def default_tuning_path() -> Path:
    env = os.environ.get("FOTBAL_REPO")
    if env:
        return Path(env) / "scripts/util/tuning.gd"
    here = Path(__file__).resolve()
    for cand in (here.parent / "FotballSimulator", here.parents[2] / "w2/FotballSimulator",
                 Path("/srv/work/w2/uc14/FotballSimulator")):
        p = cand / "scripts/util/tuning.gd"
        if p.exists():
            return p
    raise FileNotFoundError("tuning.gd nenalezen — nastav FOTBAL_REPO")


# ── serve generator: bitová kopie Godot RandomNumberGenerator ─

_PCG_MULT = np.uint64(6364136223846793005)
_PCG_INC = np.uint64((1442695040888963407 << 1 | 1) & 0xFFFFFFFFFFFFFFFF)  # (DEFAULT_INC << 1) | 1


class _GodotRng:
    """Vektorizovaný PCG32 (thirdparty/misc/pcg.cpp) + RandomPCG::randf (core/math/random_pcg.h)."""

    def __init__(self, seeds: np.ndarray):
        with np.errstate(over="ignore"):
            self.state = np.zeros(len(seeds), np.uint64)
            self._next()
            self.state = self.state + np.asarray(seeds, np.uint64)
            self._next()

    def _next(self) -> np.ndarray:
        with np.errstate(over="ignore"):
            old = self.state
            self.state = old * _PCG_MULT + _PCG_INC
            xs = (((old >> np.uint64(18)) ^ old) >> np.uint64(27)) & np.uint64(0xFFFFFFFF)
            rot = old >> np.uint64(59)
            out = (xs >> rot) | (xs << ((np.uint64(32) - rot) & np.uint64(31)))
            return (out & np.uint64(0xFFFFFFFF)).astype(np.uint32)

    def randf(self) -> np.ndarray:
        proto = self._next()
        mant = self._next() | np.uint32(0x80000001)
        # CLZ32 přes bit_length
        clz = 32 - np.floor(np.log2(np.maximum(proto, 1).astype(np.float64))).astype(np.int64) - 1
        f = np.ldexp(mant.astype(np.float32), (-32 - clz).astype(np.int32)).astype(np.float32)
        # proto == 0 → Godot vrací 0 a spotřebuje jen jeden rand(); pravděpodobnost 2^-32, ignorujeme
        return np.where(proto == 0, np.float32(0), f)

    def randf_range(self, lo: float, hi: float, fma: bool) -> np.ndarray:
        """real_t: randf()*(to-from)+from ve float32 (arm64 build kontrahuje na FMA)."""
        r = self.randf()
        lo32, hi32 = np.float32(lo), np.float32(hi)
        span = np.float32(hi32 - lo32)
        if fma:
            return (r.astype(np.float64) * np.float64(span) + np.float64(lo32)).astype(np.float32)
        return (r * span + lo32).astype(np.float32)


def godot_serves(seeds, tun: dict, fma: bool = True) -> dict:
    """První serve z `ServeGenerator.new(seed).next_serve()` pro každý seed.

    Vrací {pos, vel, omega}: float64 pole (B,3) s hodnotami přesně reprezentovatelnými ve
    float32 (Godot Vector3 = float32)."""
    seeds = np.asarray(seeds, np.int64)
    rng = _GodotRng(seeds)
    f32 = np.float32
    sx = rng.randf_range(-2.5, 2.5, fma)
    sy = rng.randf_range(0.6, 1.4, fma)
    sz = -rng.randf_range(6.0, 9.0, fma)
    lx = rng.randf_range(-0.5, 0.5, fma)
    ly = np.full_like(sx, f32(tun["BALL_RADIUS"]))
    lz = -rng.randf_range(1.5, 2.5, fma)
    ft = rng.randf_range(0.9, 1.4, fma).astype(np.float64)
    dx, dy, dz = (lx - sx).astype(f32), (ly - sy).astype(f32), (lz - sz).astype(f32)
    g = float(f32(-tun["GRAVITY"][1]))  # Vector3 GRAVITY je float32
    vx = (dx.astype(np.float64) / ft).astype(f32)
    vy = (dy.astype(np.float64) / ft + 0.5 * g * ft).astype(f32)
    vz = (dz.astype(np.float64) / ft).astype(f32)
    wx = rng.randf_range(-8.0, 8.0, fma)
    wy = rng.randf_range(-5.0, 5.0, fma)
    wz = rng.randf_range(-3.0, 3.0, fma)
    st = lambda *a: np.stack(a, -1).astype(np.float64)
    return {"pos": st(sx, sy, sz), "vel": st(vx, vy, vz), "omega": st(wx, wy, wz)}


# ── stav míče a tick (ball_physics.gd) ───────────────────────

class Ball(NamedTuple):
    pos: jnp.ndarray      # (B,3)
    vel: jnp.ndarray      # (B,3)
    omega: jnp.ndarray    # (B,3)
    flying: jnp.ndarray   # (B,) bool
    rolling: jnp.ndarray  # (B,) bool
    bounces: jnp.ndarray  # (B,) int32


def launch(pos, vel, omega) -> Ball:
    b = pos.shape[0]
    return Ball(pos, vel, omega, jnp.ones(b, bool), jnp.zeros(b, bool), jnp.zeros(b, jnp.int32))


def _cross(a, b):
    return jnp.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                      a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                      a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], -1)


def _norm(a):
    return jnp.sqrt(jnp.sum(a * a, -1))


def tick(b: Ball, T: dict) -> Ball:
    """Jeden _physics_process(1/120) pro celou dávku; let i kutálení přes jnp.where."""
    dt = DT
    R = T["BALL_RADIUS"]
    # — let: gravitace + drag + Magnus
    vlen = _norm(b.vel)
    drag = (-0.5 * T["RHO"] * T["CD"] * T["BALL_AREA"] * vlen)[:, None] * b.vel
    magnus = T["S_COEFF"] * _cross(b.omega, b.vel)
    grav = jnp.stack([jnp.zeros_like(R), T["GRAVITY_Y"], jnp.zeros_like(R)])
    acc = grav + (drag + magnus) / T["BALL_MASS"]
    vf = b.vel + acc * dt
    pf = b.pos + vf * dt
    wf = b.omega * jnp.exp(-T["SPIN_DECAY"] * dt)
    bounce = (pf[:, 1] < R) & (vf[:, 1] < 0.0)
    # — odskok: restituce + tření + přenos spinu (surface = ω × (0,-R,0) = (Rωz, 0, -Rωx))
    vy_b = -T["RESTITUTION"] * vf[:, 1]
    keep = 1.0 - T["GROUND_FRICTION"]
    vx_b = vf[:, 0] * keep - (R * wf[:, 2]) * T["BOUNCE_SPIN_TRANSFER"]
    vz_b = vf[:, 2] * keep - (-R * wf[:, 0]) * T["BOUNCE_SPIN_TRANSFER"]
    roll_start = bounce & (vy_b < T["BOUNCE_MIN_VY"])
    vy_b = jnp.where(roll_start, 0.0, vy_b)
    vel_fly = jnp.where(bounce[:, None], jnp.stack([vx_b, vy_b, vz_b], -1), vf)
    pos_fly = jnp.where(bounce[:, None], pf.at[:, 1].set(R), pf)
    om_fly = jnp.where(bounce[:, None], wf * T["BOUNCE_SPIN_RETAIN"], wf)
    # — kutálení: exponenciální útlum, pod STOP_SPEED míč usne
    fr = jnp.exp(-T["ROLL_FRICTION"] * dt)
    vr = b.vel * fr
    pr = (b.pos + vr * dt).at[:, 1].set(R)
    wr = b.omega * fr
    stop = _norm(vr) < T["STOP_SPEED"]
    vr = jnp.where(stop[:, None], 0.0, vr)
    wr = jnp.where(stop[:, None], 0.0, wr)

    roll = b.rolling[:, None]
    act = b.flying
    pos = jnp.where(roll, pr, pos_fly)
    vel = jnp.where(roll, vr, vel_fly)
    om = jnp.where(roll, wr, om_fly)
    flying = jnp.where(b.rolling, ~stop, True)
    rolling = jnp.where(b.rolling, ~stop, roll_start)
    bounces = b.bounces + (bounce & ~b.rolling).astype(jnp.int32)
    a3 = act[:, None]
    return Ball(jnp.where(a3, pos, b.pos), jnp.where(a3, vel, b.vel), jnp.where(a3, om, b.omega),
                jnp.where(act, flying, b.flying), jnp.where(act, rolling, b.rolling),
                jnp.where(act, bounces, b.bounces))


# ── serve: pre-simulace ideálu a okno stavů kolem něj ────────

@jax.jit
def presim_ideal(pos, vel, omega, T):
    """game_state._presim_ideal_time pro dávku serve → ideální tick (int32, ≥1).

    Ideál = první tick po 1. odskoku, kdy míč klesá skrz IDEAL_KICK_HEIGHT; jinak vrchol
    odskoku; jinak tick ukončení smyčky."""
    n_max = int(round(PRESIM_MAX_T * TICKS_PER_SEC))
    b0 = launch(pos, vel, omega)
    bsz = pos.shape[0]
    z_i = jnp.zeros(bsz, jnp.int32)
    init = (b0, jnp.ones(bsz, bool), z_i, z_i, jnp.zeros(bsz, pos.dtype), z_i, z_i)

    def body(i, c):
        b, active, found, apex_t, apex_h, end_t, ticks = c
        nb = tick(b, T)
        a3 = active
        b = jax.tree_util.tree_map(lambda n, o: jnp.where(a3.reshape((-1,) + (1,) * (n.ndim - 1)), n, o), nb, b)
        ticks = ticks + active.astype(jnp.int32)
        one = active & (b.bounces == 1)
        higher = one & (b.pos[:, 1] > apex_h)
        apex_h = jnp.where(higher, b.pos[:, 1], apex_h)
        apex_t = jnp.where(higher, ticks, apex_t)
        hit = one & (b.vel[:, 1] < 0.0) & (b.pos[:, 1] <= T["IDEAL_KICK_HEIGHT"])
        found = jnp.where(hit & (found == 0), ticks, found)
        active = active & ~hit & b.flying & (b.bounces < 2) & (ticks < n_max)
        return b, active, found, apex_t, apex_h, end_t, ticks

    b, active, found, apex_t, apex_h, end_t, ticks = jax.lax.fori_loop(0, n_max, body, init)
    return jnp.where(found > 0, found, jnp.where(apex_t > 0, apex_t, ticks))


@jax.jit
def _advance_to(pos, vel, omega, target, T):
    """Odsimuluje každý serve přesně target[i] ticků (maskovaně)."""
    b = launch(pos, vel, omega)
    n = jnp.max(target)

    def cond(c):
        return c[1] < n

    def body(c):
        b, i = c
        nb = tick(b, T)
        m = i < target
        b = jax.tree_util.tree_map(lambda x, o: jnp.where(m.reshape((-1,) + (1,) * (x.ndim - 1)), x, o), nb, b)
        return b, i + 1

    b, _ = jax.lax.while_loop(cond, body, (b, jnp.int32(0)))
    return b


def serve_window(pos, vel, omega, ideal_tick, T, w: int = 16) -> Ball:
    """Stavy míče v tickách ideal + d, d ∈ [-w, w] → Ball s tvarem (B, 2w+1, …)."""
    start = jnp.maximum(ideal_tick - w, 0)
    b = _advance_to(pos, vel, omega, start, T)
    states = [b]
    for _ in range(2 * w):
        b = tick(b, T)
        states.append(b)
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, 1), *states)


# ── contact model (contact_model.gd) ─────────────────────────

def rate_timing(err, T):
    e = jnp.abs(err)
    return jnp.where(e < T["RATING_PERFECT"], PERFECT,
           jnp.where(e < T["RATING_GOOD"], GOOD,
           jnp.where(e < T["RATING_LATE_EARLY"], jnp.where(err > 0, LATE_R, EARLY), MISS)))


def contact(ox, oy, curve, power, err, ball_vel, T):
    """ContactModel.compute_kick v lokálním rámci (-z na bránu). Vrací (hit, rating, vel, omega)."""
    rating = rate_timing(err, T)
    hit = rating != MISS
    oy = oy + T["LATE_SHIFT"] * jnp.clip(err / T["TIMING_WINDOW"], -1.0, 1.0)
    ln = jnp.sqrt(ox * ox + oy * oy)
    s = jnp.where(ln > 0.95, 0.95 / jnp.maximum(ln, 1e-12), 1.0)
    ox, oy = ox * s, oy * s
    tq = jnp.exp(-jnp.square(err / T["TIMING_SIGMA"]))
    v_foot = T["FOOT_SPEED_MAX"] * power * tq
    u = jnp.stack([ox, oy, jnp.sqrt(jnp.maximum(0.0, 1.0 - (ox * ox + oy * oy)))], -1)
    u = u / _norm(u)[..., None]
    n = -u
    sw = jnp.stack([jnp.zeros_like(T["SWING_LIFT"]), T["SWING_LIFT"], -jnp.ones_like(T["SWING_LIFT"])])
    sw = sw / jnp.sqrt(jnp.sum(sw * sw))
    sdn = jnp.sum(sw * n, -1)
    fns = v_foot * jnp.maximum(0.0, sdn)
    closing = jnp.maximum(0.0, fns - jnp.sum(ball_vel * n, -1))
    vel = ball_vel + n * (closing * T["COR_KICK"] * T["FOOT_MASS_EFF"] / T["BALL_MASS"])[..., None]
    t_vec = sw - sdn[..., None] * n
    om = T["SPIN_TRANSFER"] * v_foot[..., None] * _cross(u, t_vec)
    om = om + (T["CURVE_SPIN"] * v_foot * curve)[..., None] * jnp.array([0.0, -1.0, 0.0], om.dtype)
    vel = jnp.where(hit[..., None], vel, ball_vel)
    om = jnp.where(hit[..., None], om, 0.0)
    return hit, rating, vel, om


def launch_from_kick(kpos, kvel, ox, oy, curve, power, err, T):
    """Lokální rámec kopu z game_state (fwd vodorovně na střed brány, right = fwd × UP).
    Stav míče v okamžiku puštění → světová rychlost a spin po kontaktu (+ hit, rating)."""
    fx = -kpos[:, 0]
    fz = -T["GOAL_DISTANCE"] - kpos[:, 2]
    fl = jnp.sqrt(fx * fx + fz * fz)
    fx, fz = fx / fl, fz / fl
    rx, rz = -fz, fx   # right = fwd × UP
    bvl = jnp.stack([kvel[:, 0] * rx + kvel[:, 2] * rz, kvel[:, 1],
                     -(kvel[:, 0] * fx + kvel[:, 2] * fz)], -1)
    hit, rating, vl, wl = contact(ox, oy, curve, power, err, bvl, T)
    to_w = lambda a: jnp.stack([rx * a[:, 0] - fx * a[:, 2], a[:, 1], rz * a[:, 0] - fz * a[:, 2]], -1)
    return hit, rating, to_w(vl), to_w(wl)


# ── let po kopu a výsledek ────────────────────────────────────

class KickResult(NamedTuple):
    outcome: jnp.ndarray     # (B,) int32, viz GOAL…LATE
    cross_x: jnp.ndarray
    cross_y: jnp.ndarray
    cross_speed: jnp.ndarray
    ticks: jnp.ndarray       # ticky letu do výsledku
    end_pos: jnp.ndarray     # (B,3) pozice při výsledku (pro skóre nedoletěl)
    rating: jnp.ndarray
    launch_vel: jnp.ndarray
    launch_omega: jnp.ndarray


def _flight_step(c, T, early_short=False):
    b, done, out, cx, cy, cs, nt = c
    prev_z = b.pos[:, 2]
    nb = tick(b, T)
    live = ~done
    nt = nt + live.astype(jnp.int32)
    stopped = live & ~nb.flying
    plane = -T["GOAL_DISTANCE"]
    crossed = live & nb.flying & (prev_z > plane) & (nb.pos[:, 2] <= plane)
    t = (plane - prev_z) / jnp.where(crossed, nb.pos[:, 2] - prev_z, 1.0)
    # lerpf(a, b, t) = a + (b - a) * t, b - a = vel*dt přesně jako v Godotu až na zaokrouhlení
    x_c = (nb.pos[:, 0] - nb.vel[:, 0] * DT) + (nb.pos[:, 0] - (nb.pos[:, 0] - nb.vel[:, 0] * DT)) * t
    y_c = (nb.pos[:, 1] - nb.vel[:, 1] * DT) + (nb.pos[:, 1] - (nb.pos[:, 1] - nb.vel[:, 1] * DT)) * t
    goal = (jnp.abs(x_c) < T["GOAL_HALF_WIDTH"]) & (y_c < T["GOAL_HEIGHT"])
    out = jnp.where(stopped, SHORT, jnp.where(crossed, jnp.where(goal, GOAL, WIDE), out))
    cx = jnp.where(crossed, x_c, cx)
    cy = jnp.where(crossed, y_c, cy)
    cs = jnp.where(crossed, _norm(nb.vel), cs)
    m = live
    nb_pos = nb.pos
    if early_short:
        # kutálející se míč s exponenciálním útlumem urazí v z ještě přesně vz·f·dt/(1-f)
        # (zastavení pod STOP_SPEED to jen zkrátí) → když to na rovinu brány nestačí, je to
        # jistě NEDOLETĚL; ušetří stovky ticků dojezdu. end_pos = projektovaný bod zastavení.
        f = jnp.exp(-T["ROLL_FRICTION"] * DT)
        k = f * DT / (1.0 - f)
        hopeless = live & nb.rolling & nb.flying & ~crossed & (-nb.vel[:, 2] * k < nb.pos[:, 2] - plane)
        out = jnp.where(hopeless, SHORT, out)
        nb_pos = jnp.where(hopeless[:, None], nb.pos + nb.vel * k, nb.pos)
        stopped = stopped | hopeless
    nb = nb._replace(pos=nb_pos)
    done = done | stopped | crossed
    b = jax.tree_util.tree_map(lambda x, o: jnp.where(m.reshape((-1,) + (1,) * (x.ndim - 1)), x, o), nb, b)
    return b, done, out, cx, cy, cs, nt


def kick_from_state(kpos, kvel, kflying, kbounces, ox, oy, curve, power, dtick, T,
                    max_ticks: int = MAX_FLIGHT_TICKS, record: int = 0, early_short: bool = False,
                    inner: int = 1):
    """Kop z daného stavu serve míče. `record` > 0 → navíc vrátí trajektorii (record, B, 3)."""
    dtype = kpos.dtype
    err = dtick.astype(dtype) * DT
    late = (kbounces >= 2) | ~kflying
    hit, rating, lv, lw = launch_from_kick(kpos, kvel, ox, oy, curve, power, err, T)
    bsz = kpos.shape[0]
    b = launch(kpos, lv, lw)
    pre_done = late | ~hit
    out0 = jnp.where(late, LATE, jnp.where(~hit, WHIFF, SHORT)).astype(jnp.int32)
    zf = jnp.zeros(bsz, dtype)
    c = (b, pre_done, out0, zf, zf, zf, jnp.zeros(bsz, jnp.int32))
    traj = None
    if record:
        def sbody(c, _):
            c = _flight_step(c, T)
            return c, c[0].pos
        c, traj = jax.lax.scan(sbody, c, None, length=record)
    else:
        def cond(cc):
            c, i = cc
            return (i < max_ticks) & ~jnp.all(c[1])

        def body(cc):
            # `inner` ticků na iteraci: na GPU je drahá synchronizace podmínky while smyčky,
            # ne samotný tick (hotové kopy se maskují, takže víc ticků nic nemění)
            c, i = cc
            for _ in range(inner):
                c = _flight_step(c, T, early_short)
            return c, i + inner

        c, _ = jax.lax.while_loop(cond, body, (c, jnp.int32(0)))
    b, done, out, cx, cy, cs, nt = c
    res = KickResult(out, cx, cy, cs, nt, b.pos, rating, lv, lw)
    return (res, traj) if record else res


@partial(jax.jit, static_argnames=("inner",))
def kick_eval(win: Ball, w_half, idx, ox, oy, curve, power, dtick, T, inner: int = 8):
    """Dávka kopů nad předpočítaným oknem serve.

    win: Ball (S, 2w+1, …) ze serve_window; idx: (B,) index serve; dtick: (B,) int32,
    |dtick| ≤ w. Vrací KickResult (B,). Beznadějně kutálející se míče končí hned jako SHORT
    (výsledek stejný, jen `ticks`/`end_pos` jsou zkrácené/projektované)."""
    j = jnp.clip(dtick + w_half, 0, win.pos.shape[1] - 1)
    kpos = win.pos[idx, j]
    kvel = win.vel[idx, j]
    kfl = win.flying[idx, j]
    kbc = win.bounces[idx, j]
    return kick_from_state(kpos, kvel, kfl, kbc, ox, oy, curve, power, dtick, T, early_short=True,
                           inner=inner)


# ── skóre / kvalita gólu ─────────────────────────────────────

def goal_quality(res: KickResult, T):
    """Kvalita gólu ∈ [0,1]: třetina rychlost u čáry (25 m/s = plná), třetina blízkost
    tyče (|x| → GOAL_HALF_WIDTH), třetina výška (y → GOAL_HEIGHT). Mimo gól 0."""
    hw, h = T["GOAL_HALF_WIDTH"], T["GOAL_HEIGHT"]
    q = (jnp.clip(res.cross_speed / 25.0, 0.0, 1.0)
         + jnp.clip(jnp.abs(res.cross_x) / hw, 0.0, 1.0)
         + jnp.clip(res.cross_y / h, 0.0, 1.0)) / 3.0
    return jnp.where(res.outcome == GOAL, q, 0.0)


def kick_score(res: KickResult, T):
    """Spojitější cíl pro optimalizaci: gól = 1 + kvalita, jinak −(vzdálenost minutí v m)."""
    hw, h, gd = T["GOAL_HALF_WIDTH"], T["GOAL_HEIGHT"], T["GOAL_DISTANCE"]
    q = goal_quality(res, T)
    dx = jnp.maximum(jnp.abs(res.cross_x) - hw, 0.0)
    dy = jnp.maximum(res.cross_y - h, 0.0)
    wide = -jnp.sqrt(dx * dx + dy * dy) - 0.01
    ex = jnp.maximum(jnp.abs(res.end_pos[:, 0]) - hw, 0.0)
    short = -(jnp.maximum(gd + res.end_pos[:, 2], 0.0) + ex) - 0.01
    return jnp.where(res.outcome == GOAL, 1.0 + q,
           jnp.where(res.outcome == WIDE, wide,
           jnp.where(res.outcome == SHORT, short, -100.0)))
