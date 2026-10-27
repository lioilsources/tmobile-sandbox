"""UC14 krok 1 — Oracle: pro každý serve seed najdi kop s maximální kvalitou gólu.

  python oracle.py bench  --out UC14_bench.json          # kopů/s na GPU (různé batch)
  python oracle.py oracle --serves 10000 --out UC14_oracle.json
  python oracle.py d1d2   --serves 2000  --out UC14_d1d2.json

Oracle = hrubý grid nad 5 parametry kopu (offset x/y, oblouk, síla, posun puštění v ticích)
→ start pro dávkový CMA-ES (vlastní implementace v JAX, všechny serve paralelně; evosax
nepoužit — sdílený venv má jax 0.4.33, novější evosax chce novější jax).
Serve seed s = první `ServeGenerator.new(s).next_serve()` (bitově shodné s Godotem).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import volley_jax as vj  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

W = 14   # |dtick| ≤ 14 ⇔ |err| ≤ 116.7 ms < RATING_LATE_EARLY (120 ms); dál už je to WHIFF
CHUNK = 1 << 20


def load(overrides=None):
    tun = vj.parse_tuning(vj.default_tuning_path())
    return tun, vj.tuning_pytree(tun, overrides)


def prep_serves(seeds, tun, T):
    S = vj.godot_serves(seeds, tun)
    pos, vel, om = (jnp.asarray(S[k], vj.DTYPE) for k in ("pos", "vel", "omega"))
    ideal = vj.presim_ideal(pos, vel, om, T)
    win = vj.serve_window(pos, vel, om, ideal, T, w=W)
    return S, np.asarray(ideal), win


def x_to_params(x):
    """Normalizovaný prostor [-1,1]^5 → parametry kopu."""
    x = jnp.clip(x, -1.0, 1.0)
    return (x[..., 0] * 0.95, x[..., 1] * 0.95, x[..., 2], (x[..., 3] + 1.0) * 0.5,
            jnp.round(x[..., 4] * W).astype(jnp.int32))


def params_to_x(ox, oy, cu, pw, dt):
    return np.stack([ox / 0.95, oy / 0.95, cu, pw * 2.0 - 1.0, dt / W], -1)


@jax.jit
def _eval(win, idx, ox, oy, cu, pw, dt, T):
    r = vj.kick_eval(win, W, idx, ox.astype(vj.DTYPE), oy.astype(vj.DTYPE), cu.astype(vj.DTYPE),
                     pw.astype(vj.DTYPE), dt, T)
    return r.outcome, vj.kick_score(r, T), vj.goal_quality(r, T), r.cross_x, r.cross_y, r.cross_speed, r.ticks


def eval_flat(win, idx, ox, oy, cu, pw, dt, T, chunk=CHUNK):
    """Vyhodnotí libovolně dlouhou dávku po kusech (pevná velikost → jedna kompilace)."""
    n = len(idx)
    outs = []
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        pad = chunk - (e - s) if n > chunk else 0
        sl = lambda a: np.concatenate([a[s:e], np.repeat(a[e - 1:e], pad, 0)]) if pad else a[s:e]
        o = _eval(win, jnp.asarray(sl(idx)), jnp.asarray(sl(ox)), jnp.asarray(sl(oy)), jnp.asarray(sl(cu)),
                  jnp.asarray(sl(pw)), jnp.asarray(sl(dt)), T)
        outs.append([np.asarray(a)[: e - s] for a in o])
    return [np.concatenate(z) for z in zip(*outs)]


def make_grid(ox, oy, cu, pw, dt):
    g = np.meshgrid(ox, oy, cu, pw, dt, indexing="ij")
    return [a.ravel() for a in g]


# ── dávkový CMA-ES (všechny serve najednou) ──────────────────

def cma_es(win, T, x0, sigma0=0.15, lam=16, gens=30, seed=0):
    S, n = x0.shape
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)); w /= w.sum()
    mueff = 1.0 / np.sum(w ** 2)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chin = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    rng = np.random.default_rng(seed)
    m = x0.copy(); sig = np.full(S, sigma0); C = np.tile(np.eye(n), (S, 1, 1))
    pc = np.zeros((S, n)); ps = np.zeros((S, n))
    best_f = np.full(S, -np.inf); best_x = x0.copy()
    idx = np.repeat(np.arange(S), lam)
    for g in range(gens):
        evals, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(evals, 1e-20))
        z = rng.standard_normal((S, lam, n))
        y = np.einsum("sij,skj->ski", B * D[:, None, :], z)
        x = m[:, None, :] + sig[:, None, None] * y
        p = x_to_params(jnp.asarray(x.reshape(-1, n)))
        out = eval_flat(win, idx, *[np.asarray(a) for a in p], T)
        f = out[1].reshape(S, lam) - 0.1 * np.linalg.norm(x - np.clip(x, -1, 1), axis=-1)
        order = np.argsort(-f, axis=1)
        top = np.take_along_axis(f, order[:, :1], 1)[:, 0]
        upd = top > best_f
        best_f = np.where(upd, top, best_f)
        best_x[upd] = np.take_along_axis(x, order[:, :1, None], 1)[upd, 0]
        ysel = np.take_along_axis(y, order[:, :mu, None], 1)
        yw = np.einsum("k,skn->sn", w, ysel)
        m = m + sig[:, None] * yw
        Cinvsqrt = np.einsum("sij,sj,skj->sik", B, 1 / D, B)
        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * np.einsum("sij,sj->si", Cinvsqrt, yw)
        hsig = (np.linalg.norm(ps, axis=1) / np.sqrt(1 - (1 - cs) ** (2 * (g + 1))) / chin) < (1.4 + 2 / (n + 1))
        pc = (1 - cc) * pc + (hsig * math.sqrt(cc * (2 - cc) * mueff))[:, None] * yw
        rank_mu = np.einsum("k,ski,skj->sij", w, ysel, ysel)
        C = ((1 - c1 - cmu) * C + c1 * (np.einsum("si,sj->sij", pc, pc)
             + ((1 - hsig) * cc * (2 - cc))[:, None, None] * C) + cmu * rank_mu)
        sig = sig * np.exp((cs / damps) * (np.linalg.norm(ps, axis=1) / chin - 1))
        sig = np.minimum(sig, 1.0)
    return best_x, best_f


# ── režimy ───────────────────────────────────────────────────

def cmd_bench(a):
    tun, T = load()
    seeds = np.arange(1000)
    _, _, win = prep_serves(seeds, tun, T)
    rng = np.random.default_rng(0)
    res = {"device": str(jax.devices()[0]), "dtype": str(np.dtype(vj.DTYPE)), "runs": []}
    for bs in a.batches:
        idx = rng.integers(0, 1000, bs)
        ox = rng.uniform(-0.6, 0.6, bs).astype(np.float32); oy = rng.uniform(-0.8, 0.2, bs).astype(np.float32)
        cu = rng.uniform(-1, 1, bs).astype(np.float32); pw = rng.uniform(0.5, 1, bs).astype(np.float32)
        dt = rng.integers(-6, 7, bs).astype(np.int32)
        args = [jnp.asarray(v) for v in (idx, ox, oy, cu, pw, dt)]
        t0 = time.time(); o = _eval(win, *args, T); jax.block_until_ready(o); tc = time.time() - t0
        ts = []
        for _ in range(a.reps):
            t0 = time.time(); o = _eval(win, *args, T); jax.block_until_ready(o); ts.append(time.time() - t0)
        best = min(ts)
        oc = np.bincount(np.asarray(o[0]), minlength=5)
        r = {"batch": bs, "compile_s": round(tc, 3), "best_s": round(best, 4), "kicks_per_s": bs / best,
             "mean_flight_ticks": float(np.asarray(o[6]).mean()), "max_flight_ticks": int(np.asarray(o[6]).max()),
             "outcomes": dict(zip(vj.OUTCOME_NAMES, map(int, oc)))}
        print(r, flush=True)
        res["runs"].append(r)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


ORACLE_GRID = dict(ox=np.linspace(-0.8, 0.8, 9), oy=np.linspace(-0.9, 0.3, 9),
                   cu=np.array([-1, -0.5, 0, 0.5, 1.0]), pw=np.array([0.55, 0.7, 0.85, 1.0]),
                   dt=np.array([-8, -5, -3, -1, 0, 1, 3, 5, 8]))


def grid_search(win, T, n_serves, grid, serve_chunk=64):
    g = make_grid(*grid.values())
    G = len(g[0])
    best_i = np.zeros(n_serves, np.int64); best_f = np.zeros(n_serves); n_goal = np.zeros(n_serves, np.int64)
    best_q = np.zeros(n_serves)
    for s0 in range(0, n_serves, serve_chunk):
        s1 = min(n_serves, s0 + serve_chunk)
        k = s1 - s0
        idx = np.repeat(np.arange(s0, s1), G)
        args = [np.tile(a, k).astype(np.float32) for a in g[:4]] + [np.tile(g[4], k).astype(np.int32)]
        out = eval_flat(win, idx, *args, T)
        f = out[1].reshape(k, G)
        best_i[s0:s1] = f.argmax(1); best_f[s0:s1] = f.max(1)
        n_goal[s0:s1] = (out[0].reshape(k, G) == vj.GOAL).sum(1)
        best_q[s0:s1] = out[2].reshape(k, G).max(1)
    return g, best_i, best_f, n_goal, best_q, G


def cmd_oracle(a):
    tun, T = load()
    seeds = np.arange(a.seed0, a.seed0 + a.serves)
    t_all = time.time()
    S, ideal, win = prep_serves(seeds, tun, T)
    jax.block_until_ready(win.pos)
    t0 = time.time()
    g, bi, bf, ng, bq, G = grid_search(win, T, a.serves, ORACLE_GRID)
    t_grid = time.time() - t0
    n_grid_kicks = G * a.serves
    print(f"grid: {n_grid_kicks/1e6:.1f} M kopů za {t_grid:.1f} s → {n_grid_kicks/t_grid/1e6:.2f} M/s; "
          f"serve s gólem v gridu {np.mean(ng > 0):.4f}", flush=True)

    x0 = params_to_x(*(np.asarray(gg)[bi] for gg in g))
    t0 = time.time()
    bx, bfc = cma_es(win, T, x0, sigma0=a.sigma0, lam=a.lam, gens=a.gens)
    t_cma = time.time() - t0
    n_cma = a.serves * a.lam * a.gens
    print(f"CMA-ES: {n_cma/1e6:.1f} M kopů za {t_cma:.1f} s", flush=True)

    # finální vyhodnocení nejlepšího kopu (grid vs CMA) per serve
    p_cma = [np.asarray(v) for v in x_to_params(jnp.asarray(bx))]
    p_grid = [np.asarray(gg)[bi] for gg in g]
    idx = np.arange(a.serves)
    oc = eval_flat(win, idx, *[p.astype(np.float32 if i < 4 else np.int32) for i, p in enumerate(p_cma)], T)
    og = eval_flat(win, idx, *[p.astype(np.float32 if i < 4 else np.int32) for i, p in enumerate(p_grid)], T)
    use_cma = oc[1] >= og[1]
    pick = lambda c, gg: np.where(use_cma, c, gg)
    best = {k: pick(c, gg) for k, c, gg in zip(["ox", "oy", "curve", "power", "dtick"], p_cma, p_grid)}
    outc = pick(oc[0], og[0]); qual = pick(oc[2], og[2])
    cx, cy, cs = pick(oc[3], og[3]), pick(oc[4], og[4]), pick(oc[5], og[5])
    solved = outc == vj.GOAL
    t_total = time.time() - t_all

    # rysy serve pro mapu "kde trefit při jakém nadhozu"
    j = W
    kp = np.asarray(win.pos[:, j]); kv = np.asarray(win.vel[:, j])
    feats = {"kick_height": kp[:, 1], "kick_x": kp[:, 0], "kick_z": kp[:, 2],
             "in_vx": kv[:, 0], "in_vy": kv[:, 1], "in_vz": kv[:, 2], "ideal_tick": ideal}
    corr = {f: {p: float(np.corrcoef(v[solved], best[p][solved])[0, 1]) for p in ("ox", "oy", "curve", "power", "dtick")}
            for f, v in feats.items()}
    q = lambda v: {"p05": float(np.quantile(v, .05)), "p50": float(np.median(v)), "p95": float(np.quantile(v, .95))}
    # binned map: kick height × oy
    hb = np.quantile(feats["kick_height"], [0, .2, .4, .6, .8, 1])
    height_map = []
    for lo, hi in zip(hb[:-1], hb[1:]):
        m = solved & (feats["kick_height"] >= lo) & (feats["kick_height"] <= hi)
        height_map.append({"kick_height_m": [float(lo), float(hi)], "n": int(m.sum()),
                           "best_oy": q(best["oy"][m]), "best_power": q(best["power"][m]),
                           "best_dtick": q(best["dtick"][m].astype(float))})
    unsolved = seeds[~solved]
    rep = {
        "device": str(jax.devices()[0]), "dtype": str(np.dtype(vj.DTYPE)),
        "serves": a.serves, "seed_range": [int(seeds[0]), int(seeds[-1])],
        "grid": {k: v.tolist() for k, v in ORACLE_GRID.items()}, "grid_per_serve": G,
        "grid_kicks": n_grid_kicks, "grid_seconds": t_grid, "grid_kicks_per_s": n_grid_kicks / t_grid,
        "cma": {"lam": a.lam, "gens": a.gens, "sigma0": a.sigma0, "kicks": n_cma, "seconds": t_cma},
        "total_seconds": t_total,
        "solved_frac": float(solved.mean()), "unsolved_frac": float(1 - solved.mean()),
        "unsolved_seeds": unsolved[:200].tolist(),
        "serves_with_goal_in_grid_frac": float(np.mean(ng > 0)),
        "grid_goal_frac_per_serve": q(ng / G),
        "cma_improved_frac": float(np.mean(use_cma & (oc[1] > og[1] + 1e-6))),
        "best_quality": q(qual[solved]), "best_cross_speed_ms": q(cs[solved]),
        "best_cross_x_abs": q(np.abs(cx[solved])), "best_cross_y": q(cy[solved]),
        "best_params": {k: q(v[solved].astype(float)) for k, v in best.items()},
        "best_dtick_hist": {int(k): int(v) for k, v in zip(*np.unique(best["dtick"][solved], return_counts=True))},
        "corr_serve_feature_vs_best_param": corr,
        "height_map": height_map,
    }
    print(json.dumps({k: v for k, v in rep.items() if k not in ("unsolved_seeds", "corr_serve_feature_vs_best_param", "height_map")}, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))
        per = {"seed": seeds.tolist(), **{k: v.astype(float).round(5).tolist() for k, v in best.items()},
               "outcome": outc.tolist(), "quality": qual.round(4).tolist(),
               "kick_height": feats["kick_height"].round(4).tolist(), "in_vz": feats["in_vz"].round(4).tolist()}
        Path(a.out).with_name(Path(a.out).stem + "_map.json").write_text(json.dumps(per))
        # kopy pro Godot cross-check (prvních N serve s nejlepším kopem)
        kk = [{"seed": int(seeds[i]), "ox": float(np.float32(best["ox"][i])), "oy": float(np.float32(best["oy"][i])),
               "curve": float(np.float32(best["curve"][i])), "power": float(np.float32(best["power"][i])),
               "dtick": int(best["dtick"][i])} for i in range(min(1000, a.serves))]
        Path(a.out).with_name(Path(a.out).stem + "_kicks1000.json").write_text(json.dumps(kk))


STRONG_Y, STRONG_V = 0.5, 15.0
D1_GRID = dict(ox=np.linspace(-0.6, 0.6, 7), oy=np.linspace(-0.9, 0.3, 9), cu=np.array([-1.0, 0.0, 1.0]),
               pw=np.array([0.6, 0.8, 1.0]), dt=np.arange(-W, W + 1))


def cmd_d1d2(a):
    tun, T0 = load()
    seeds = np.arange(a.seed0, a.seed0 + a.serves)
    rep = {"device": str(jax.devices()[0]), "serves": a.serves, "grid": {k: v.tolist() for k, v in D1_GRID.items()},
           "D1": {}, "D2": {}}
    g = make_grid(*D1_GRID.values())
    G = len(g[0]); nd = len(D1_GRID["dt"])
    for sigma in a.sigmas:
        tun_o, T = load({"TIMING_SIGMA": sigma})
        S, ideal, win = prep_serves(seeds, tun, T)
        t0 = time.time()
        goal_dt = np.zeros((a.serves, nd), bool); strong_dt = np.zeros((a.serves, nd), bool); goal_cnt = np.zeros(a.serves); rating_goal = np.zeros((3, 2))
        sc = 32
        for s0 in range(0, a.serves, sc):
            s1 = min(a.serves, s0 + sc); k = s1 - s0
            idx = np.repeat(np.arange(s0, s1), G)
            args = [np.tile(x, k).astype(np.float32) for x in g[:4]] + [np.tile(g[4], k).astype(np.int32)]
            out = eval_flat(win, idx, *args, T)
            isg = (out[0] == vj.GOAL).reshape(k, -1, nd)  # grid je ravel v pořadí (…, dt) → dt poslední osa
            strong = ((out[0] == vj.GOAL) & (out[4] > STRONG_Y) & (out[5] >= STRONG_V)).reshape(k, -1, nd)
            strong_dt[s0:s1] = strong.any(1)
            goal_dt[s0:s1] = isg.any(1)
            goal_cnt[s0:s1] = isg.sum((1, 2))
            adt = np.abs(np.tile(g[4], k))
            for ri, (lo, hi) in enumerate([(0, 3), (4, 8), (9, 14)]):
                m = (adt >= lo) & (adt <= hi)
                rating_goal[ri] += [(out[0][m] == vj.GOAL).sum(), m.sum()]
        dtv = D1_GRID["dt"]
        # šířka timing okna: počet ticků, kdy existuje gól (×8.33 ms)
        width_ms = goal_dt.sum(1) * 1000.0 / vj.TICKS_PER_SEC
        # speed tabulka z ROADMAP D1: středový kop, plná síla, míč v klidu
        errs = np.array([0, 0.03, 0.05, 0.07, 0.09])
        _, _, v, _ = vj.contact(jnp.zeros(5), jnp.zeros(5), jnp.zeros(5), jnp.ones(5), jnp.asarray(errs, vj.DTYPE),
                                jnp.zeros((5, 3), vj.DTYPE), T)
        rep["D1"][f"sigma_{int(round(sigma*1000))}ms"] = {
            "timing_sigma": sigma, "seconds": time.time() - t0, "kicks": int(G * a.serves),
            "serves_with_goal_frac": float(goal_dt.any(1).mean()),
            "grid_goal_frac": float(goal_cnt.sum() / (G * a.serves)),
            "goal_exists_by_dtick_frac": {int(d): float(goal_dt[:, i].mean()) for i, d in enumerate(dtv)},
            "goal_exists_by_err_ms": {f"{d*1000/120:+.0f}": float(goal_dt[:, i].mean()) for i, d in enumerate(dtv)},
            "strong_goal_def": f"GÓL a průlet výš než {STRONG_Y} m a rychleji než {STRONG_V} m/s (ne dokutálený)",
            "strong_goal_exists_by_err_ms": {f"{d*1000/120:+.0f}": float(strong_dt[:, i].mean()) for i, d in enumerate(dtv)},
            "strong_timing_window_width_ms": {"p05": float(np.quantile(strong_dt.sum(1) * 1000 / 120, .05)),
                                              "p50": float(np.median(strong_dt.sum(1) * 1000 / 120))},
            "timing_window_width_ms": {"p05": float(np.quantile(width_ms, .05)), "p50": float(np.median(width_ms)),
                                       "p95": float(np.quantile(width_ms, .95))},
            "goal_frac_by_rating_band": {b: float(rg[0] / rg[1]) for b, rg in
                                         zip(["PERFECT(|err|<=25ms)", "GOOD(33-67ms)", "LATE_EARLY(75-117ms)"], rating_goal)},
            "center_kick_speed_vs_err": {f"{int(e*1000)}ms": float(np.linalg.norm(np.asarray(v)[i])) for i, e in enumerate(errs)},
        }
        print(json.dumps(rep["D1"][f"sigma_{int(round(sigma*1000))}ms"], indent=1), flush=True)

    # D2: středový zásah, plná síla, ideální timing, bez oblouku
    tun, T = load()
    S, ideal, win = prep_serves(seeds, tun, T)
    n = a.serves
    idx = np.arange(n)
    r = vj.kick_eval(win, W, jnp.asarray(idx), jnp.zeros(n, vj.DTYPE), jnp.zeros(n, vj.DTYPE),
                     jnp.zeros(n, vj.DTYPE), jnp.ones(n, vj.DTYPE), jnp.zeros(n, jnp.int32), T)
    lv = np.asarray(r.launch_vel)
    elev = np.degrees(np.arctan2(lv[:, 1], np.hypot(lv[:, 0], lv[:, 2])))
    oc = np.bincount(np.asarray(r.outcome), minlength=5)
    # jaký offset y je potřeba: sken oy pro středový kop (ox=0, curve 0, power 1, dtick 0)
    oys = np.round(np.linspace(-0.9, 0.3, 25), 3)
    goal_by_oy = {}
    elev_by_oy = {}
    for oy in oys:
        rr = vj.kick_eval(win, W, jnp.asarray(idx), jnp.zeros(n, vj.DTYPE), jnp.full(n, oy, vj.DTYPE),
                          jnp.zeros(n, vj.DTYPE), jnp.ones(n, vj.DTYPE), jnp.zeros(n, jnp.int32), T)
        goal_by_oy[f"{oy:+.2f}"] = float(np.mean(np.asarray(rr.outcome) == vj.GOAL))
        lv2 = np.asarray(rr.launch_vel)
        elev_by_oy[f"{oy:+.2f}"] = float(np.median(np.degrees(np.arctan2(lv2[:, 1], np.hypot(lv2[:, 0], lv2[:, 2])))))
    # sonda z ROADMAP: míč v klidu, offset 0
    _, _, v0, _ = vj.contact(jnp.zeros(1), jnp.zeros(1), jnp.zeros(1), jnp.ones(1), jnp.zeros(1, vj.DTYPE),
                             jnp.zeros((1, 3), vj.DTYPE), T)
    rep["D2"] = {
        "center_kick_ball_at_rest_vel": np.asarray(v0)[0].tolist(),
        "center_kick_launch_elevation_deg": {"p05": float(np.quantile(elev, .05)), "p50": float(np.median(elev)),
                                             "p95": float(np.quantile(elev, .95))},
        "center_kick_outcomes": dict(zip(vj.OUTCOME_NAMES, map(int, oc))),
        "center_kick_goal_frac": float(oc[0] / n),
        "center_kick_cross_y": {"p50": float(np.median(np.asarray(r.cross_y))), "p95": float(np.quantile(np.asarray(r.cross_y), .95))},
        "center_kick_cross_speed": {"p50": float(np.median(np.asarray(r.cross_speed)))},
        "center_kick_flight_ticks_p50": float(np.median(np.asarray(r.ticks))),
        "goal_frac_by_offset_y": goal_by_oy,
        "median_elevation_deg_by_offset_y": elev_by_oy,
    }
    print(json.dumps(rep["D2"], indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("bench"); b.add_argument("--batches", type=int, nargs="+", default=[1 << 14, 1 << 17, 1 << 20])
    b.add_argument("--reps", type=int, default=3); b.add_argument("--out")
    o = sp.add_parser("oracle"); o.add_argument("--serves", type=int, default=10000); o.add_argument("--seed0", type=int, default=0)
    o.add_argument("--lam", type=int, default=16); o.add_argument("--gens", type=int, default=30)
    o.add_argument("--sigma0", type=float, default=0.15); o.add_argument("--out")
    d = sp.add_parser("d1d2"); d.add_argument("--serves", type=int, default=2000); d.add_argument("--seed0", type=int, default=0)
    d.add_argument("--sigmas", type=float, nargs="+", default=[0.045, 0.09]); d.add_argument("--out")
    a = ap.parse_args()
    {"bench": cmd_bench, "oracle": cmd_oracle, "d1d2": cmd_d1d2}[a.cmd](a)


if __name__ == "__main__":
    main()
