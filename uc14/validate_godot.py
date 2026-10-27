"""UC14 validace: N náhodných kopů JAX vs Godot headless (tests/tools/kick_dump.gd).

  FOTBAL_REPO=…/FotballSimulator GODOT=…/Godot python validate_godot.py --n 1000 --out UC14_validation.json

Kroky: (1) vygeneruje kopy (parametry reprezentovatelné ve float32, jako Vector2 v Godotu),
(2) pustí Godot dump (nebo načte hotový --godot-json), (3) přepočítá stejné kopy v JAXu
(dtype podle VOLLEY_X64), (4) porovná serve (bitově), ideální tick, rychlost/spin po kontaktu,
celou trajektorii letu po ticku, bod průletu a výsledek. Tolerance 1e-4 (absolutní, m · m/s).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import volley_jax as vj  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def gen_kicks(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    f = lambda lo, hi: rng.uniform(lo, hi, n).astype(np.float32)
    ox, oy, cu, pw = f(-0.7, 0.7), f(-0.8, 0.5), f(-1, 1), f(0.2, 1.0)
    dt = np.where(rng.random(n) < 0.85, rng.integers(-9, 10, n), rng.integers(-16, 17, n))
    seeds = rng.integers(0, 2**31 - 1, n)
    return [{"seed": int(seeds[i]), "ox": float(ox[i]), "oy": float(oy[i]), "curve": float(cu[i]),
             "power": float(pw[i]), "dtick": int(dt[i])} for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--work", default="/tmp/uc14_val")
    ap.add_argument("--godot-json", default=None, help="hotový výstup kick_dump.gd (přeskočí běh Godotu)")
    ap.add_argument("--kicks", default=None, help="vlastní seznam kopů (JSON) místo náhodných")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    repo = Path(os.environ["FOTBAL_REPO"])
    tun = vj.parse_tuning(repo / "scripts/util/tuning.gd")
    T = vj.tuning_pytree(tun)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    kicks = json.loads(Path(a.kicks).read_text()) if a.kicks else gen_kicks(a.n, a.seed)
    a.n = len(kicks)
    kin = work / "kicks_in.json"
    kin.write_text(json.dumps(kicks))

    if a.godot_json:
        gout = Path(a.godot_json)
        godot_s = None
    else:
        gout = work / "kicks_godot.json"
        godot = os.environ["GODOT"]
        t0 = time.time()
        subprocess.run([godot, "--headless", "--path", str(repo), "--script",
                        "res://tests/tools/kick_dump.gd", "--", str(kin), str(gout)], check=True,
                       stdout=subprocess.DEVNULL)
        godot_s = time.time() - t0
    G = json.loads(gout.read_text())
    assert len(G) == len(kicks)

    seeds = np.array([k["seed"] for k in kicks])
    report = {"n": a.n, "tol": a.tol, "dtype": str(np.dtype(vj.DTYPE)), "godot_seconds": godot_s}

    # (1) serve generator — FMA vs bez FMA (arm64 build kontrahuje a*b+c)
    g_pos = np.array([g["serve_pos"] for g in G]); g_vel = np.array([g["serve_vel"] for g in G])
    g_om = np.array([g["serve_omega"] for g in G])
    best = None
    for fma in (True, False):
        S = vj.godot_serves(seeds, tun, fma=fma)
        d = max(np.abs(S["pos"] - g_pos).max(), np.abs(S["vel"] - g_vel).max(), np.abs(S["omega"] - g_om).max())
        exact = float(np.mean(np.all(S["pos"] == g_pos, 1) & np.all(S["vel"] == g_vel, 1) & np.all(S["omega"] == g_om, 1)))
        report[f"serve_fma_{fma}"] = {"max_abs_diff": float(d), "bitwise_equal_frac": exact}
        if best is None or d < best[0]:
            best = (d, fma, S)
    S = best[2]
    report["serve_rng_fma"] = best[1]

    # (2) ideální tick
    pos, vel, om = (jnp.asarray(S[k], vj.DTYPE) for k in ("pos", "vel", "omega"))
    ideal = np.asarray(vj.presim_ideal(pos, vel, om, T))
    g_ideal = np.array([g["ideal_tick"] for g in G])
    report["ideal_tick_equal_frac"] = float(np.mean(ideal == g_ideal))

    # (3) kopy s plnou trajektorií
    W = 16
    win = vj.serve_window(pos, vel, om, jnp.asarray(ideal), T, w=W)
    arr = lambda k, dt=vj.DTYPE: jnp.asarray(np.array([x[k] for x in kicks]), dt)
    dtick = jnp.asarray(np.array([x["dtick"] for x in kicks]), jnp.int32)
    j = dtick + W
    idx = jnp.arange(a.n)
    kp, kv = win.pos[idx, j], win.vel[idx, j]
    import jax
    rec = jax.jit(lambda *xs: vj.kick_from_state(*xs, T, record=vj.MAX_FLIGHT_TICKS))
    t0 = time.time()
    res, traj = rec(kp, kv, win.flying[idx, j], win.bounces[idx, j], arr("ox"), arr("oy"), arr("curve"),
                    arr("power"), dtick)
    traj = np.asarray(traj).transpose(1, 0, 2)
    report["jax_seconds_incl_compile"] = time.time() - t0

    names = vj.OUTCOME_NAMES
    out = np.asarray(res.outcome)
    rows = []
    max_traj, max_launch, max_cross, max_kick = [], [], [], []
    outcome_eq = tick_eq = 0
    for i, g in enumerate(G):
        o_j = names[out[i]]
        o_g = g["outcome"]
        outcome_eq += o_j == o_g
        dk = float(np.abs(np.asarray(kp[i]) - np.array(g["kick_pos"])).max())
        max_kick.append(dk)
        r = {"i": i, "godot": o_g, "jax": o_j, "kick_pos_diff": dk}
        if "launch_vel" in g and o_j not in ("WHIFF", "LATE"):
            dl = max(np.abs(np.asarray(res.launch_vel[i]) - np.array(g["launch_vel"])).max(),
                     np.abs(np.asarray(res.launch_omega[i]) - np.array(g["launch_omega"])).max())
            max_launch.append(float(dl)); r["launch_diff"] = float(dl)
            gt = np.array(g["traj"]).reshape(-1, 3)
            nt = int(res.ticks[i])
            tick_eq += nt == len(gt)
            m = min(nt, len(gt))
            dtj = float(np.abs(traj[i, :m] - gt[:m]).max()) if m else 0.0
            max_traj.append(dtj); r["traj_diff"] = dtj; r["ticks"] = [nt, len(gt)]
            if "cross" in g and o_j in ("GOAL", "WIDE"):
                dc = max(abs(float(res.cross_x[i]) - g["cross"][0]), abs(float(res.cross_y[i]) - g["cross"][1]))
                max_cross.append(dc); r["cross_diff"] = dc
        else:
            tick_eq += 1
        rows.append(r)

    def stats(v):
        v = np.array(v)
        return {"n": int(len(v)), "max": float(v.max()) if len(v) else 0.0,
                "p99": float(np.quantile(v, 0.99)) if len(v) else 0.0,
                "median": float(np.median(v)) if len(v) else 0.0,
                "within_tol_frac": float(np.mean(v <= a.tol)) if len(v) else 1.0}

    report.update({
        "outcome_equal_frac": outcome_eq / a.n,
        "flight_tick_count_equal_frac": tick_eq / a.n,
        "kick_state_pos_diff": stats(max_kick),
        "launch_vel_omega_diff": stats(max_launch),
        "trajectory_pos_diff": stats(max_traj),
        "goal_plane_cross_diff": stats(max_cross),
        "outcomes_godot": {k: sum(g["outcome"] == k for g in G) for k in names},
        "mismatches": [r for r in rows if r["godot"] != r["jax"] or r.get("traj_diff", 0) > a.tol][:30],
    })
    print(json.dumps({k: v for k, v in report.items() if k != "mismatches"}, indent=1))
    print("mismatches:", len([r for r in rows if r["godot"] != r["jax"] or r.get("traj_diff", 0) > a.tol]))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
