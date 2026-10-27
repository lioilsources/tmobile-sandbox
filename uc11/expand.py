#!/usr/bin/env python3
"""UC11: rozbalí grid.yaml na JSONL specifikace běhů pro `grid` runner (AukrofySimulation/cmd/grid).

    python expand.py grid.yaml specs.jsonl [--subset smoke] [--replicates N]

Každý běh nese explicitní pool bidderů (budget, valuace už vylosované) → runner je čistý executor
a běh je reprodukovatelný z jednoho řádku. Parametry formátu se uvnitř buňky losují Latin
hypercube (stratifikovaně přes replikace).
"""
import argparse
import hashlib
import json
import sys

import numpy as np
import yaml


def cell_rng(seed, *parts):
    h = hashlib.sha256(("|".join(map(str, (seed,) + parts))).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def alloc(shares, n):
    """Largest remainder: podíly rolí → celočíselné počty se součtem n."""
    roles = list(shares)
    w = np.array([shares[r] for r in roles], float)
    q = w / w.sum() * n
    base = np.floor(q).astype(int)
    rest = n - base.sum()
    order = np.argsort(-(q - base), kind="stable")
    base[order[:rest]] += 1
    return {r: int(c) for r, c in zip(roles, base) if c > 0}


def lhs(rng, ranges, n):
    """ranges: {name: [lo, hi]} → {name: array(n)} stratifikovaně (LHS)."""
    out = {}
    for k, (lo, hi) in ranges.items():
        u = (rng.permutation(n) + rng.random(n)) / n
        out[k] = lo + u * (hi - lo)
    return out


def split_params(fmt_cfg):
    ranges, fixed = {}, {}
    for k, v in fmt_cfg.items():
        if isinstance(v, list) and len(v) == 2:
            ranges[k] = [float(v[0]), float(v[1])]
        else:
            fixed[k] = v
    return ranges, fixed


def build_params(fmt, p, retail, fee_pct):
    r = retail
    if fmt == "DUTCH":
        return dict(start_price=round(r * p["start_mult"], 2), decrement=round(r * p["step_pct"] / 100, 2),
                    reserve_price=round(r * p["reserve_mult"], 2), tick_s=round(p["tick_s"]),
                    max_ticks=int(p["max_ticks"]), platform_fee_pct=fee_pct)
    if fmt == "VICKREY":
        res = round(r * p["reserve_mult"], 2)
        # engine ukazuje agentům start_price jako „rezervní cenu (minimum)“
        return dict(start_price=res, reserve_price=res, window_s=p["window_s"], platform_fee_pct=fee_pct)
    if fmt == "PENNY":
        return dict(start_price=0.0, bid_fee=round(r * p["bid_fee_pct"] / 100, 2),
                    price_increment=round(r * p["increment_pct"] / 100, 2), timer_s=round(p["timer_s"]),
                    tick_s=p["tick_s"], max_ticks=int(p["max_ticks"]), platform_fee_pct=fee_pct)
    raise ValueError(fmt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("out")
    ap.add_argument("--subset")
    ap.add_argument("--replicates", type=int)
    ap.add_argument("--formats", help="čárkou oddělené, např. DUTCH,PENNY")
    a = ap.parse_args()

    g = yaml.safe_load(open(a.grid))
    seed = g["seed"]
    reps = g["replicates"]
    pops = list(g["populations"])
    values = list(g["goods"]["value"])
    elas = list(g["goods"]["elasticity"])
    fmts = list(g["formats"])
    if a.subset:
        s = g["subsets"][a.subset]
        reps = s.get("replicates", reps)
        pops = s.get("populations", pops)
        values = s.get("value", values)
        elas = s.get("elasticity", elas)
        fmts = s.get("formats", fmts)
    if a.replicates:
        reps = a.replicates
    if a.formats:
        fmts = a.formats.split(",")

    n = g["n_bidders"]
    bmin, bmax = g["goods"]["budget_mult"]
    rmult = g["role_valuation_mult"]
    fee_pct = g["platform_fee_pct"]
    specs = []
    if True:
        for fmt in fmts:
            ranges, fixed = split_params(g["formats"][fmt])
            for pop in pops:
                counts = alloc(g["populations"][pop], n)
                tot = sum(g["populations"][pop].values())
                shares = {r: g["populations"][pop].get(r, 0) / tot for r in rmult}
                for val in values:
                    retail = float(g["goods"]["value"][val])
                    for el in elas:
                        e = g["goods"]["elasticity"][el]
                        cell = f"{fmt}|{pop}|{val}|{el}"
                        rng = cell_rng(seed, cell)
                        draws = lhs(rng, ranges, reps)
                        for rep in range(reps):
                            p = dict(fixed)
                            norm = {}
                            for k, (lo, hi) in ranges.items():
                                p[k] = float(draws[k][rep])
                                norm[k] = (p[k] - lo) / (hi - lo) if hi > lo else 0.0
                            # bidder pool: stejný seed populace+zboží+replikace napříč formáty →
                            # formáty se srovnávají na STEJNÝCH kupujících (common random numbers)
                            brng = cell_rng(seed, "bidders", pop, val, el, rep)
                            bidders = []
                            for role, c in counts.items():
                                for i in range(c):
                                    v = retail * rmult[role] * e["level"] * brng.uniform(1 - e["spread"], 1 + e["spread"])
                                    b = retail * brng.uniform(bmin, bmax)
                                    bidders.append(dict(id=f"{role}-{len(bidders)+1}", role=role,
                                                        budget=round(b, 2), valuation=round(v, 2)))
                            rid = f"{cell}#{rep}"
                            spec = dict(
                                run_id=rid, cell=cell,
                                seed=int(cell_rng(seed, rid).integers(1, 2**31 - 1)),
                                format=fmt,
                                item=dict(name={"low": "Bezdrátová sluchátka", "mid": "Herní konzole",
                                                "high": "Ojeté auto"}.get(val, val),
                                          retail_price=retail, quantity=1, category=val),
                                params=dict(build_params(fmt, p, retail, fee_pct),
                                            enforce_budget=bool(g.get("enforce_budget", False))),
                                bidders=bidders,
                                factors=dict(format=fmt, population=pop, value=val, elasticity=el,
                                             retail=retail, rep=rep,
                                             **{f"share_{r.lower()}": round(s_, 4) for r, s_ in shares.items()},
                                             **{f"p_{k}": round(v_, 5) for k, v_ in p.items() if k in ranges},
                                             **{f"u_{k}": round(v_, 5) for k, v_ in norm.items()}),
                            )
                            specs.append(spec)
    # deterministicky zamíchat: GPU dostává mix formátů (penny je ~100× dražší než Vickrey)
    # a předčasně ukončený grid má vyvážené pokrytí buněk
    order = cell_rng(seed, "order").permutation(len(specs))
    with open(a.out, "w") as f:
        for i in order:
            f.write(json.dumps(specs[i], ensure_ascii=False) + "\n")
    print(f"{len(specs)} běhů → {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
