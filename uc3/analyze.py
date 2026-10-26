#!/usr/bin/env python3
"""UC3: dávkové ohodnocení pozic sítí lc0 (Python bindings, CUDA) → analysis.jsonl.

Na pozici: q/d/m sítě, pravděpodobnost prvního tahu řešení (p_sol), pořadí řešení v policy,
počet kandidátů s p > 5 % (n_cand) a entropie policy. Málo kandidátů + vysoké p_sol = snadné.
Shardování (--shard i/n) pouští víc procesů nad jednou GPU, aby se CPU kódování překrylo s výpočtem.

  analyze.py --inp positions.jsonl --out runs/x/analysis.s0.jsonl --net nets/BT4….pb.gz --batch 1024 --shard 0/2
"""
import argparse, json, math, time
import backends as lc

CASTLE = {"e1g1": "e1h1", "e1c1": "e1a1", "e8g8": "e8h8", "e8c8": "e8a8"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True); ap.add_argument("--net", required=True)
    ap.add_argument("--backend", default="cuda-fp16"); ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))
    be = lc.Backend(weights=lc.Weights(a.net), backend=a.backend)
    t0 = time.time(); n = 0; t_gpu = 0.0
    with open(a.inp) as fi, open(a.out, "w") as fo:
        def flush(buf):
            nonlocal t_gpu
            states = [lc.GameState(fen=r["fen"]) for r in buf]
            inputs = [s.as_input(be) for s in states]
            tg = time.time(); outs = be.evaluate(*inputs); t_gpu += time.time() - tg
            for r, s, o in zip(buf, states, outs):
                moves = s.moves(); p = o.p_softmax(*s.policy_indices())
                sol = r["solution"][0]; sol = sol if sol in moves else CASTLE.get(sol, sol)
                order = sorted(range(len(moves)), key=lambda i: -p[i])
                rank = next((k for k, i in enumerate(order) if moves[i] == sol), -1)
                fo.write(json.dumps({
                    "id": r["id"], "q": round(o.q(), 4), "d": round(o.d(), 4), "m": round(o.m(), 1),
                    "p_sol": round(p[moves.index(sol)], 4) if sol in moves else None, "rank_sol": rank,
                    "n_cand": sum(x > 0.05 for x in p), "n_legal": len(moves),
                    "top": moves[order[0]], "p_top": round(p[order[0]], 4),
                    "entropy": round(-sum(x * math.log(x) for x in p if x > 0), 3)}) + "\n")
        buf = []
        for i, line in enumerate(fi):
            if i % sn != si: continue
            buf.append(json.loads(line))
            if len(buf) == a.batch:
                flush(buf); n += len(buf); buf = []
                if n % (a.batch * 50) == 0:
                    el = time.time() - t0
                    print(f"{n} pozic  {n / el:.0f}/s  gpu {100 * t_gpu / el:.0f} % času", flush=True)
            if a.limit and n + len(buf) >= a.limit: break
        if buf: flush(buf); n += len(buf)
    el = time.time() - t0
    print(json.dumps({"shard": a.shard, "pozic": n, "sekund": round(el, 1), "pozic_za_s": round(n / el), "gpu_cas_%": round(100 * t_gpu / el)}))


if __name__ == "__main__":
    main()
