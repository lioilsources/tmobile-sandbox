#!/usr/bin/env python3
"""UC3: positions.jsonl + analysis*.jsonl → kid-level kvízy (quizzes.json) pro Chess4Princesses.

Kvíz = řešení na jeden tah, jediný správný tah, jasný motiv, málo figur. Jednoznačnost:
mat 1 tahem počítá python-chess přesně (právě jeden matující tah); ostatní motivy ověří
Stockfish multipv 2 (náskok nejlepšího tahu ≥ 2 pěšce a nejlepší = řešení).
Obtížnost 1–5 = kvintily skóre z intuice sítě (p_sol), počtu kandidátů a počtu figur.

  select_quizzes.py --pos positions.jsonl --ana 'runs/full/analysis.s*.jsonl' --out quizzes.json --per-lesson 400
"""
import argparse, collections, glob, json, multiprocessing as mp
import chess, chess.engine

# motiv → lekce kurikula (Prompts/01-PRINCESSCHESS_LESSONS.md)
LESSONS = [
    ("m08_back_rank", "Mat na poslední řadě", lambda t, pc: "mateIn1" in t and "backRankMate" in t),
    ("m08_queen", "Mat Princeznou", lambda t, pc: "mateIn1" in t and pc == chess.QUEEN),
    ("m08_mix", "Mat v 1 – mix", lambda t, pc: "mateIn1" in t),
    ("m07_hanging", "Sněz nechráněnou figuru", lambda t, pc: "hangingPiece" in t),
    ("m11_fork", "Vidlička (dva zajíci)", lambda t, pc: "fork" in t),
    ("m11_pin", "Vazba", lambda t, pc: "pin" in t),
]
def unique_mate(fen, sol):
    b = chess.Board(fen); n = 0
    for m in b.legal_moves:
        b.push(m); n += b.is_checkmate(); b.pop()
    return n == 1


def sf_unique(args):
    fen, sol = args
    with chess.engine.SimpleEngine.popen_uci("stockfish") as e:
        e.configure({"Threads": 1, "Hash": 32})
        info = e.analyse(chess.Board(fen), chess.engine.Limit(depth=12), multipv=2)
    if len(info) < 2: return True
    best, second = info[0], info[1]
    if best["pv"][0].uci() != sol: return False
    s1 = best["score"].relative.score(mate_score=10000); s2 = second["score"].relative.score(mate_score=10000)
    return s1 - s2 >= 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", required=True); ap.add_argument("--ana", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max-rating", type=int, default=1100); ap.add_argument("--max-pieces", type=int, default=14)
    ap.add_argument("--per-lesson", type=int, default=400); ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    ana = {}
    for f in glob.glob(a.ana):
        for l in open(f):
            j = json.loads(l); ana[j["id"]] = j
    cand = collections.defaultdict(list); stats = collections.Counter()
    for l in open(a.pos):
        r = json.loads(l)
        if r["rating"] > a.max_rating or r["pieces"] > a.max_pieces or len(r["solution"]) != 1:
            continue
        x = ana.get(r["id"])
        if not x or x["p_sol"] is None: continue
        stats["kandidat"] += 1
        b = chess.Board(r["fen"]); m = chess.Move.from_uci(r["solution"][0])
        pc = b.piece_type_at(m.from_square)
        t = set(r["themes"])
        for lid, name, pred in LESSONS:
            if pred(t, pc):
                cand[lid].append({**r, **{k: x[k] for k in ("p_sol", "rank_sol", "n_cand", "q", "entropy")},
                                  "piece": chess.piece_name(pc)})
                break
    out = []
    with mp.Pool(a.procs) as pool:
        for lid, name, _ in LESSONS:
            c = sorted(cand[lid], key=lambda r: -r["popularity"])
            if lid.startswith("m08"):
                ok = [r for r in c if unique_mate(r["fen"], r["solution"][0])]
            else:
                pre = [r for r in c if r["rank_sol"] == 0][: a.per_lesson * 3]
                flags = pool.map(sf_unique, [(r["fen"], r["solution"][0]) for r in pre], chunksize=8)
                ok = [r for r, f in zip(pre, flags) if f]
            ok = ok[: a.per_lesson]
            stats[f"{lid}_kandidatu"] = len(c); stats[f"{lid}_vybrano"] = len(ok)
            for r in ok:
                # vyšší p_sol a méně kandidátů a figur = snazší
                r["score"] = (1 - r["p_sol"]) * 2 + min(r["n_cand"], 6) / 6 + r["pieces"] / a.max_pieces
                r["lesson"] = lid; r["lesson_name"] = name
            ok.sort(key=lambda r: r["score"])
            for i, r in enumerate(ok):
                r["difficulty"] = 1 + i * 5 // max(len(ok), 1)
            out += ok
    keep = ("id", "fen", "solution", "lesson", "lesson_name", "difficulty", "piece", "rating", "themes", "p_sol", "n_cand", "pieces", "popularity")
    json.dump([{k: r[k] for k in keep} for r in out], open(a.out, "w"), ensure_ascii=False, indent=0)
    print(json.dumps({**stats, "kvizu": len(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
