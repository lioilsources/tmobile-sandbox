#!/usr/bin/env python3
"""UC3: Lichess puzzle DB (CSV.zst) → positions.jsonl.

Lichess FEN je pozice PŘED tahem soupeře; první tah z Moves zahraje soupeř, zbytek je řešení.
Ukládáme pozici, kterou řeší dítě (po prvním tahu), řešení, rating, témata a počet figur.

  prepare.py raw/lichess_db_puzzle.csv.zst --out positions.jsonl [--max-rating 0]
"""
import argparse, csv, io, json
import chess
import zstandard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("--out", required=True)
    ap.add_argument("--max-rating", type=int, default=0, help="0 = všechny puzzle (benchmark), jinak filtr")
    a = ap.parse_args()
    n = kept = 0
    with open(a.src, "rb") as fh, open(a.out, "w") as fo:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        for row in csv.DictReader(text):
            n += 1
            rating = int(row["Rating"])
            if a.max_rating and rating > a.max_rating:
                continue
            moves = row["Moves"].split()
            b = chess.Board(row["FEN"])
            b.push_uci(moves[0])
            fo.write(json.dumps({
                "id": row["PuzzleId"], "fen": b.fen(), "solution": moves[1:], "rating": rating,
                "popularity": int(row["Popularity"]), "plays": int(row["NbPlays"]),
                "themes": row["Themes"].split(), "pieces": len(b.piece_map()),
            }) + "\n")
            kept += 1
    print(json.dumps({"puzzle": n, "pozic": kept}))


if __name__ == "__main__":
    main()
