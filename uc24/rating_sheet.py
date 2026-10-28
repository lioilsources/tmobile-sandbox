"""UC24 — blind hodnoticí list pro Olina (150 klipů, 3 otázky) → vstup pro kappa.py.

  python rating_sheet.py --manifest <clips.jsonl> --gate /cold/uc24/gate_dump.jsonl --n 150 --out /cold/uc24/rating
  → <out>/audio/r001.wav … (zamíchané, přejmenované — soubor neprozradí zdroj ani verdikt)
    <out>/ratings.csv (vyplnit: nepříjemné ano/ne, odpovídá ano/ne, vada none|hiss|rumble|clipping|clicks|other)
    <out>/index.html  (přehrávač; poslouchat NA TELEFONU, ne ve sluchátkách)
    <out>/key.json    (blind id → zdroj; NEOTVÍRAT před hodnocením)

Výběr je stratifikovaný podle verdiktu sfxgate (třetina pass / repair / reject), aby měl κ dost
"down" případů (kappa.go chce ≥ 8). Pro skutečnou kalibraci musí manifest obsahovat GENEROVANÉ klipy
(Stable Audio / ACE-Step / MOSS), ne syntetické vady — ty slouží jen k validaci příznaků.
STATUS: čeká na Olinova hodnocení.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import random
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="jsonl s path, prompt, kind")
    ap.add_argument("--gate", default=None, help="jsonl z eval_gate.py --dump (stratifikace podle verdiktu)")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, seen = [], set()
    for line in open(a.manifest):
        r = json.loads(line)
        if r["path"] in seen or r.get("defect") == "mismatch":
            continue
        seen.add(r["path"])
        rows.append(r)
    verdict = {}
    if a.gate:
        for line in open(a.gate):
            g = json.loads(line)
            verdict[g["id"]] = g["verdict"]
    rnd = random.Random(a.seed)
    strata: dict[str, list] = {}
    for r in rows:
        strata.setdefault(verdict.get(r.get("id"), r.get("verdict", "all")), []).append(r)
    pick = []
    per = a.n // max(1, len(strata))
    for k in sorted(strata):
        rnd.shuffle(strata[k])
        pick += strata[k][:per]
    rest = [r for k in sorted(strata) for r in strata[k][per:]]
    rnd.shuffle(rest)
    pick += rest[: a.n - len(pick)]
    rnd.shuffle(pick)
    os.makedirs(os.path.join(a.out, "audio"), exist_ok=True)
    key, cards = {}, []
    with open(os.path.join(a.out, "ratings.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip", "popis", "nepříjemné", "odpovídá", "vada", "poznámka"])
        for i, r in enumerate(pick, 1):
            cid = f"r{i:03d}"
            ext = os.path.splitext(r["path"])[1]
            shutil.copyfile(r["path"], os.path.join(a.out, "audio", cid + ext))
            key[cid] = {"path": r["path"], "prompt": r["prompt"], "kind": r.get("kind"), "id": r.get("id"),
                        "stratum": verdict.get(r.get("id"), r.get("verdict", "all"))}
            w.writerow([cid, r["prompt"], "", "", "", ""])
            cards.append(f'<li><b>{cid}</b> — {html.escape(r["prompt"])}<br><audio controls preload="none" src="audio/{cid}{ext}"></audio></li>')
    json.dump(key, open(os.path.join(a.out, "key.json"), "w"), indent=1, ensure_ascii=False)
    open(os.path.join(a.out, "index.html"), "w", encoding="utf-8").write(
        "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>"
        "<title>UC24 blind poslech</title><style>body{font:16px system-ui;margin:16px;max-width:720px}li{margin:12px 0}</style>"
        "<h1>UC24 blind poslech</h1><p>Poslouchej na telefonu. Do ratings.csv: nepříjemné ano/ne, odpovídá popisu ano/ne, "
        "vada none|hiss|rumble|clipping|clicks|other.</p><ol>" + "".join(cards) + "</ol>")
    print(f"{len(pick)} klipů → {a.out} (strata: { {k: len(v) for k, v in strata.items()} })")


if __name__ == "__main__":
    main()
