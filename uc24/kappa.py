"""UC24 κ-gate — soudce smí do produkce jen s prokázanou shodou s Olinem.

Logika 1:1 podle FineTuneGallery server/kappa.go (Cohen κ, 2×2 tabulka human × judge nad ±1,
asymptotická SE, rozhoduje DOLNÍ mez 95% intervalu; prahy jsou konstanty, ne env):
  n ≥ 40 párů, ≥ 8 lidských "down", κ ≥ 0.60 a dolní mez ≥ 0.40 → pass;
  jinak fail; pod minimy nebo κ nedefinované (pe = 1) → uncalibrated.

Konvence: +1 = "up" (v pořádku), -1 = "down" (vada). Otázky blind listu (rating_sheet.py):
  pleasant   nepříjemné na poslech?   ano → -1
  matches    odpovídá popisu?          ne  → -1
  defect     jaká vada (none/hiss/rumble/clipping/clicks/other) — popisné, κ per vada = jedna vs ostatní

  python kappa.py --ratings ratings.csv --key key.json --judges /cold/uc24/judge_votes.jsonl \
                  --out ../bench/results/UC24_kappa.json
judge_votes.jsonl: {"path": <zdrojová cesta> | "clip": <blind id>, "judge": "sfxgate@0.1.0", "question": "pleasant", "vote": ±1}
(cesty se přes key.json převedou na blind id; votes vyrábí judge_votes.py)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass

GATE_MIN_PAIRS = 40
GATE_MIN_HUMAN_DOWN = 8
GATE_MIN_KAPPA = 0.60
GATE_MIN_KAPPA_LOWER = 0.40
Z = 1.959963984540054


@dataclass
class Agreement:
    n: int = 0
    upUp: int = 0
    upDown: int = 0
    downUp: int = 0
    downDown: int = 0
    po: float | None = None
    pe: float | None = None
    kappa: float | None = None
    lower: float | None = None
    upper: float | None = None
    downAgree: float | None = None
    gate: str = "uncalibrated"

    def add(self, human: int, judge: int) -> None:
        key = {(1, 1): "upUp", (1, -1): "upDown", (-1, 1): "downUp", (-1, -1): "downDown"}.get((human, judge))
        if key is None:
            return  # jiné hodnoty než ±1 se nepočítají (jako v kappa.go)
        setattr(self, key, getattr(self, key) + 1)
        self.n += 1

    def human_down(self) -> int:
        return self.downUp + self.downDown

    def finish(self) -> "Agreement":
        self.gate = "uncalibrated"
        if self.n == 0:
            return self
        n = float(self.n)
        po = (self.upUp + self.downDown) / n
        hu = (self.upUp + self.upDown) / n
        ju = (self.upUp + self.downUp) / n
        pe = hu * ju + (1 - hu) * (1 - ju)
        self.po, self.pe = po, pe
        if (hd := self.human_down()) > 0:
            self.downAgree = self.downDown / hd
        if pe >= 1:
            return self  # oba použili jedinou třídu: κ = 0/0, nedefinované
        k = (po - pe) / (1 - pe)
        se = math.sqrt(po * (1 - po) / n) / (1 - pe)
        self.kappa, self.lower, self.upper = k, max(-1.0, k - Z * se), min(1.0, k + Z * se)
        self.gate = gate_state(self)
        return self


def gate_state(a: Agreement) -> str:
    if a.n < GATE_MIN_PAIRS or a.human_down() < GATE_MIN_HUMAN_DOWN or a.kappa is None:
        return "uncalibrated"
    if a.kappa < GATE_MIN_KAPPA or a.lower < GATE_MIN_KAPPA_LOWER:
        return "fail"
    return "pass"


def table(uu, ud, du, dd) -> Agreement:
    a = Agreement()
    for h, j, c in [(1, 1, uu), (1, -1, ud), (-1, 1, du), (-1, -1, dd)]:
        for _ in range(c):
            a.add(h, j)
    return a.finish()


def human_votes(ratings_csv: str) -> dict:
    """{(clip, question): ±1} z vyplněného blind listu."""
    yes = {"ano", "a", "y", "yes", "1", "true"}
    no = {"ne", "n", "no", "0", "false"}
    out = {}
    for r in csv.DictReader(open(ratings_csv, encoding="utf-8")):
        c = r["clip"]
        u = (r.get("nepříjemné") or "").strip().lower()
        m = (r.get("odpovídá") or "").strip().lower()
        d = (r.get("vada") or "").strip().lower()
        if u in yes | no:
            out[(c, "pleasant")] = -1 if u in yes else 1
        if m in yes | no:
            out[(c, "matches")] = 1 if m in yes else -1
        if d:
            for kind in ("hiss", "rumble", "clipping", "clicks"):
                out[(c, f"defect:{kind}")] = -1 if d == kind else 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--judges", required=True)
    ap.add_argument("--key", default=None, help="key.json z rating_sheet.py (blind id → zdroj)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    hv = human_votes(a.ratings)
    by_path = {v["path"]: k for k, v in json.load(open(a.key)).items()} if a.key else {}
    ag = defaultdict(Agreement)
    for line in open(a.judges):
        j = json.loads(line)
        clip = j.get("clip") or by_path.get(j.get("path"))
        h = hv.get((clip, j["question"]))
        if h is not None:
            ag[(j["judge"], j["question"])].add(h, int(j["vote"]))
    rows = [{"judge": jd, "question": q, **asdict(x.finish())} for (jd, q), x in sorted(ag.items())]
    res = {"uc": "UC24", "what": "κ-gate soudců vůči Olinovým blind hodnocením (pravidla = FinetuneGallery kappa.go)",
           "thresholds": {"min_pairs": GATE_MIN_PAIRS, "min_human_down": GATE_MIN_HUMAN_DOWN,
                          "min_kappa": GATE_MIN_KAPPA, "min_kappa_lower": GATE_MIN_KAPPA_LOWER},
           "n_human_votes": len(hv), "table": rows}
    js = json.dumps(res, indent=1, ensure_ascii=False)
    if a.out:
        open(a.out, "w").write(js)
    print(js)


if __name__ == "__main__":
    main()
