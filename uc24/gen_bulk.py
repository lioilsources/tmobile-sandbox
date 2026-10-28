"""UC24 krok 4 — hromadné generování pro slovníkovou analýzu (faktoriální návrh nad slovy).

Každý řádek = katalogové zadání Storytelleru + 0–3 náhodné modifikátory ze slovníku MODS (vyvážené,
nezávislé na zadání → regrese slovo → p(vada) je identifikovatelná) → generace → sfxgate → JSONL.

  UC24_GEN=sim     python gen_bulk.py --n 3000 --out /cold/uc24/bulk_sim.jsonl
  UC24_GEN=aistack python gen_bulk.py --n 5000 --out /cold/uc24/bulk_moss.jsonl --save /cold/uc24/bulk_moss_wav
  UC24_GEN=comfy-sao python gen_bulk.py --n 5000 --out /cold/uc24/bulk_sao.jsonl --save ...
Resumable: řádky s již hotovým idx se přeskočí.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import zlib

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feedback import storyteller_tasks  # noqa: E402
from gen import gen  # noqa: E402
from sfxgate.gate import gate  # noqa: E402

MODS = ["deep", "soft", "distant", "loud", "gentle", "crisp", "warm", "dry", "echoing", "cinematic", "realistic",
        "lo-fi", "high quality", "thunderous", "booming", "clean", "subtle", "powerful", "vintage", "close-up",
        "airy", "heavy", "bright", "muffled", "punchy", "magical", "dark", "tiny", "huge", "studio recording"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save", default=None)
    ap.add_argument("--seed", type=int, default=24)
    ap.add_argument("--kind", default=os.environ.get("KIND", "sfx"), choices=["sfx", "music"])
    a = ap.parse_args()
    tasks = storyteller_tasks(a.kind)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(line)["idx"] for line in open(a.out)}
    rnd = random.Random(a.seed)
    plan = []
    for i in range(a.n):
        t = rnd.choice(tasks)
        mods = rnd.sample(MODS, rnd.choice([0, 1, 1, 2, 2, 3]))
        parts = t["prompt"].split(", ")  # modifikátory hned za první segment (zdroj zvuku)
        plan.append((i, t, mods, ", ".join(parts[:1] + mods + parts[1:])))
    if a.save:
        os.makedirs(a.save, exist_ok=True)
    with open(a.out, "a") as f:
        for i, t, mods, prompt in plan:
            if i in done:
                continue
            try:
                x, sr, m = gen(prompt, t["kind"], seed=zlib.crc32(f"{i}".encode()), dur=t["dur"])
            except Exception as e:  # noqa: BLE001
                f.write(json.dumps({"idx": i, "task": t["id"], "prompt": prompt, "error": repr(e)}) + "\n")
                continue
            g = gate(x, sr, t["kind"])
            if a.save:
                sf.write(os.path.join(a.save, f"{i:05d}.wav"), x, sr, subtype="PCM_16")
            f.write(json.dumps({"idx": i, "task": t["id"], "mods": mods, "prompt": prompt, "verdict": g["verdict"],
                                "codes": sorted({r["code"] for r in g["reasons"]}), "score": g["score"],
                                **{k: v for k, v in m.items() if k.startswith("sim_") or k in ("backend", "gen_ms")}}) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
