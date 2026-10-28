"""OpenEvolve evaluátor pro UC24 šablonu promptu.

combined_score = podíl z 50 fixních Storyteller zadání (katalog, seed 24), jejichž klip projde bránou
NAPOPRVÉ (verdikt pass) A prompt zachová obsah zadání (≥ 70 % obsahových slov mimo rizikový slovník;
jinak by evoluce "vyhrála" tichem nebo jiným zvukem). Backend generování = UC24_GEN (sim | aistack | comfy-sao).
Se `sim` je to jen test harnessu — skóre nic neříká o MOSS/Stable Audio.
"""
import importlib.util
import json
import os
import random
import re
import sys
import zlib

UC24 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, UC24)
from feedback import storyteller_tasks  # noqa: E402
from gen import gen  # noqa: E402
from sfxgate.gate import gate  # noqa: E402

N_TASKS = int(os.environ.get("UC24_EVOLVE_N", "50"))
RISKY = {"deep", "thunderous", "rumbling", "booming", "bass", "heavy", "huge", "giant", "dark", "loud", "explosive",
         "blasting", "crash", "smash", "powerful", "roaring", "vintage", "radio", "distant", "misty", "static", "crackling"}
STOP = {"with", "and", "the", "from", "into", "onto", "over", "very", "softly", "gently", "that", "this"}


def tasks():
    t = [x for x in storyteller_tasks("sfx")]
    random.Random(24).shuffle(t)
    out = []
    for x in t[:N_TASKS]:
        subject = x["prompt"].split(", clear, close")[0]  # katalogový prompt bez stylu
        out.append({**x, "subject": subject})
    return out


def content_words(s):
    return {w for w in re.findall(r"[a-z]+", s.lower()) if len(w) > 3 and w not in RISKY and w not in STOP}


def evaluate(program_path):
    spec = importlib.util.spec_from_file_location("tmpl", program_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ok = kept = passed = 0
    words = 0
    rows = []
    for t in tasks():
        try:
            p = str(mod.build(t["subject"], t["kind"]))
        except Exception as e:  # noqa: BLE001
            return {"combined_score": 0.0, "error": repr(e)}
        cw = content_words(t["subject"])
        keep = len(cw & content_words(p)) / max(1, len(cw)) >= 0.7 and len(p.split()) <= 60
        x, sr, _ = gen(p, t["kind"], seed=zlib.crc32(t["id"].encode()), dur=t["dur"])
        g = gate(x, sr, t["kind"])
        passed += g["pass"]
        kept += keep
        ok += g["pass"] and keep
        words += len(p.split())
        rows.append({"id": t["id"], "prompt": p, "verdict": g["verdict"], "keep": keep})
    n = len(rows)
    return {"combined_score": ok / n, "first_try_pass": passed / n, "subject_kept": kept / n, "mean_words": words / n,
            "sample": json.dumps(rows[:5])}
