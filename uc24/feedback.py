"""UC24 krok 4 — feedback smyčka: generuj → brána → (oprava | přepis promptu LLM) → znovu, max 3 iterace.
Loguje se celá cesta (prompt, verdikt, důvody, změny přepisu, časy) do JSONL.

  UC24_GEN=sim     python feedback.py --tasks storyteller --out /cold/uc24/loop_sim.jsonl      # suchý běh (CPU + LLM)
  UC24_GEN=aistack python feedback.py --tasks storyteller --save /cold/uc24/loop_wav ...        # MOSS přes AiStack audio
  UC24_GEN=comfy-ace python feedback.py --tasks storyteller-music ...                            # ACE-Step v ComfyUI (profil D)

Úlohy "storyteller" = katalog zvuků Storytelleru (rag/audio/catalog.json: 80 tvorů + 93 akcí, prompt + styl
jako v internal/audiogen/cmd/render-audio). Souhrn → stdout (+ --summary JSON).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen import gen  # noqa: E402
from rewrite import rewrite  # noqa: E402
from sfxgate.gate import gate  # noqa: E402
from sfxgate.repair import repair  # noqa: E402

CATALOG = os.environ.get("UC24_CATALOG", "/cold/uc24/storyteller_catalog.json")


def storyteller_tasks(which: str = "sfx") -> list[dict]:
    c = json.load(open(CATALOG))
    out = []
    if which in ("sfx", "all"):
        for grp in ("creatures", "actions"):
            for k, e in sorted(c["sfx"][grp].items()):
                out.append({"id": f"{grp[:-1]}-{k}", "kind": "sfx", "prompt": e["prompt"] + ", " + c["sfx"]["style"],
                            "dur": c["sfx"]["duration_s"]})
    if which in ("music", "all"):
        for env, e in sorted(c["music"]["environments"].items()):
            for mood, m in sorted(c["music"]["moods"].items()):
                out.append({"id": f"music-{env}-{mood}", "kind": "music",
                            "prompt": e["prompt"] + ", " + m + ", " + c["music"]["style"], "dur": c["music"]["duration_s"]})
    return out


def run_task(t: dict, max_iter: int = 3, clap=None, clap_min: float = 0.15, save: str | None = None, use_repair=True,
             use_rewrite=True) -> dict:
    prompt, steps, accepted, best = t["prompt"], [], False, None
    seed0 = zlib.crc32(("sound\x1f" + t["id"]).encode())  # jako seed(id) v render-audio (jiný hash, stejná role)
    for it in range(max_iter):
        x, sr, gm = gen(prompt, t["kind"], seed=(seed0 + it) % 2**32, dur=t.get("dur"))
        g = gate(x, sr, t["kind"])
        if clap is not None:
            c = float(clap(x, sr, t["prompt"]))  # shoda s PŮVODNÍM zadáním, ne s přepsaným promptem
            g["features"]["clap"] = c
            if c < clap_min:
                g["reasons"].append({"code": "clap_mismatch", "metric": "clap", "value": round(c, 4), "threshold": clap_min,
                                     "stage": "semantic", "severity": "reject", "message": f"CLAP {c:.2f}: neodpovídá zadání"})
                g["verdict"], g["pass"] = "reject", False
        step = {"iter": it + 1, "prompt": prompt, "verdict": g["verdict"], "score": g["score"],
                "reasons": [{k: r[k] for k in ("code", "value", "threshold", "stage", "message")} for r in g["reasons"]],
                "gate_ms": g["ms"], **{k: v for k, v in gm.items() if k != "file"}}
        final = x
        if g["pass"]:
            accepted = True
        elif g["verdict"] == "repair" and use_repair:
            rp = repair(x, sr, t["kind"], t["prompt"], clap, first=g)
            step.update(repaired=rp["accepted"], repair_steps=rp["steps"], clap_before=rp["clap_before"], clap_after=rp["clap_after"],
                        repair_reasons=[r["code"] for r in rp["gate"]["reasons"]])
            if rp["accepted"]:
                accepted, final = True, rp["audio"]
        if best is None or g["score"] < best[0]:
            best = (g["score"], final, sr)
        if accepted:
            best = (0.0, final, sr)
            steps.append(step)
            break
        if it < max_iter - 1 and use_rewrite:
            rw = rewrite(prompt, g["reasons"])
            step.update(rewrite_changes=rw["changes"], rewrite_ms=rw["ms"], rewrite_model=rw["model"], rewrite_tokens=rw["tokens"])
            prompt = rw["prompt"]
        steps.append(step)
    if save:
        os.makedirs(save, exist_ok=True)
        sf.write(os.path.join(save, t["id"] + ".wav"), best[1], best[2], subtype="PCM_16")
    return {"id": t["id"], "kind": t["kind"], "original_prompt": t["prompt"], "final_prompt": prompt, "accepted": accepted,
            "iterations": len(steps), "first_try_pass": steps[0]["verdict"] == "pass",
            "first_try_accepted": accepted and len(steps) == 1, "steps": steps}


def summarize(res: list[dict]) -> dict:
    n = len(res)
    it = [r["iterations"] for r in res]
    codes: dict[str, int] = {}
    for r in res:
        for c in {x["code"] for x in r["steps"][0]["reasons"]}:
            codes[c] = codes.get(c, 0) + 1
    rw_ms = [s["rewrite_ms"] for r in res for s in r["steps"] if "rewrite_ms" in s]
    return {"n": n, "first_try_pass": round(sum(r["first_try_pass"] for r in res) / n, 3),
            "first_try_accepted_incl_repair": round(sum(r["first_try_accepted"] for r in res) / n, 3),
            "accepted_within_3": round(sum(r["accepted"] for r in res) / n, 3),
            "mean_iterations": round(float(np.mean(it)), 2), "generations_total": int(sum(it)),
            "rewrites": len(rw_ms), "rewrite_ms_median": int(np.median(rw_ms)) if rw_ms else None,
            "first_try_reason_counts": dict(sorted(codes.items(), key=lambda kv: -kv[1]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="storyteller", choices=["storyteller", "storyteller-music", "storyteller-all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-iter", type=int, default=3)
    ap.add_argument("--clap", default=None, help="HF id CLAP (volitelné; na CPU ~3 s/klip)")
    ap.add_argument("--save", default=None)
    ap.add_argument("--no-rewrite", action="store_true", help="kontrola: jen nový seed, stejný prompt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", default=None)
    a = ap.parse_args()
    tasks = storyteller_tasks({"storyteller": "sfx", "storyteller-music": "music", "storyteller-all": "all"}[a.tasks])
    tasks = tasks[: a.limit] if a.limit else tasks
    clap = None
    if a.clap:
        from sfxgate.clap import Clap
        clap = Clap(a.clap, "cpu").score
    res, t0 = [], time.time()
    with open(a.out, "w") as f:
        for t in tasks:
            r = run_task(t, a.max_iter, clap, save=a.save, use_rewrite=not a.no_rewrite)
            res.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
    s = summarize(res) | {"backend": os.environ.get("UC24_GEN", "sim"), "rewrite": not a.no_rewrite,
                          "wall_s": round(time.time() - t0, 1)}
    if a.summary:
        json.dump(s, open(a.summary, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(s, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
