#!/usr/bin/env python3
"""UC19: translations.jsonl → gen_matrix jobs (one image per prompt × technique × target checkpoint).

  build_jobs.py --translations /cold/w2/uc19/translations.jsonl --out /cold/w2/uc19/jobs.jsonl \
                [--techniques t1,t2] [--all-targets] [--t1-prefix]

t1  : raw English translation, identical for every model, no quality/rating prefix (the true baseline).
t1p : (with --t1-prefix) the same translation + each checkpoint's quality/rating prefix/suffix — separates
      "the dialect helped" from "the score_ prefix helped" on Pony/Illustrious.
t2  : dialect body + the checkpoint's own prefix/suffix (models.py); SDXL uses the LLM negative + SFW terms,
      the others the checkpoint's default negative (always containing the SFW terms).
A final prompt that fails the SFW text check is NOT queued (logged with the reason).
"""
import argparse, json, os, sys

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", "uc18"))
from common import read_jsonl, sfw_hits  # noqa: E402
from models import CKPTS, DIALECT_TARGETS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--translations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--techniques", default="t1,t2")
    ap.add_argument("--all-targets", action="store_true")
    ap.add_argument("--t1-prefix", action="store_true")
    a = ap.parse_args()
    last = {}
    for r in read_jsonl(a.translations):
        if not r.get("error"):
            last[r["key"]] = r
    techs = a.techniques.split(",")
    jobs, skipped = [], []
    targets = {d: (cks if a.all_targets else cks[:1]) for d, cks in DIALECT_TARGETS.items()}
    t1 = {r["cid"]: r for r in last.values() if r["technique"] == "t1"}
    for r in last.values():
        if r["technique"] == "t1" or r["technique"] not in techs:
            continue
        for ck in targets[r["dialect"]]:
            c = CKPTS[ck]
            neg = r.get("negative") if r["dialect"] == "sdxl" else c["negative"]
            jobs.append({"pid": r["cid"], "ckpt": ck, "variant": r["technique"], "style": r["style"],
                         "prompt": f"{c['prefix']}{r['body']}{c['suffix']}", "negative": neg,
                         "meta": {"technique": r["technique"], "dialect": r["dialect"], "category": r["category"],
                                  "version": r["version"]}})
    if "t1" in techs:
        for cid, r in t1.items():
            for d, cks in targets.items():
                for ck in cks:
                    c = CKPTS[ck]
                    base = {"pid": cid, "ckpt": ck, "style": r["style"], "negative": c["negative"],
                            "meta": {"dialect": d, "category": r["category"], "version": r["version"]}}
                    jobs.append({**base, "variant": "t1", "prompt": r["body"], "meta": {**base["meta"], "technique": "t1"}})
                    if a.t1_prefix and (c["prefix"] or c["suffix"]):
                        jobs.append({**base, "variant": "t1p", "prompt": f"{c['prefix']}{r['body']}{c['suffix']}",
                                     "meta": {**base["meta"], "technique": "t1p"}})
    ok = []
    for j in jobs:
        h = sfw_hits(j["prompt"])
        (skipped if h else ok).append({**j, "sfw_hits": h} if h else j)
    ok.sort(key=lambda j: (list(CKPTS).index(j["ckpt"]), j["variant"], j["pid"]))
    with open(a.out, "w") as f:
        for j in ok:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    per = {}
    for j in ok:
        per[f"{j['variant']}:{j['ckpt']}"] = per.get(f"{j['variant']}:{j['ckpt']}", 0) + 1
    est = sum(CKPTS[j["ckpt"]]["est_s"] for j in ok) / 3600
    print(json.dumps({"jobs": len(ok), "skipped_sfw": [(s["pid"], s["variant"], s["ckpt"], s["sfw_hits"]) for s in skipped],
                      "per_variant_ckpt": per, "est_gen_hours": round(est, 2)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
