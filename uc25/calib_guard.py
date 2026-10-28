#!/usr/bin/env python3
"""UC25: pojistka kalibrační sady — po vyhodnocení smaže každý obrázek, kde se signál
nezletilosti potká s čímkoli jiným než jisté safe. Do výstupu jen počty, žádné obrázky ani prompty.

  calib_guard.py --combined combined.jsonl --manifest /cold/w2/uc25/calib/manifest.jsonl --out guard.json [--dry-run]

Pravidla:
  1. tier ≠ safe (dospělý prompt) a jakýkoli signál nezletilosti (combine.py `minor` ≠ None,
     tj. WD14 tag / věk child|teen od kteréhokoli hodnotitele)   → smazat
  2. prompt s dítětem a ensemble ≠ jisté safe                    → smazat
Smazané id jdou do guard.json (jen id + pravidlo), aby se nevygenerovaly znovu.
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--combined", required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

man = {}
for l in open(a.manifest):
    r = json.loads(l)
    man[r["image"]] = r
comb = {}
for l in open(a.combined):
    r = json.loads(l)
    if r["image"] in man:
        comb[r["image"]] = r

deleted, kept = [], 0
child_detected = child_total = 0
for iid, m in man.items():
    c = comb.get(iid)
    if c is None:
        continue
    rule = None
    if m["tier"] != "safe" and c.get("minor"):
        rule = "minor signal in adult tier"
    if m["child"]:
        child_total += 1
        ages = [lab.get("person.age_group", {}).get("value") for lab in c["labels"].values()]
        if "child" in ages or c.get("minor_signal", 0) >= 0.3:
            child_detected += 1
        rt = c["rating"]
        if rt.get("band") != "safe" or rt.get("unsure"):
            rule = "child prompt not confidently safe"
    if rule:
        deleted.append({"image": iid, "rule": rule})
        if not a.dry_run and os.path.exists(m["path"]):
            os.remove(m["path"])
    else:
        kept += 1

out = {"evaluated": len(comb), "kept": kept, "deleted": len(deleted),
       "deleted_by_rule": {r: sum(1 for d in deleted if d["rule"] == r) for r in {d["rule"] for d in deleted}},
       "child_prompts": child_total, "child_prompts_flagged_child_by_any_evaluator": child_detected,
       "deleted_ids": deleted, "dry_run": a.dry_run}
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "deleted_ids"}))
