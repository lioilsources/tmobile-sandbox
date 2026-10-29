#!/usr/bin/env python3
"""UC19 report: techniques × dialect (checkpoint) table from translations + jobs + scores.

  report.py --dir /cold/w2/uc19 --md /cold/w2/uc19/report_UC19.md --json ~/tmobile-sandbox/bench/results/UC19_dialects.json

Text side (no GPU needed): CLIP-L overflow rate for SDXL (raw LLM / baseline vs after post-processing), T5 length
for FLUX, dropped (out-of-vocabulary) tags, removed quality/unsafe tags, Pony prompts without the score_ prefix,
latency. Image side: PickScore + SigLIP vs the independent English gist, WD14 recall of must_tags, negation
violations (must_not detected), VLM match 1–5; t2 − t1 paired per prompt with a 90 % bootstrap CI.
"""
import argparse, json, os, random, statistics, sys
from collections import defaultdict

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", "uc18"))
from common import read_jsonl  # noqa: E402


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 4) if xs else None


def share(xs):
    xs = list(xs)
    return round(sum(map(bool, xs)) / len(xs), 3) if xs else None


def boot_ci(diffs, B=1000, seed=0):
    if len(diffs) < 5:
        return None
    rng = random.Random(seed)
    ms = sorted(statistics.mean(rng.choice(diffs) for _ in diffs) for _ in range(B))
    return [round(ms[int(0.05 * B)], 4), round(ms[int(0.95 * B)], 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    d = a.dir
    tr = {}
    for r in read_jsonl(os.path.join(d, "translations.jsonl")):
        if r["key"] not in tr or not r.get("error"):
            tr[r["key"]] = r
    text = {}
    for (t, dia) in sorted({(r["technique"], r["dialect"]) for r in tr.values()}):
        rs = [r for r in tr.values() if r["technique"] == t and r["dialect"] == dia]
        ok = [r for r in rs if not r.get("error")]
        text[f"{t}/{dia}"] = {
            "n": len(rs), "errors": len(rs) - len(ok),
            "latency_p50_ms": statistics.median([r["ms"] for r in ok]) if ok else None,
            "clip_overflow_raw": share(r.get("clip_overflow_raw") for r in ok if r.get("clip_tokens_raw") is not None),
            "clip_tokens_median": statistics.median([r["clip_tokens"] for r in ok if r.get("clip_tokens") is not None] or [0]) or None,
            "truncated": share(r.get("truncated") for r in ok) if dia == "sdxl" else None,
            "t5_tokens_median": statistics.median([r["t5_tokens"] for r in ok if r.get("t5_tokens")] or [0]) or None,
            "taglike": share(r.get("taglike") for r in ok) if dia == "flux" else None,
            "dropped_per_prompt": mean([len(r.get("dropped", [])) for r in ok]) if dia in ("pony", "illustrious") else None,
            "oov_per_prompt": mean([len(r.get("oov", [])) for r in ok]) if dia in ("pony", "illustrious") else None,
            "quality_tags_removed": sum(len(r.get("quality_removed", [])) for r in ok),
            "unsafe_removed_or_flagged": sum(bool(r.get("unsafe_removed") or r.get("unsafe")) for r in ok),
        }
    jobs = read_jsonl(os.path.join(d, "jobs.jsonl"))
    pony_no_score = {}
    for v in sorted({j["variant"] for j in jobs}):
        pj = [j for j in jobs if j["variant"] == v and j["ckpt"] == "pony"]
        if pj:
            pony_no_score[v] = share(not j["prompt"].startswith("score_9") for j in pj)

    man = {}
    for r in read_jsonl(os.path.join(d, "manifest.jsonl")):
        if not r.get("error") and r.get("path"):
            man[r["id"]] = r
    sc = defaultdict(dict)
    for f in ("pick", "siglip", "wd14", "ir", "vlm_single"):
        for s in read_jsonl(os.path.join(d, "scores", f + ".jsonl")):
            if not s.get("error") and s["id"] in man:
                sc[s["id"]].update({k: v for k, v in s.items() if k in ("pickscore", "siglip_cos", "wd14_recall", "imagereward",
                                                                        "must_not_hits", "nsfw_flag", "match", "quality")})
    cats = {r["cid"]: r["category"] for r in read_jsonl(os.path.join(H, "prompts", "cz_prompts.jsonl"))}
    img = defaultdict(list)
    for i, r in man.items():
        img[(r["variant"], r["ckpt"])].append({**r, **sc.get(i, {}), "category": cats.get(r["pid"], "")})
    table = {}
    for (v, ck), rs in sorted(img.items()):
        ok = [r for r in rs if not r.get("nsfw_flag")]
        neg = [r for r in ok if r["category"] == "negation" and "must_not_hits" in r]
        table[f"{v}/{ck}"] = {"images": len(rs), "nsfw_flagged": len(rs) - len(ok),
                              "pickscore": mean(r.get("pickscore") for r in ok), "siglip_cos": mean(r.get("siglip_cos") for r in ok),
                              "wd14_recall": mean(r.get("wd14_recall") for r in ok), "imagereward": mean(r.get("imagereward") for r in ok),
                              "vlm_match": mean(r.get("match") for r in ok), "vlm_quality": mean(r.get("quality") for r in ok),
                              "negation_violation": share(bool(r["must_not_hits"]) for r in neg),
                              "match_by_category": {c: mean(r.get("match") for r in ok if r["category"] == c)
                                                    for c in sorted({r["category"] for r in ok})}}
    paired = {}
    for ck in sorted({ck for _, ck in img}):
        by = {v: {r["pid"]: r for r in img.get((v, ck), [])} for v in ("t1", "t1p", "t2")}
        for m in ("pickscore", "match", "wd14_recall"):
            for base in ("t1", "t1p"):
                diffs = [by["t2"][p][m] - by[base][p][m] for p in by["t2"] if p in by[base]
                         and by["t2"][p].get(m) is not None and by[base][p].get(m) is not None]
                if diffs:
                    paired[f"{ck}:{m}:t2-{base}"] = {"n": len(diffs), "mean": round(statistics.mean(diffs), 4),
                                                     "ci90": boot_ci(diffs), "t2_better": share(x > 0 for x in diffs)}
    out = {"uc": 19, "text": text, "pony_without_score_prefix": pony_no_score, "images": table, "paired": paired}
    L = ["# UC19 — z českého promptu dialekt pro Pony / Illustrious / SDXL / FLUX (automatický návrh)", "",
         "## Textová strana (bez GPU)", "", "| technika/dialekt | n | chyby | p50 ms | CLIP overflow (raw) | CLIP tok. medián | "
         "zkráceno | T5 tok. medián | tag-list (FLUX) | vyřazené / mimo WD14 tagy na prompt | unsafe |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, x in text.items():
        L.append(f"| {k} | {x['n']} | {x['errors']} | {x['latency_p50_ms']} | {x['clip_overflow_raw']} | {x['clip_tokens_median']} | "
                 f"{x['truncated']} | {x['t5_tokens_median']} | {x['taglike']} | {x['dropped_per_prompt']} / {x['oov_per_prompt']} | {x['unsafe_removed_or_flagged']} |")
    L += ["", f"Pony prompty bez `score_` prefixu: {pony_no_score}", "", "## Obrázky", "",
          "| technika/checkpoint | obr. | PickScore | SigLIP cos | WD14 recall | VLM shoda 1–5 | VLM kvalita | negace porušena | NSFW gate |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, x in table.items():
        tag = " (FLUX.1-dev, jen interní měření)" if "flux_dev" in k else ""
        L.append(f"| {k}{tag} | {x['images']} | {x['pickscore']} | {x['siglip_cos']} | {x['wd14_recall']} | {x['vlm_match']} | "
                 f"{x['vlm_quality']} | {x['negation_violation']} | {x['nsfw_flagged']} |")
    L += ["", "## t2 − t1 párově (stejný prompt, stejný seed)", "", "| checkpoint:metrika:srovnání | n | Δ průměr | 90% CI | t2 lepší |",
          "|---|---|---|---|---|"]
    for k, x in paired.items():
        L.append(f"| {k} | {x['n']} | {x['mean']:+.4f} | {x['ci90']} | {x['t2_better']} |")
    L += ["", "VLM shoda podle kategorie promptu (idiom / reálie / negace / diakritika / běžné):", ""]
    for k, x in table.items():
        L.append(f"- {k}: {x['match_by_category']}")
    open(a.md, "w").write("\n".join(L) + "\n")
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=False)
    print(f"report → {a.md}, {a.json}")


if __name__ == "__main__":
    main()
