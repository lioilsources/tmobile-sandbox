#!/usr/bin/env python3
"""UC18+19 end-to-end on 50 held-out Czech prompts: CZ prompt → router picks a checkpoint → t2 dialect → image.

  e2e.py jobs   --dir /cold/w2/uc19e2e --router /cold/w2/uc18/router.json      # after translate.py (t1 + t2)
  e2e.py report --dir /cold/w2/uc19e2e --md …/report_UC19_e2e.md --json ~/tmobile-sandbox/bench/results/UC19_e2e.json

Arms (same seed per prompt, so pairs differ only by the pipeline):
  naive  = juggernaut + t1 (plain CZ→EN translation)
  fixed  = juggernaut + t2/sdxl (dialect, no router) — juggernaut is the router's style-free favourite (p_best 0.79)
  routed = router top-1 among commercially usable checkpoints (FLUX.1-dev excluded) + t2 in that checkpoint's dialect
Routing embeds the t1 English text with SigLIP 2 (CPU) and uses the prompt's style; the CZ prompts are not in UC18.
"""
import argparse, importlib.util, json, os, statistics, sys
from collections import Counter, defaultdict

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", "uc18"))
from common import limit_threads, read_jsonl, sfw_hits  # noqa: E402
from models import CKPTS  # noqa: E402

# uc18/ also has a report.py (earlier on sys.path) → load UC19's helpers by file path
_spec = importlib.util.spec_from_file_location("uc19_report", os.path.join(H, "report.py"))
_r = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_r)
boot_ci, mean, share = _r.boot_ci, _r.mean, _r.share

limit_threads(4)
PROMPTS = os.path.join(H, "prompts", "e2e_cz50.jsonl")
FIXED = "juggernaut"
METRICS = ("pickscore", "siglip_cos", "wd14_recall", "imagereward", "match", "quality")


def embed(texts):
    import torch
    from transformers import AutoModel, AutoProcessor
    repo = "google/siglip2-so400m-patch16-512"
    proc, m = AutoProcessor.from_pretrained(repo), AutoModel.from_pretrained(repo).eval()
    with torch.inference_mode():
        t = proc(text=texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        f = m.get_text_features(**t)
        return (f if torch.is_tensor(f) else f.pooler_output).float().numpy()


def translations(d):
    last = {}
    for r in read_jsonl(os.path.join(d, "translations.jsonl")):
        if r["key"] not in last or not r.get("error"):
            last[r["key"]] = r
    return {k: r for k, r in last.items() if not r.get("error")}


def cmd_jobs(a):
    import numpy as np
    from route import route
    R = json.load(open(a.router))
    tr = translations(a.dir)
    prompts = read_jsonl(PROMPTS)
    t1 = {p["cid"]: tr.get(f"{p['cid']}/t1/*") for p in prompts}
    have = [p for p in prompts if t1[p["cid"]]]
    E = embed([t1[p["cid"]]["body"] for p in have])
    allowed = {m for m, c in CKPTS.items() if not c.get("licence_restricted")}
    jobs, routes, missing = [], {}, []

    def job(p, ck, variant, body, neg):
        c = CKPTS[ck]
        return {"pid": p["cid"], "ckpt": ck, "variant": variant, "style": p["style"],
                "prompt": f"{c['prefix']}{body}{c['suffix']}" if variant != "naive" else body, "negative": neg,
                "meta": {"category": p["category"], "dialect": c["dialect"]}}

    for p, e in zip(have, E):
        ranked = [x for x in route(R, np.asarray(e, dtype=np.float32), p["style"]) if x["model"] in allowed]
        ck = ranked[0]["model"]
        routes[p["cid"]] = {"model": ck, "score": ranked[0]["score"], "p_best": ranked[0]["p_best"],
                            "runner_up": ranked[1]["model"] if len(ranked) > 1 else None}
        sd = tr.get(f"{p['cid']}/t2/sdxl")
        rt = tr.get(f"{p['cid']}/t2/{CKPTS[ck]['dialect']}")
        if not (sd and rt):
            missing.append(p["cid"])
            continue
        jobs.append(job(p, FIXED, "naive", t1[p["cid"]]["body"], CKPTS[FIXED]["negative"]))
        jobs.append(job(p, FIXED, "fixed", sd["body"], sd.get("negative") or CKPTS[FIXED]["negative"]))
        neg = rt.get("negative") if CKPTS[ck]["dialect"] == "sdxl" else CKPTS[ck]["negative"]
        jobs.append(job(p, ck, "routed", rt["body"], neg or CKPTS[ck]["negative"]))
    ok = [j for j in jobs if not sfw_hits(j["prompt"])]
    ok.sort(key=lambda j: (list(CKPTS).index(j["ckpt"]), j["variant"], j["pid"]))
    with open(os.path.join(a.dir, "jobs.jsonl"), "w") as f:
        for j in ok:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    json.dump(routes, open(os.path.join(a.dir, "routes.json"), "w"), indent=1)
    print(json.dumps({"jobs": len(ok), "skipped_sfw": len(jobs) - len(ok), "no_translation": missing,
                      "routed_to": Counter(r["model"] for r in routes.values()),
                      "est_gen_min": round(sum(CKPTS[j["ckpt"]]["est_s"] for j in ok) / 60, 1)}, indent=1))
    return 0


def cmd_report(a):
    d = a.dir
    man = {r["id"]: r for r in read_jsonl(os.path.join(d, "manifest.jsonl")) if not r.get("error") and r.get("path")}
    sc = defaultdict(dict)
    for f in ("pick", "siglip", "wd14", "ir", "vlm_single"):
        for s in read_jsonl(os.path.join(d, "scores", f + ".jsonl")):
            if not s.get("error") and s["id"] in man:
                sc[s["id"]].update({k: v for k, v in s.items() if k in METRICS + ("must_not_hits", "nsfw_flag")})
    prompts = {p["cid"]: p for p in read_jsonl(PROMPTS)}
    routes = json.load(open(os.path.join(d, "routes.json")))
    by = defaultdict(dict)  # variant → pid → record
    for i, r in man.items():
        by[r["variant"]][r["pid"]] = {**r, **sc.get(i, {}), "category": prompts[r["pid"]]["category"]}
    arms = {}
    for v in ("naive", "fixed", "routed"):
        rs = [r for r in by[v].values() if not r.get("nsfw_flag")]
        neg = [r for r in rs if r["category"] == "negation" and "must_not_hits" in r]
        arms[v] = {"images": len(by[v]), "nsfw_flagged": len(by[v]) - len(rs),
                   **{m: mean(r.get(m) for r in rs) for m in METRICS},
                   "negation_violation": share(bool(r["must_not_hits"]) for r in neg),
                   "match_by_category": {c: mean(r.get("match") for r in rs if r["category"] == c)
                                         for c in sorted({r["category"] for r in rs})}}
    paired = {}
    for hi, lo in (("routed", "fixed"), ("fixed", "naive"), ("routed", "naive")):
        for m in METRICS:
            diffs = [by[hi][p][m] - by[lo][p][m] for p in by[hi] if p in by[lo]
                     and by[hi][p].get(m) is not None and by[lo][p].get(m) is not None]
            if diffs:
                paired[f"{hi}-{lo}:{m}"] = {"n": len(diffs), "mean": round(statistics.mean(diffs), 4),
                                            "ci90": boot_ci(diffs), "better": share(x > 0 for x in diffs)}
    moved = [p for p, r in routes.items() if r["model"] != FIXED]
    sub = {}
    for m in ("pickscore", "match", "imagereward"):  # only prompts where the router left juggernaut
        diffs = [by["routed"][p][m] - by["fixed"][p][m] for p in moved if p in by["routed"] and p in by["fixed"]
                 and by["routed"][p].get(m) is not None and by["fixed"][p].get(m) is not None]
        if diffs:
            sub[m] = {"n": len(diffs), "mean": round(statistics.mean(diffs), 4), "ci90": boot_ci(diffs)}
    dist = Counter(r["model"] for r in routes.values())
    style_dist = defaultdict(Counter)
    for p, r in routes.items():
        style_dist[prompts[p]["style"]][r["model"]] += 1
    out = {"uc": "19-e2e", "prompts": len(prompts), "routed_to": dist, "routed_by_style": style_dist,
           "arms": arms, "paired": paired, "routed_vs_fixed_where_moved": sub}
    L = ["# UC18+19 · celý řetězec na 50 nových českých promptech", "",
         "CZ prompt → router (`uc18/route.py`, SigLIP 2 nad t1 překladem + styl) → t2 dialekt (`uc19/translate.py`) → obrázek. "
         "Stejný seed pro všechna ramena. FLUX.1-dev router nevybírá (licence).", "",
         "- **naive**: Juggernaut + prostý překlad (t1)", "- **fixed**: Juggernaut + dialekt sdxl (t2), bez routeru",
         "- **routed**: checkpoint z routeru + jeho dialekt (t2)", "",
         f"Router poslal prompty na: {dict(dist)}", "", "## Ramena", "",
         "| rameno | obr. | PickScore | SigLIP cos | WD14 recall | ImageReward | VLM shoda 1–5 | VLM kvalita | negace porušena | NSFW gate |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for v, x in arms.items():
        L.append(f"| {v} | {x['images']} | {x['pickscore']} | {x['siglip_cos']} | {x['wd14_recall']} | {x['imagereward']} | "
                 f"{x['match']} | {x['quality']} | {x['negation_violation']} | {x['nsfw_flagged']} |")
    L += ["", "## Párově (stejný prompt, stejný seed)", "", "| srovnání:metrika | n | Δ průměr | 90% CI | lepší |", "|---|---|---|---|---|"]
    for k, x in paired.items():
        L.append(f"| {k} | {x['n']} | {x['mean']:+.4f} | {x['ci90']} | {x['better']} |")
    L += ["", f"Jen prompty, kde router zvolil jiný checkpoint než Juggernaut ({len(moved)}): "
          + ", ".join(f"{m} Δ {x['mean']:+.3f} CI {x['ci90']}" for m, x in sub.items()), "",
          "Router podle stylu:", ""]
    for s, c in sorted(style_dist.items()):
        L.append(f"- {s}: {dict(c)}")
    L += ["", "VLM shoda podle kategorie:", ""]
    for v, x in arms.items():
        L.append(f"- {v}: {x['match_by_category']}")
    open(a.md, "w").write("\n".join(L) + "\n")
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=False)
    print(f"report → {a.md}, {a.json}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("jobs", "report"))
    ap.add_argument("--dir", default="/cold/w2/uc19e2e")
    ap.add_argument("--router", default="/cold/w2/uc18/router.json")
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.cmd == "jobs":
        return cmd_jobs(a)
    a.md = a.md or os.path.join(a.dir, "report_UC19_e2e.md")
    a.json = a.json or os.path.join(H, "..", "bench", "results", "UC19_e2e.json")
    return cmd_report(a)


if __name__ == "__main__":
    sys.exit(main())
