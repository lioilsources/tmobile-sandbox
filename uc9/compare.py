#!/usr/bin/env python3
# UC9: shoda extrakce (běh A) s rozhodčím (běh B, větší model) na stejném vzorku + kontrola IČO proti metadatům.
#   compare.py runs/full/extracted.jsonl runs/judge32b/extracted.jsonl --inp contracts.jsonl --out results.json
import argparse, collections, json

ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--inp", required=True); ap.add_argument("--out")
x = ap.parse_args()
load = lambda p: {j["id"]: j["x"] for j in map(json.loads, open(p)) if j.get("x")}
A, B = load(x.a), load(x.b)
ids = set(A) & set(B)
meta = {}
for l in open(x.inp):
    r = json.loads(l)
    if r["id"] in ids: meta[r["id"]] = r
agree, conf = collections.Counter(), collections.defaultdict(collections.Counter)
ico_in_parties = collections.Counter()
for i in ids:
    a, b = A[i], B[i]
    for k in ("typ", "plneni", "odvetvi", "dodavatel_ico", "it_relevance"):
        agree[k] += a[k] == b[k]
    agree["it_binarni"] += (a["it_relevance"] >= 2) == (b["it_relevance"] >= 2)
    conf["odvetvi"][(a["odvetvi"], b["odvetvi"])] += a["odvetvi"] != b["odvetvi"]
    m = meta[i]; icos = {m["subjekt"]["ico"]} | {s["ico"] for s in m["strany"]}
    for name, v in (("A", a), ("B", b)):
        ico_in_parties[name] += (v["dodavatel_ico"] in icos) or (v["dodavatel_ico"] == "" and not any(icos))
it_a = sum(A[i]["it_relevance"] >= 2 for i in ids); it_b = sum(B[i]["it_relevance"] >= 2 for i in ids)
both = sum(A[i]["it_relevance"] >= 2 and B[i]["it_relevance"] >= 2 for i in ids)
n = len(ids)
res = {"vzorek": n, "shoda_%": {k: round(100 * v / n, 1) for k, v in agree.items()},
       "ico_mezi_stranami_%": {k: round(100 * v / n, 1) for k, v in ico_in_parties.items()},
       "it>=2": {"A": it_a, "B": it_b, "oba": both,
                 "precision_vs_B_%": round(100 * both / it_a, 1) if it_a else None, "recall_vs_B_%": round(100 * both / it_b, 1) if it_b else None},
       "nejcastejsi_neshody_odvetvi": [[f"{p} ↔ {q}", c] for (p, q), c in conf["odvetvi"].most_common(8) if c]}
print(json.dumps(res, ensure_ascii=False, indent=1))
if x.out: open(x.out, "w").write(json.dumps(res, ensure_ascii=False, indent=1))
