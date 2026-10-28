"""UC24 krok 2 — vyhodnocení soudců na syntetické sadě (pravda známá ze synth.py).

  python eval_judges.py [--qwen /cold/uc24/judges/qwen2audio.jsonl] --out ../bench/results/UC24_judges.json

Dvě otázky:
  (1) VADA: umí soudce odlišit klip s vloženou vadou od jeho čisté verze? AUC (test split) +
      záchyt při prahu z dev (90. percentil čistých dev klipů ⇒ ~10 % falešných poplachů).
  (2) SHODA S POPISEM: klip × vlastní popis vs klip × cizí popis (mismatch řádky) → AUC + přesnost
      při prahu z dev (Youdenův bod).
Skóre "horší = víc": sfxgate score, -PQ (Audiobox), CLAP sondy (cos k popisu vady − cos ke
"clean recording"), PANNs suma pravděpodobností šumových tříd AudioSet.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

J = os.environ.get("UC24_JUDGES_DIR", "/cold/uc24/judges")
DEFECTS = ["rumble", "clipping", "hiss", "haze", "clicks", "dc_offset", "lead_silence"]
PROBES = {  # CLAP zero-shot sondy vad (angl., jak CLAP trénoval)
    "hiss": "a recording with loud background hiss and static noise",
    "rumble": "a recording with low frequency rumble and boomy bass hum",
    "clipping": "a harsh distorted clipped recording",
    "clicks": "a recording with clicks and pops",
    "generic": "a noisy low quality recording",
}
CLEAN_PROBE = "a clean high quality recording"
NOISE_CLASSES = ["Static", "White noise", "Pink noise", "Hum", "Mains hum", "Buzz", "Rumble", "Distortion",
                 "Noise", "Crackle", "Clicking", "Hiss"]


def auc(pos, neg):
    """P(skóre vadného > skóre čistého), ties = 1/2 (Mann–Whitney)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    # průměrné pořadí pro shody
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load_npz(name):
    p = os.path.join(J, name + ".npz")
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/cold/uc24/synth/manifest.jsonl")
    ap.add_argument("--gate", default="/cold/uc24/gate_dump.jsonl")
    ap.add_argument("--classes", default="/cold/uc24/prompt_classes.json")
    ap.add_argument("--qwen", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.manifest)]
    gate = {json.loads(line)["id"]: json.loads(line) for line in open(a.gate)}
    scores: dict[str, dict] = {}  # judge -> {row id: badness}
    match: dict[str, dict] = {}   # judge -> {row id: match score (vyšší = sedí)}
    meta = {}

    scores["sfxgate"] = {r["id"]: 0.0 for r in rows}
    from sfxgate.gate import judge as gjudge
    for r in rows:
        g = gate.get(r["id"])
        if g:
            scores["sfxgate"][r["id"]] = gjudge(g["features"], r["kind"])["score"]

    aes = load_npz("aesthetics")
    if aes is not None:
        idx = {p: i for i, p in enumerate(aes["paths"])}
        for ax in ("PQ", "CE", "CU", "PC"):
            scores[f"aesthetics_{ax}"] = {r["id"]: -float(aes[ax][idx[r["path"]]]) for r in rows if r["path"] in idx}
        meta["aesthetics"] = {k: float(aes[k]) for k in ("seconds", "n", "audio_s") if k in aes}

    for cname, mid in (("clap-general", "laion/larger_clap_general"), ("clap-unfused", "laion/clap-htsat-unfused")):
        c = load_npz(cname)
        if c is None:
            continue
        from sfxgate.clap import Clap
        model = Clap(mid, "cpu")
        idx = {p: i for i, p in enumerate(c["paths"])}
        E = c["emb"]
        clean_t = model.text_emb(CLEAN_PROBE)
        for k, txt in PROBES.items():
            t = model.text_emb(txt)
            scores[f"{cname}_probe_{k}"] = {r["id"]: float(E[idx[r["path"]]] @ t - E[idx[r["path"]]] @ clean_t)
                                            for r in rows if r["path"] in idx}
        match[cname] = {r["id"]: float(E[idx[r["path"]]] @ model.text_emb(r["prompt"])) for r in rows if r["path"] in idx}
        meta[cname] = {k: float(c[k]) for k in ("seconds", "n", "audio_s") if k in c}

    pn = load_npz("panns")
    if pn is not None:
        labels = list(pn["labels"])
        idx = {p: i for i, p in enumerate(pn["paths"])}
        P = pn["probs"]
        nz = [labels.index(n) for n in NOISE_CLASSES if n in labels]
        scores["panns_noise"] = {r["id"]: float(P[idx[r["path"]], nz].sum()) for r in rows if r["path"] in idx}
        pc = json.load(open(a.classes)) if os.path.exists(a.classes) else {}
        m = {}
        for r in rows:
            cl = [labels.index(c) for c in pc.get(r["prompt"], {}).get("classes", []) if c in labels]
            if r["path"] in idx and cl:
                m[r["id"]] = float(P[idx[r["path"]], cl].max())
        match["panns_class"] = m
        meta["panns"] = {k: float(pn[k]) for k in ("seconds", "n", "audio_s") if k in pn}

    if a.qwen and os.path.exists(a.qwen):
        q = {json.loads(line)["id"]: json.loads(line) for line in open(a.qwen)}
        scores["qwen2audio_unpleasant"] = {i: float(bool(v.get("unpleasant"))) for i, v in q.items() if v.get("parsed")}
        scores["qwen2audio_any_defect"] = {i: float(bool([d for d in (v.get("defects") or []) if d != "none"]))
                                           for i, v in q.items() if v.get("parsed")}
        match["qwen2audio"] = {i: float(bool(v.get("matches"))) for i, v in q.items() if v.get("parsed")}
        meta["qwen2audio_parsed"] = {"n": len(q), "parsed": sum(v.get("parsed", False) for v in q.values())}

    by = {r["id"]: r for r in rows}
    res = {"uc": "UC24", "what": "soudci kroku 2 na syntetické sadě (test split; prahy z dev)", "meta": meta,
           "defect": {}, "match": {}}
    for jn, sc in scores.items():
        out = {}
        for kind in ("sfx", "ambient", "music"):
            dev_clean = [s for i, s in sc.items() if by[i]["split"] == "dev" and by[i]["kind"] == kind and by[i]["defect"] == "clean"]
            if not dev_clean:
                continue
            # sfxgate má vlastní ruční prahy: práh = "jakýkoli důvod" (verdikt ≠ pass), ne p90 dev
            thr = 0.0 if jn == "sfxgate" else float(np.percentile(dev_clean, 90))
            te_clean = [s for i, s in sc.items() if by[i]["split"] == "test" and by[i]["kind"] == kind and by[i]["defect"] == "clean"]
            o = {"thr_dev_p90_clean": round(thr, 4), "fpr_test": round(float(np.mean(np.array(te_clean) > thr)), 3) if te_clean else None}
            for d in DEFECTS:
                pos = [s for i, s in sc.items() if by[i]["split"] == "test" and by[i]["kind"] == kind and by[i]["defect"] == d]
                if pos:
                    o[d] = {"auc": round(auc(pos, te_clean), 3) if te_clean else None, "det": round(float(np.mean(np.array(pos) > thr)), 3)}
            out[kind] = o
        res["defect"][jn] = out
    for jn, sc in match.items():
        out = {}
        for kind in ("sfx", "ambient"):
            def sel(split, mm):
                return [s for i, s in sc.items() if by[i]["split"] == split and by[i]["kind"] == kind
                        and ((by[i]["defect"] == "mismatch") == mm) and by[i]["defect"] in ("clean", "mismatch")]
            dv_t, dv_f, te_t, te_f = sel("dev", False), sel("dev", True), sel("test", False), sel("test", True)
            if not te_t or not te_f:
                continue
            cand = np.unique(np.concatenate([dv_t, dv_f]))
            best = max(cand, key=lambda t: np.mean(np.array(dv_t) >= t) + np.mean(np.array(dv_f) < t))
            acc = (np.sum(np.array(te_t) >= best) + np.sum(np.array(te_f) < best)) / (len(te_t) + len(te_f))
            out[kind] = {"auc_match_vs_mismatch": round(auc(te_t, te_f), 3), "thr_dev": round(float(best), 4),
                         "acc_test": round(float(acc), 3), "mean_match": round(float(np.mean(te_t)), 3),
                         "mean_mismatch": round(float(np.mean(te_f)), 3), "n": len(te_t) + len(te_f)}
        res["match"][jn] = out
    js = json.dumps(res, indent=1, ensure_ascii=False)
    if a.out:
        open(a.out, "w").write(js)
    print(js)


if __name__ == "__main__":
    main()
