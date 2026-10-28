"""UC24 krok 3 — učený gate místo ručních prahů: GBM nad příznaky kroku 1 (+ CLAP embedding)
→ p(nepříjemné), s konformním prahem pro garantovaný recall ≥ 0.95, export do ONNX.

  python learned.py --out-dir /cold/uc24/learned --json ../bench/results/UC24_learned.json

!! Štítky jsou zatím SYNTETICKÉ (vložené vady ze synth.py): nepříjemné = rumble|clipping|hiss|haze|clicks.
!! Po Olinových 150 hodnoceních + pseudo-labelech (shoda ≥ 2 soudců, viz plán) přetrénovat — tenhle model
!! se naučil hlavně "jak vypadají MOJE syntetické vady", ne co vadí posluchači.
Rozdělení podle base klipu: train = seed 0–1 (ACE 1–2), kalibrace = seed 2 (ACE 3), test = seed 3–5 (ACE 4–6).
Varianty: "signal" (jen příznaky, CPU, bez neuronky — tohle jde na Spark do audio služby) a
"signal+clap" (+512 d CLAP clap-htsat-unfused — má embeddingy pro celou sadu; potřebuje CLAP audio věž při inferenci).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

UNPLEASANT = {"rumble", "clipping", "hiss", "haze", "clicks"}
ALPHA = 0.05  # cíl recall ≥ 1-α


def part(r):
    if r["split"] == "test":
        return "test"
    s = r.get("seed", 0)
    if r["type"] == "ace_music":
        return "train" if s <= 2 else "cal"
    return "train" if s <= 1 else "cal"


def conformal_threshold(pos_scores: np.ndarray, alpha: float = ALPHA) -> float:
    """Split-conformal práh pro recall: P(skóre nové vady ≥ t) ≥ 1-α (výměnnost kal. a test vad).
    t = k-té nejmenší skóre kalibračních vad, k = ⌊α·(n+1)⌋ (k = 0 → -inf, tj. vše vadné)."""
    s = np.sort(pos_scores)
    k = int(np.floor(alpha * (len(s) + 1)))
    return float(s[k - 1]) if k >= 1 else -np.inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/cold/uc24/synth/manifest.jsonl")
    ap.add_argument("--gate", default="/cold/uc24/gate_dump.jsonl")
    ap.add_argument("--clap", default="/cold/uc24/judges/clap-unfused.npz")
    ap.add_argument("--out-dir", default="/cold/uc24/learned")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    import lightgbm as lgb
    os.makedirs(a.out_dir, exist_ok=True)
    G = [json.loads(line) for line in open(a.gate)]
    man = {json.loads(line)["id"]: json.loads(line) for line in open(a.manifest)}
    fnames = sorted(G[0]["features"])
    kinds = ["sfx", "ambient", "music"]
    X = np.array([[g["features"][f] for f in fnames] + [float(g["kind"] == k) for k in kinds] for g in G], np.float32)
    names = fnames + [f"kind_{k}" for k in kinds]
    y = np.array([g["defect"] in UNPLEASANT for g in G], int)
    P = np.array([part(man[g["id"]]) for g in G])
    variants = {"signal": (X, names)}
    if os.path.exists(a.clap):
        c = np.load(a.clap, allow_pickle=True)
        idx = {p: i for i, p in enumerate(c["paths"])}
        E = np.array([c["emb"][idx[man[g["id"]]["path"]]] for g in G], np.float32)
        variants["signal+clap"] = (np.hstack([X, E]), names + [f"clap{i}" for i in range(E.shape[1])])
    res = {"uc": "UC24", "what": "učený gate (LightGBM) nad syntetickými štítky — NUTNO přetrénovat na Olinových",
           "labels": "nepříjemné = " + "|".join(sorted(UNPLEASANT)), "alpha": ALPHA,
           "n": {p: int((P == p).sum()) for p in ("train", "cal", "test")}, "variants": {}}
    hand = np.array([g["verdict"] != "pass" for g in G])
    te = P == "test"
    clean = np.array([g["defect"] == "clean" for g in G])
    res["hand_thresholds_test"] = {"recall": round(float(hand[te & (y == 1)].mean()), 3),
                                   "fpr_clean": round(float(hand[te & clean].mean()), 3)}
    for vn, (Xv, nv) in variants.items():
        tr, ca = P == "train", P == "cal"
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=24, n_jobs=2)
        m.fit(Xv[tr], y[tr])
        p = m.predict_proba(Xv)[:, 1]
        t = conformal_threshold(p[ca & (y == 1)])
        flag = p >= t
        from sklearn.metrics import roc_auc_score
        per = {}
        for d in sorted({g["defect"] for g in G}):
            msk = te & np.array([g["defect"] == d for g in G])
            per[d] = round(float(flag[msk].mean()), 3)
        imp = sorted(zip(nv, m.booster_.feature_importance("gain")), key=lambda kv: -kv[1])[:12]
        res["variants"][vn] = {
            "conformal_threshold": round(t, 4), "auc_test": round(float(roc_auc_score(y[te], p[te])), 4),
            "recall_test": round(float(flag[te & (y == 1)].mean()), 3), "fpr_test_clean": round(float(flag[te & clean].mean()), 3),
            "flag_rate_by_defect_test": per, "top_features_gain": [(n, round(float(v), 1)) for n, v in imp],
            "n_features": len(nv)}
        # ONNX
        try:
            import onnxmltools
            from onnxmltools.convert.common.data_types import FloatTensorType
            onx = onnxmltools.convert_lightgbm(m.booster_, initial_types=[("x", FloatTensorType([None, Xv.shape[1]]))],
                                               target_opset=15, zipmap=False)
            path = os.path.join(a.out_dir, f"sfxgate_{vn.replace('+', '_')}.onnx")
            open(path, "wb").write(onx.SerializeToString())
            import onnxruntime as ort
            s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            out = s.run(None, {"x": Xv[te][:256]})
            po = np.asarray(out[1])[:, 1] if len(out) > 1 else np.asarray(out[0]).ravel()
            res["variants"][vn]["onnx"] = {"path": path, "bytes": os.path.getsize(path),
                                           "max_abs_diff_vs_lgbm": float(np.abs(po - p[te][:256]).max())}
            json.dump({"features": nv, "threshold": t, "alpha": ALPHA, "labels": "synthetic"},
                      open(path.replace(".onnx", ".json"), "w"), indent=1)
        except Exception as e:  # noqa: BLE001
            res["variants"][vn]["onnx_error"] = repr(e)
    js = json.dumps(res, indent=1, ensure_ascii=False)
    if a.json:
        open(a.json, "w").write(js)
    print(js)


if __name__ == "__main__":
    main()
