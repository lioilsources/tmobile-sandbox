#!/usr/bin/env python3
"""UC18 router prototype: which checkpoint for this prompt + style?

  route.py --router /cold/w2/uc18/router.json --pid photo_portrait-03          # in-set prompt (LOO: excludes itself)
  route.py --router … --text "a lighthouse in a storm, oil painting" --style painting   # embeds with SigLIP 2 (CPU ok)

Answer = what GET /api/route?prompt=&style= would return: [{model, score, p_best, n}] best first.
utility = α · mean composite of the k nearest prompts (SigLIP text cosine) + (1-α) · style BT log-strength;
p_best = (wins among the neighbours + 1) / (k + #models), wins = neighbour prompts where the model was best.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import limit_threads  # noqa: E402

limit_threads(4)
import numpy as np  # noqa: E402


def route(R, emb, style, exclude=None):
    pids = R["pids"]
    E = np.array(R["emb"], dtype=np.float32)
    sims = E @ (emb / np.linalg.norm(emb))
    if exclude is not None:
        sims[pids.index(exclude)] = -9
    nb = [pids[j] for j in np.argsort(-sims)[: R["k"]]]
    models = sorted({m for p in pids for m in R["utility"][p]})
    prior = R["style_prior"].get(style, R["style_prior"].get("all", {}))
    wins = {m: 0 for m in models}
    for q in nb:
        u = R["utility"][q]
        wins[max(u, key=u.get)] += 1
    out = []
    for m in models:
        knn = float(np.mean([R["utility"][q][m] for q in nb if m in R["utility"][q]] or [0.0]))
        s = R["alpha"] * knn + (1 - R["alpha"]) * prior.get(m, {}).get("log_strength", 0.0)
        out.append({"model": m, "score": round(s, 3), "p_best": round((wins[m] + 1) / (len(nb) + len(models)), 3), "n": len(nb)})
    return sorted(out, key=lambda x: -x["score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", required=True)
    ap.add_argument("--pid")
    ap.add_argument("--text")
    ap.add_argument("--style", default="")
    a = ap.parse_args()
    R = json.load(open(a.router))
    if not R.get("emb"):
        raise SystemExit("router.json has no embeddings (score.sh phase A writes scores/text_emb.npz)")
    if a.pid:
        i = R["pids"].index(a.pid)
        res = route(R, np.array(R["emb"][i], dtype=np.float32), a.style or R["style_of"][a.pid], exclude=a.pid)
    else:
        import torch
        from transformers import AutoModel, AutoProcessor
        repo = "google/siglip2-so400m-patch16-512"
        proc, m = AutoProcessor.from_pretrained(repo), AutoModel.from_pretrained(repo).eval()
        with torch.inference_mode():
            t = proc(text=[a.text], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
            f = m.get_text_features(**t)
            f = (f if torch.is_tensor(f) else f.pooler_output)[0].float().numpy()
        res = route(R, f, a.style)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
