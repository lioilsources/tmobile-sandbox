#!/usr/bin/env python3
"""UC18 report: rank.json (+ gen progress, scorer throughput) → Markdown draft + bench/results JSON.

  report.py --rank /cold/w2/uc18/rank.json --dir /cold/w2/uc18 \
            --md /cold/w2/uc18/report_uc18.md --json ~/tmobile-sandbox/bench/results/UC18_matrix.json

FLUX.1-dev rows are labelled "jen interní měření" (non-commercial licence). Images flagged by the WD14
rating gate are only counted, never linked.
"""
import argparse, json, os


def fmt(x, d=3):
    return "–" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    R = json.load(open(a.rank))
    prog = {}
    p = os.path.join(a.dir, "progress.json")
    if os.path.exists(p):
        prog = json.load(open(p))
    models = R["models"]
    L = ["# UC18 — FinetuneGallery: výběr modelu podle promptu a stylu (automaticky vygenerovaný návrh)", ""]
    L += [f"Obrázků: {R['images']}, modelů {len(models)}, stylů {len(R['styles'])}. "
          f"Generování: {fmt(prog.get('img_per_h'))} obr./h, medián {fmt(prog.get('exec_median_s'), 2)} s/obr., "
          f"GPU {fmt(prog.get('gpu_util_mean'), 0)} %, VRAM špička {fmt(prog.get('vram_peak_mib'))} MiB.", ""]
    L += ["## Metriky per model", "",
          "| model | obr. | PickScore | SigLIP cos | WD14 recall | ImageReward | composite | s/obr. | NSFW gate |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        x = R["per_model"][m]
        tag = " (FLUX.1-dev — jen interní měření)" if "flux_dev" in m else ""
        L.append(f"| {m}{tag} | {x['images']} | {fmt(x['pickscore'], 2)} | {fmt(x['siglip_cos'])} | {fmt(x['wd14_recall'])} | "
                 f"{fmt(x['imagereward'])} | {fmt(x['composite'])} | {fmt(x['exec_median_s'], 2)} | {x['nsfw_flagged']} |")
    for name, title in (("vlm", "VL soudce Qwen3-VL-8B (párově, adherence)"), ("metric", "metriky (composite)")):
        bt, ts = R.get(f"bt_{name}"), R.get(f"trueskill_{name}")
        if not bt:
            continue
        L += ["", f"## Bradley–Terry + TrueSkill — {title}", "",
              "| model | BT log-síla | 90% CI | P(nejlepší) | TrueSkill μ ± σ |", "|---|---|---|---|---|"]
        for m in sorted(models, key=lambda m: -bt["all"][m]["log_strength"]):
            b, t = bt["all"][m], ts["all"][m]
            L.append(f"| {m} | {b['log_strength']:+.2f} | [{b['ci90'][0]:+.2f}, {b['ci90'][1]:+.2f}] | {b['p_best']:.2f} | "
                     f"{t['mu']:.1f} ± {t['sigma']:.1f} |")
        L += ["", f"Nejlepší model per styl ({title}):", "", "| styl | 1. | 2. | P(best) 1. |", "|---|---|---|---|"]
        for st in R["styles"]:
            if st not in bt:
                continue
            o = sorted(bt[st], key=lambda m: -bt[st][m]["log_strength"])
            L.append(f"| {st} | {o[0]} | {o[1] if len(o) > 1 else '–'} | {bt[st][o[0]]['p_best']:.2f} |")
    v = R.get("vlm", {})
    L += ["", "## Soudce", "", f"Párů {v.get('pairs')}, kontrola prohozeného pořadí na {v.get('swap_checked')} → "
          f"konzistence pozice {fmt(v.get('position_consistency'))}; shoda soudce s composite metrikou (bez remíz) "
          f"{fmt(R.get('vlm_vs_metric_agreement'))}."]
    r = R.get("router_loo")
    if r:
        L += ["", "## Router (prototyp, leave-one-prompt-out)", "",
              f"kNN k={r['k']} nad {'SigLIP embeddingem promptu' if r['knn_embeddings'] else 'stejným stylem (bez embeddingů)'}, "
              f"α={r['alpha']} (zbytek = BT prior stylu). Prompty: {r['n_prompts']}.", "",
              "| strategie | top-1 = orákulum | průměrný regret (z) |", "|---|---|---|",
              f"| router | {r['router_top1']:.3f} | {r['router_regret']:.3f} |",
              f"| nejlepší model stylu (BT) | {r['style_best_top1']:.3f} | {r['style_best_regret']:.3f} |",
              f"| globálně nejlepší ({r['global_best']}) | {r['global_best_top1']:.3f} | {r['global_best_regret']:.3f} |",
              f"| náhodně | {r['chance_top1']:.3f} | {r['random_regret']:.3f} |"]
    os.makedirs(os.path.dirname(os.path.abspath(a.md)), exist_ok=True)
    open(a.md, "w").write("\n".join(L) + "\n")
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    json.dump({"uc": 18, "generation": prog, **{k: R.get(k) for k in ("images", "models", "styles", "per_model", "vlm",
              "vlm_vs_metric_agreement", "router_loo")},
               "bt_vlm_all": (R.get("bt_vlm") or {}).get("all"), "bt_metric_all": (R.get("bt_metric") or {}).get("all"),
               "licence_note": "flux_dev = FLUX.1-dev non-commercial, internal measurement only"},
              open(a.json, "w"), indent=1, ensure_ascii=False)
    print(f"report → {a.md}, {a.json}")


if __name__ == "__main__":
    main()
