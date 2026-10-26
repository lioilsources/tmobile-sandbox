#!/usr/bin/env python3
"""Porovná extrakci s ground truth → přesnost po polích, čisté vs. sken, rychlost, cena.

  evaluate.py --docs /srv/work/demo3/docs --run /srv/work/demo3/runs/<run> [--kc-per-gpu-hour 60]
Zapíše <run>/report.json a <run>/report.md.
"""
import argparse, json, pathlib, re, unicodedata, collections, statistics

def norm(v):
    if v is None: return None
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return round(float(v), 2)
    s = unicodedata.normalize("NFC", str(v)).strip().lower()
    s = re.sub(r"\s+", " ", s).replace("„", '"').replace("“", '"')
    return s

def same(gt, pr, key):
    if key.endswith("_ico") or key == "variabilni_symbol":
        return re.sub(r"\D", "", str(gt or "")) == re.sub(r"\D", "", str(pr or ""))
    if key == "bankovni_ucet" or key.endswith("_dic"):
        return re.sub(r"\s", "", str(gt or "")).upper() == re.sub(r"\s", "", str(pr or "")).upper()
    g, p = norm(gt), norm(pr)
    if isinstance(g, float) and isinstance(p, (float, int)): return abs(g - p) < 0.011
    return g == p

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--docs", default="/srv/work/demo3/docs"); ap.add_argument("--run", required=True)
    ap.add_argument("--kc-per-gpu-hour", type=float, default=None, help="cena L40S za hodinu (Kč); bez ní se cena nepočítá")
    a = ap.parse_args(); run = pathlib.Path(a.run)
    field = collections.defaultdict(lambda: [0, 0]); field_scan = collections.defaultdict(lambda: [0, 0])
    docs_ok = collections.Counter(); docs_n = collections.Counter(); errors = []; lat, pages, toks_in, toks_out = [], 0, 0, 0
    for gtf in sorted(pathlib.Path(a.docs).glob("*.json")):
        gt = json.loads(gtf.read_text()); rf = run / gtf.name
        if not rf.exists(): errors.append((gtf.stem, "chybí výstup", "", "")); continue
        r = json.loads(rf.read_text()); pr = r["pred"]
        lat.append(r["_llm_s"]); pages += r["_stran"]; toks_in += r["_usage"]["prompt_tokens"]; toks_out += r["_usage"]["completion_tokens"]
        grp = f"{gt['typ']} · {'sken' if gt['_sken'] else 'čisté PDF'}"; ok_all = True
        for k, v in gt.items():
            if k.startswith("_"): continue
            ok = same(v, pr.get(k), k); ok_all &= ok
            field[(gt["typ"], k)][0] += ok; field[(gt["typ"], k)][1] += 1
            field_scan[grp][0] += ok; field_scan[grp][1] += 1
            if not ok: errors.append((gtf.stem, k, v, pr.get(k)))
        docs_n[grp] += 1; docs_ok[grp] += ok_all
    meta = json.loads((run / "_run.json").read_text())
    tot_ok = sum(v[0] for v in field.values()); tot = sum(v[1] for v in field.values())
    wall_h = meta["wall_s"] / 3600
    rep = {"model": meta["model"], "dokumentu": sum(docs_n.values()), "stran": pages, "soubeh": meta["concurrency"], "dpi": meta.get("dpi", 150), "prompt": meta.get("prompt", "v1"),
           "presnost_poli": round(tot_ok / tot, 4), "dokumenty_bez_chyby": round(sum(docs_ok.values()) / max(1, sum(docs_n.values())), 4),
           "podle_skupiny": {g: {"pole": round(field_scan[g][0] / field_scan[g][1], 4), "dokumenty_bez_chyby": f"{docs_ok[g]}/{docs_n[g]}"} for g in sorted(docs_n)},
           "podle_pole": {f"{t}.{k}": round(o / n, 3) for (t, k), (o, n) in sorted(field.items())},
           "latence_dokument_s": {"median": round(statistics.median(lat), 2), "p95": round(sorted(lat)[int(len(lat) * .95) - 1], 2)},
           "propustnost": {"stran_za_hodinu": round(pages / wall_h), "dokumentu_za_hodinu": round(len(lat) / wall_h),
                           "beh_s": meta["wall_s"]},
           "tokeny_prumer_na_stranu": {"vstup": round(toks_in / pages), "vystup": round(toks_out / pages)},
           "chyby": [{"doc": d, "pole": k, "spravne": g, "model": p} for d, k, g, p in errors]}
    if a.kc_per_gpu_hour:
        rep["cena_kc_za_1000_stran"] = round(a.kc_per_gpu_hour / rep["propustnost"]["stran_za_hodinu"] * 1000, 2)
    (run / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    md = [f"# Demo 3 – vytěžování dokumentů\n", f"Model **{rep['model']}** na 1× L40S, {rep['dokumentu']} dokumentů / {rep['stran']} stran, souběh {rep['soubeh']}, {rep['dpi']} dpi, prompt {rep['prompt']}.\n",
          f"| Metrika | Hodnota |\n|---|---|",
          f"| Přesnost polí | **{rep['presnost_poli']:.1%}** |", f"| Dokumenty zcela bez chyby | {rep['dokumenty_bez_chyby']:.0%} |",
          f"| Propustnost | **{rep['propustnost']['stran_za_hodinu']} stran/h** ({rep['propustnost']['dokumentu_za_hodinu']} dok./h) |",
          f"| Latence dokumentu (medián / p95) | {rep['latence_dokument_s']['median']} s / {rep['latence_dokument_s']['p95']} s |"]
    if "cena_kc_za_1000_stran" in rep: md.append(f"| Cena (GPU {a.kc_per_gpu_hour:.0f} Kč/h) | {rep['cena_kc_za_1000_stran']} Kč / 1000 stran |")
    md += ["\n## Podle typu a kvality\n| Skupina | Přesnost polí | Dokumenty bez chyby |\n|---|---|---|"]
    md += [f"| {g} | {v['pole']:.1%} | {v['dokumenty_bez_chyby']} |" for g, v in rep["podle_skupiny"].items()]
    md += ["\n## Nejslabší pole\n| Pole | Přesnost |\n|---|---|"]
    md += [f"| {k} | {v:.0%} |" for k, v in sorted(rep["podle_pole"].items(), key=lambda x: x[1])[:8]]
    md += ["\n## Chyby\n| Dokument | Pole | Správně | Model |\n|---|---|---|---|"]
    md += [f"| {e['doc']} | {e['pole']} | {e['spravne']} | {e['model']} |" for e in rep["chyby"][:40]]
    (run / "report.md").write_text("\n".join(md)); print("\n".join(md[:12]))

if __name__ == "__main__": main()
