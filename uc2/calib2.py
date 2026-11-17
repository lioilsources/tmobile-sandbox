#!/usr/bin/env python3
"""UC2 kolo 2, plán §7: ověří skóre evolve2 na Olinově hodnocení filtrů (líbí / nelíbí) před plným během.

Každý shader projde stejnými branami a skórem jako dítě v evolve2. Rodina: lab* podle SEED_MAP, c-čísla z evo1 podle
nejvyšší shody s textem rodiny. Ref (průměr a rozptyl shody s tématem) se počítá ze seedů rodiny jako v Island.run,
novost proti ostatním shaderům téže rodiny. Výstup: AUC celkového skóre a každé složky, brány pro líbí / nelíbí.

  calib2.py --mirrorbooth /srv/work/uc2/MirrorBooth --selfies /srv/work/uc2/selfies --flutter-root /srv/work/uc4/flutter \
            --seeds /srv/work/uc2/MirrorBooth/mirrorbooth/shaders --seeds /srv/work/uc2/roundb \
            --evo1 /srv/work/uc2/runs/evo1 --ratings ratings.json --out /srv/work/uc2/runs/calib2
"""
import argparse, asyncio, concurrent.futures, json, pathlib, sys
from types import SimpleNamespace

import httpx
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import evolve2 as V  # noqa: E402
from families import FAMILIES, SEED_MAP, gate, probe_errors  # noqa: E402


def auc(pos, neg):
    """Pravděpodobnost, že náhodný líbivý má vyšší hodnotu než náhodný nelíbivý (shoda = 0,5)."""
    if not pos or not neg: return None
    return round(sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg)), 3)


async def main_async(a):
    sys.path.insert(0, str(a.mirrorbooth / "pipeline"))
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rt = json.loads(pathlib.Path(a.ratings).read_text())
    rating = {v: k for k in ("like", "dislike") for v in rt[k].values()}
    seeds = {p.stem.replace("filter_", ""): p for d in a.seeds for p in pathlib.Path(d).glob("*.frag")}
    paths = {**{p.stem: p for p in pathlib.Path(a.evo1).glob("*/shaders/c*.frag")}, **seeds}
    # seeds dropped from SEED_MAP after rating (labLed, 11. 11.) keep their original family
    where = {"labLed": "mozaika", **{n: f for f, names in SEED_MAP.items() for n in names.split()}}
    # trial exports ("trial" section): ids repeat across families, so the key is t:<family>/<id> and the family is given
    for k in ("like", "dislike"):
        for fid in rt.get("trial", {}).get(k, []):
            f, c = fid.split("/"); n = f"t:{fid}"
            rating[n], where[n], paths[n] = k, f, pathlib.Path(a.trial) / f / "shaders" / f"{c}.frag"
    missing = [n for n in rating if n not in paths or not paths[n].exists()]
    if missing: raise SystemExit(f"chybí shadery: {missing}")

    from checks.contract import uniform_shape
    pool = concurrent.futures.ThreadPoolExecutor(1)
    loop = asyncio.get_running_loop()
    scorer = await loop.run_in_executor(pool, V.Scorer2, a.mirrorbooth, a.selfies)
    for f in FAMILIES: V.E.THEMES[f] = FAMILIES[f]["desc"]
    th = {f: await loop.run_in_executor(pool, scorer._txt, f) for f in FAMILIES}
    args = SimpleNamespace(max_frame_ms=16.0, max_alu=8000, max_flicker=0.6, dup_cos=2.0, theme_floor=a.theme_floor, w_theme=a.w_theme, w_novel=a.w_novel,
                           w_pick=a.w_pick, ev_sem=asyncio.Semaphore(4), imp_pool=concurrent.futures.ThreadPoolExecutor(3), flutter_root=a.flutter_root)
    rows = []
    # evo1 karty neznalo: karta se v kalibraci jen zaznamená do "gates", jinak by vyřadila většinu c-čísel
    card = V.check_card; V.check_card = lambda code, f: []
    async with httpx.AsyncClient() as http:
        # c-čísla: rodina podle nejvyšší shody renderu s textem rodiny (karta se nekontroluje, evo1 karty neznalo)
        fam = {}
        for n in rating:
            if n in where: fam[n] = where[n]; continue
            code = paths[n].read_text(); shape = uniform_shape(code)
            if shape is None: fam[n] = None; continue
            sc = await loop.run_in_executor(pool, scorer.score2, code, "mozaika", *shape)
            if "emb" not in sc: fam[n] = None; continue
            e = scorer._img([sc["frame"]])
            fam[n] = max(FAMILIES, key=lambda f: float((e @ th[f].T).mean()))
        for f in FAMILIES:
            isl = V.Island(f, args, out, scorer, pool, http)
            isl.ref = (0.0, 1.0)
            ref = []
            for s in SEED_MAP[f].split():           # ref ze seedů rodiny, stejně jako Island.run
                r = await isl.assess(f"seed_{s}", seeds[s].read_text(), None, s, "seed", seed=True)
                if r["stage"] == "ok": ref.append(r["theme_cos"] * 100)
            isl.ref = (float(np.mean(ref)), float(np.std(ref))) if ref else (0.0, 1.0)
            recs = []
            for n in [n for n in rating if fam[n] == f]:
                p = paths[n]
                # seed=True: skóre dostane i to, co by brána sila/sonda vyřadila; brány se vyhodnotí zvlášť do "gates"
                r = await isl.assess(n, p.read_text(), None, n, "calib", seed=True)
                r["rating"], r["family"] = rating[n], f
                r["set"] = "trial" if n.startswith("t:") else "lab" if n in seeds else "evo1"
                if r["stage"] == "ok":
                    mh = gate(f, "max_hist"); g = []
                    if r["face_keep"] > gate(f, "max_keep") or r["strength"] < gate(f, "min_strength") or (mh and r["hist"] > mh): g.append("sila")
                    g += probe_errors(f, r["probe"])
                    if card(p.read_text(), f): g.append("karta")
                    r["gates"] = g; recs.append(r)
                rows.append(r)
            embs = {r["id"]: r["_emb"] for r in recs if "_emb" in r}
            for r in recs:
                if "_emb" not in r: continue
                isl.embs = {k: e for k, e in embs.items() if k != r["id"]}
                isl.recs = {x["id"]: x for x in recs if x["id"] != r["id"] and "ident" in x}
                r["novelty"] = round(isl.novelty(r), 3); r["score"] = isl.score(r)
                r["z_theme"] = round((r["theme_cos"] * 100 - isl.ref[0]) / max(isl.ref[1], isl.a.theme_floor), 3)
            print(f"[{f}] ref {isl.ref[0]:.1f}±{isl.ref[1]:.2f}, {len(recs)} hodnocených", flush=True)
    for r in rows:
        for k in [k for k in r if k.startswith("_")]: r.pop(k)
    for n, f in fam.items():
        if f is None: rows.append({"id": n, "rating": rating[n], "stage": "render", "family": None})
    (out / "calib.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=str))

    def split(key):
        ok = [r for r in rows if key in r and r["stage"] == "ok"]
        return [r[key] for r in ok if r["rating"] == "like"], [r[key] for r in ok if r["rating"] == "dislike"]
    res = {"n": {"like": sum(r["rating"] == "like" for r in rows), "dislike": sum(r["rating"] == "dislike" for r in rows)},
           "auc": {k: auc(*split(k)) for k in ("score", "z_theme", "novelty", "pick", "composite", "strength", "face_keep")},
           "auc_bez_bran": auc([r["score"] for r in rows if "score" in r and not r["gates"] and r["rating"] == "like"],
                               [r["score"] for r in rows if "score" in r and not r["gates"] and r["rating"] == "dislike"]),
           "brany_sila_sonda": {k: sum(bool(r.get("gates")) and r["rating"] == k for r in rows) for k in ("like", "dislike")},
           "brany": {s: {"like": sum(r["stage"] == s and r["rating"] == "like" for r in rows),
                         "dislike": sum(r["stage"] == s and r["rating"] == "dislike" for r in rows)}
                     for s in sorted({r["stage"] for r in rows})},
           "rodiny": {f: sum(v == f for v in fam.values()) for f in FAMILIES},
           "auc_sady": {st: auc([r["score"] for r in rows if r.get("set") == st and "score" in r and r["rating"] == "like"],
                                [r["score"] for r in rows if r.get("set") == st and "score" in r and r["rating"] == "dislike"])
                        for st in ("lab", "evo1", "trial")}}
    (out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False), flush=True)
    for r in sorted((r for r in rows if "score" in r), key=lambda r: -r["score"]):
        print(f"{r['score']:7.3f} {r['rating']:8s} {r['family']:9s} {r['id']:20s} z={r['z_theme']:.2f} nov={r['novelty']:.2f} "
              f"pick={r['pick']:.1f} {','.join(r['gates'])}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    for k in ("mirrorbooth", "selfies", "evo1"): ap.add_argument(f"--{k}", type=pathlib.Path, required=True)
    ap.add_argument("--trial", type=pathlib.Path, default=pathlib.Path("/srv/work/uc2/runs/r2"), help="run with the rated trial export")
    ap.add_argument("--flutter-root", required=True); ap.add_argument("--seeds", action="append", required=True)
    ap.add_argument("--ratings", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--w-theme", dest="w_theme", type=float, default=1.0); ap.add_argument("--w-novel", dest="w_novel", type=float, default=0.5)
    ap.add_argument("--w-pick", dest="w_pick", type=float, default=0.5)
    ap.add_argument("--theme-floor", dest="theme_floor", type=float, default=1.0)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
