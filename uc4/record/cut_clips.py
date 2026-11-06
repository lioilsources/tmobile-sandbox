#!/usr/bin/env python3
"""UC4 záznam: z celých běhů (record.py) vybere a nastříhá 5–20s klipy pro trailer.

Čte <out>/runs/<run>/{meta.json,events.jsonl,run.json}, ke každému běhu zapíše <out>/runs/<run>/clips.json
a do <out>/clips.json souhrn všech běhů; klipy (stejný formát jako celé běhy: 1920×1080, 60 fps, H.264 12 Mbit/s,
AAC 48 kHz stereo 192 kbit/s) jdou do <out>/clips/<run>_cNN_<typ>.mp4.

  cut_clips.py --out /cold/uc4/trailer/2026-11-04 [--per-run 8] [--min-score 4] [--dry]
"""
import argparse, json, pathlib, subprocess, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from record import FMT, venc

# typ události → (skóre, před, po) v sekundách
WINDOWS = {"boss_kill": (10, 8, 4), "sector_clear": (7, 7, 3), "death": (7, 7, 4), "kill_streak": (6, 2, 2),
           "shop": (8, 1, 1.5), "boss_appear": (5, 1, 7), "near_miss": (4, 3, 2), "fleet_cleared": (4, 4, 2), "hit": (1, 2, 1)}
MIN_LEN, MAX_LEN = 5.0, 20.0


def describe(e):
    """Krátký popis (cs, en) jedné události."""
    t = e["type"]; s = e.get("sector", 0) + 1
    if t == "kill_streak":
        return f"série {e['kills']} sestřelů za pár sekund (sektor {s})", f"{e['kills']}-kill streak in seconds (sector {s})"
    if t == "fleet_cleared":
        return f"celá formace sestřelena (sektor {s})", f"entire enemy formation wiped out (sector {s})"
    if t == "sector_clear":
        if e.get("all_killed"):
            return f"sektor {s} vyčištěn bez jediného uprchlíka", f"sector {s} cleared, no enemy escaped"
        return f"sektor {s} dokončen ({e['kills']}/{e['count']} sestřelů)", f"sector {s} complete ({e['kills']}/{e['count']} kills)"
    if t == "near_miss":
        return f"těsný průlet kolem nepřátelské střely ({e['dist']:.0f} px)", f"near miss, enemy shot dodged by {e['dist']:.0f} px"
    if t == "boss_appear":
        return f"nástup bosse (sektor {s})", f"boss enters (sector {s})"
    if t == "boss_kill":
        return f"boss zničen (sektor {s})", f"boss destroyed (sector {s})"
    if t == "death":
        return f"loď zničena v sektoru {s}", f"ship destroyed in sector {s}"
    if t == "shop":
        items = [p for p in e["purchases"]]
        cs = ", ".join(f"{p['cs']} → {changed(p)}" for p in items) or "jen prohlídka"
        en = ", ".join(f"{p['en']} → {changed(p)}" for p in items) or "browsing only"
        return f"AI nakupuje v ComCenteru: {cs}", f"AI shops in the ComCenter: {en}"
    if t == "hit":
        return f"zásah za {e['dmg']:.0f} (sektor {s})", f"takes a {e['dmg']:.0f} hit (sector {s})"
    return t, t


def changed(p):
    b, a = p["before"], p["after"]
    diff = [a[k] for k in ("front", "left", "right", "gen") if a.get(k) != b.get(k) and a.get(k)]
    return ", ".join(diff) if diff else "?"


def candidates(events, t0):
    """Události → kandidáti klipů [start, end, skóre, [události]] v čase videa."""
    out = []; shop = None
    for e in events:
        t = e["wall"] - t0; e = {**e, "t": round(t, 2)}
        if e["type"] == "shop_open":
            shop = {"type": "shop", "t": e["t"], "sector": e["sector"], "purchases": [], "loadout_before": e["loadout"]}
        elif e["type"] == "purchase" and shop:
            shop["purchases"].append(e)
        elif e["type"] == "shop_close" and shop:
            shop["end"] = e["t"]; shop["loadout_after"] = e["loadout"]
            sc, pre, post = WINDOWS["shop"]
            if shop["purchases"]:
                out.append([shop["t"] - pre, min(e["t"] + post, shop["t"] - pre + MAX_LEN), sc + len(shop["purchases"]), [shop]])
            shop = None
        elif e["type"] in WINDOWS:
            sc, pre, post = WINDOWS[e["type"]]
            end = (e["end"] - t0) if e["type"] == "kill_streak" else t
            if e["type"] == "kill_streak": sc += e["kills"] / 3
            if e["type"] == "sector_clear" and e.get("all_killed"): sc += 3
            out.append([t - pre, end + post, sc, [e]])
    return out


def merge(cands, dur):
    cands.sort(key=lambda c: c[0]); out = []
    for c in cands:
        c[0] = max(0.0, c[0]); c[1] = min(dur, c[1])
        if out and c[0] <= out[-1][1] + 0.5 and max(c[1], out[-1][1]) - out[-1][0] <= MAX_LEN:
            o = out[-1]; o[1] = max(o[1], c[1]); o[2] = max(o[2], c[2]) + 1; o[3] += c[3]
        else:
            out.append(c)
    for c in out:  # min. délka 5 s, max. 20 s
        if c[1] - c[0] < MIN_LEN:
            pad = (MIN_LEN - (c[1] - c[0])) / 2; c[0] = max(0.0, c[0] - pad); c[1] = min(dur, c[0] + MIN_LEN)
        c[1] = min(c[1], c[0] + MAX_LEN)
    return out


def duration(f):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(f)], capture_output=True, text=True)
    return float(r.stdout.strip() or 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--per-run", type=int, default=8)
    ap.add_argument("--min-score", type=float, default=4); ap.add_argument("--encoder", default="x264", choices=["x264", "nvenc"])
    ap.add_argument("--dry", action="store_true"); ap.add_argument("--runs", nargs="*", help="jen tyto běhy")
    a = ap.parse_args()
    out = pathlib.Path(a.out); (out / "clips").mkdir(parents=True, exist_ok=True)
    index = []
    for rdir in sorted((out / "runs").iterdir()):
        if a.runs and rdir.name not in a.runs: continue
        if not (rdir / "meta.json").exists(): continue
        meta = json.loads((rdir / "meta.json").read_text()); video = pathlib.Path(meta["video"])
        if not video.exists() or not meta.get("video_t0_wall"): continue
        events = [json.loads(l) for l in open(rdir / "events.jsonl")]
        run = json.loads((rdir / "run.json").read_text()) if (rdir / "run.json").exists() else {}
        t0 = meta["video_t0_wall"]; dur = duration(video)
        sel = [c for c in merge(candidates(events, t0), dur) if c[2] >= a.min_score]
        sel = sorted(sorted(sel, key=lambda c: -c[2])[: a.per_run], key=lambda c: c[0])
        clips = []
        for n, (s, e, sc, evs) in enumerate(sel, 1):
            main_ev = max(evs, key=lambda x: WINDOWS.get(x["type"], (0,))[0])
            f = out / "clips" / f"{rdir.name}_c{n:02d}_{main_ev['type']}.mp4"
            descs = [describe(x) for x in evs if x["type"] != "hit"] or [describe(main_ev)]
            clip = {"file": str(f), "start": round(s, 2), "end": round(e, 2), "duration": round(e - s, 2), "score": round(sc, 1),
                    "sectors": sorted({x.get("sector", 0) for x in evs}),
                    "events": [{k: v for k, v in x.items() if k not in ("wall",)} | {"t_clip": round(x["t"] - s, 2)} for x in evs],
                    "cs": "; ".join(dict.fromkeys(d[0] for d in descs)), "en": "; ".join(dict.fromkeys(d[1] for d in descs))}
            if not a.dry:
                r = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{s:.3f}", "-i", str(video),
                                    "-t", f"{e - s:.3f}", "-map", "0:v", "-map", "0:a", "-vf", "fps=60,format=yuv420p",
                                    *venc(a.encoder), *FMT, str(f)])
                clip["ok"] = r.returncode == 0
            clips.append(clip)
        all_events = [{k: v for k, v in x.items() if k != "wall"} | {"t": round(x["wall"] - t0, 2)} for x in events]
        fps = sorted(meta.get("game_fps_playing") or [0])
        rec = {"file": str(video), "run": rdir.name, "duration": round(dur, 2), "layout": meta.get("layout"),
               "game_fps_median": fps[len(fps) // 2],  # vykreslené snímky/s; pod 60 = ve videu zdvojené snímky
               "pilot": meta.get("pilot"), "shop": meta.get("shop"), "end": run.get("end"),
               "sectors": [{k: s.get(k) for k in ("sector", "end", "kills", "count", "escaped", "t", "score", "loadout_before", "loadout_after")}
                           | {"t_start": round(s["wall_start"] - t0, 2), "t_end": round(s["wall_end"] - t0, 2)} for s in run.get("sectors", [])],
               "loadout_before": run.get("loadout_start"),
               "loadout_after": (run.get("sectors") or [{}])[-1].get("loadout_after"),
               "purchases": [{k: v for k, v in p.items() if k != "wall"} | {"t": round(p["wall"] - t0, 2)} for p in run.get("purchases", [])],
               "events": all_events, "clips": clips}
        (rdir / "clips.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        index.append(rec)
        print(f"{rdir.name}: {len(clips)} klipů, {dur:.0f} s, konec {run.get('end')}", flush=True)
    (out / "clips.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
