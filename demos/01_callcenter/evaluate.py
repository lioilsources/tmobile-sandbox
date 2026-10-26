#!/usr/bin/env python3
"""Vyhodnocení dema 1: WER přepisu (operátor / zákazník), přesnost analýzy po štítcích, propustnost.

  evaluate.py --calls /srv/work/demo1/calls --run /srv/work/demo1/runs/<run> [--kc-per-gpu-hour X]
"""
import argparse, collections, json, pathlib, re, statistics, unicodedata
import jiwer
from common import LABELS, BOOLS

def norm(s):
    s = unicodedata.normalize("NFC", s.lower()); s = re.sub(r"[^\w\s]", " ", s); return re.sub(r"\s+", " ", s).strip()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--calls", default="/srv/work/demo1/calls"); ap.add_argument("--run", required=True)
    ap.add_argument("--kc-per-gpu-hour", type=float); a = ap.parse_args(); run = pathlib.Path(a.run)
    meta = json.loads((run / "_run.json").read_text()); keys = list(LABELS) + BOOLS
    ok = collections.Counter(); ref = {"operator": [], "zakaznik": []}; hyp = {"operator": [], "zakaznik": []}
    conf = collections.Counter(); errs = []; n = 0; stt_t, llm_t = [], []
    for cfile in sorted(pathlib.Path(a.calls).glob("call_*.json")):
        c = json.loads(cfile.read_text()); rf = run / cfile.name
        if not rf.exists(): continue
        r = json.loads(rf.read_text()); n += 1; stt_t.append(r["stt_s"]); llm_t.append(r["llm_s"])
        for who in ref:
            ref[who].append(norm(" ".join(t["text"] for t in c["repliky"] if t["mluvci"] == who))); hyp[who].append(norm(r["prepis"][who]))
        for k in keys:
            g, p = c["labels"][k], r["analyza"][k]; ok[k] += g == p
            if g != p: errs.append({"hovor": c["id"], "stitek": k, "spravne": g, "model": p})
            if k == "kategorie": conf[(g, p)] += 1
    wer = {w: round(jiwer.wer(ref[w], hyp[w]), 4) for w in ref}
    audio_h = meta["audio_s"] / 3600; wall_h = meta["wall_s"] / 3600
    rep = {"model": meta["model"], "whisper": meta["whisper"], "hovoru": n, "audio_min": round(meta["audio_s"] / 60, 1), "soubeh": meta["souběh"],
           "wer": wer, "presnost": {k: round(ok[k] / n, 3) for k in keys},
           "presnost_prumer": round(sum(ok.values()) / (n * len(keys)), 4),
           "propustnost": {"nasobek_realneho_casu": round(meta["audio_s"] / meta["wall_s"], 1),
                           "hodin_audia_za_hodinu_gpu": round(audio_h / wall_h, 1), "hovoru_za_hodinu": round(n / wall_h)},
           "latence_s": {"stt_median": round(statistics.median(stt_t), 1), "llm_median": round(statistics.median(llm_t), 1)},
           "chyby": errs}
    if a.kc_per_gpu_hour:
        rep["kc_za_1000_hovoru"] = round(a.kc_per_gpu_hour / rep["propustnost"]["hovoru_za_hodinu"] * 1000, 1)
    (run / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "chyby"}, ensure_ascii=False, indent=1))
    print(f"chyb: {len(errs)}")

if __name__ == "__main__": main()
