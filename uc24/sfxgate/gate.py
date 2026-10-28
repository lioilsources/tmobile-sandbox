"""Ruční prahy brány (krok 1) → verdikt pass / repair / reject se strukturovanými důvody.

Důvod = {code, metric, value, threshold, stage, severity, message}; `message` je to, co
dostane přepisovač promptu (krok 4), např. "rumble <250Hz: 38 % energie".
severity "repair" = opravitelné řetězcem kroku 5 (HP, limiter, loudnorm, trim, noisereduce),
"reject" = regenerovat. Prahy jsou per druh zvuku: ambient (déšť, vítr) je legitimně
šumový a nízkofrekvenční, SFX ne. Prahy ladit na dev části syntetické sady (uc24/synth.py),
reportovat na test části; po Olinových hodnoceních je nahradí učený gate (krok 3).
"""
from __future__ import annotations

import time

import numpy as np

from .features import extract
from .phone import simulate

VERSION = "0.1.0"

# (metric, op, threshold, severity, code, stage, message-format)
#  op ">" = vada, když hodnota > práh.  Hodnoty: SFX / ambient / music.
_COMMON = [
    ("clip_frac", ">", 5e-4, "reject", "clipping", "orig", "clipping: {v:.2%} vzorků na plné škále / ořezaný vrchol"),
    ("true_peak_dbtp", ">", -0.5, "repair", "true_peak", "orig", "true-peak {v:+.1f} dBTP (> -1 dBTP), limiter v telefonu bude prskat"),
    ("dc_offset_abs", ">", 0.01, "repair", "dc_offset", "orig", "DC offset {v:.3f}"),
    ("lufs", "<", -38.0, "repair", "too_quiet", "orig", "příliš potichu: {v:.1f} LUFS"),
    ("dur_live_s", "<", 0.25, "reject", "silent", "orig", "skoro ticho: jen {v:.2f} s zvuku"),
]
HISS_AIR = "šum/mlha nad 8 kHz: stacionární podlaha {v:.0f} dB pod hlasitými rámci"
HISS_HIGH = "širokopásmový šum 2–8 kHz: podlaha {v:.0f} dB pod hlasitými rámci"
RUMBLE = "rumble <250Hz: {v:.0%} energie"
RUMBLE_X = "rumble <250Hz: {v:.0%} energie (obsah je hlavně basa)"
DROP = "na telefonu ztratí {v:.1f} dB hlasitosti (energie pod 300 Hz)"
DIST = "na telefonu chrčí: zkreslení {v:.1f} dB"
LEAD = "ticho na začátku {v:.2f} s"
# Prahy v1 naladěné na dev části syntetické sady (uc24/tune.py, seedy 0–2 / ACE 1–3); čísla
# v reportu jsou z test části. phone_* jsou z modelového reproduktoru (phone.py), ne z měření.
THRESHOLDS = {
    "sfx": _COMMON + [
        ("low_ratio_250", ">", 0.08, "repair", "rumble", "orig", RUMBLE),
        ("low_ratio_250", ">", 0.50, "reject", "rumble", "orig", RUMBLE_X),
        ("floor_air_db", ">", -60.0, "repair", "hiss", "orig", HISS_AIR),
        ("floor_high_db", ">", -35.0, "reject", "hiss", "orig", HISS_HIGH),
        ("clicks_per_s", ">", 1.0, "reject", "clicks", "orig", "kliky: {v:.1f}/s"),
        ("lead_silence_s", ">", 0.50, "repair", "lead_silence", "orig", LEAD),
        ("phone_dist_db", ">", -14.0, "reject", "phone_distortion", "phone", DIST),
        ("phone_drop_db", ">", 1.2, "repair", "phone_drop", "phone", DROP),
    ],
    "ambient": _COMMON + [
        ("low_ratio_250", ">", 0.14, "repair", "rumble", "orig", RUMBLE),
        ("low_ratio_250", ">", 0.60, "reject", "rumble", "orig", RUMBLE_X),
        ("flat_air", ">", 0.43, "repair", "hiss", "orig", "šum/mlha nad 8 kHz: plochost spektra {v:.2f}"),
        ("clicks_per_s", ">", 1.5, "reject", "clicks", "orig", "kliky: {v:.1f}/s"),
        ("lead_silence_s", ">", 0.30, "repair", "lead_silence", "orig", LEAD),
        ("phone_dist_db", ">", -27.0, "reject", "phone_distortion", "phone", DIST),
        ("phone_drop_db", ">", 1.0, "repair", "phone_drop", "phone", DROP),
    ],
    "music": _COMMON + [
        ("low_ratio_250", ">", 0.80, "repair", "rumble", "orig", RUMBLE),
        ("floor_air_db", ">", -21.0, "repair", "hiss", "orig", HISS_AIR),
        ("clicks_per_s", ">", 1.0, "reject", "clicks", "orig", "kliky: {v:.1f}/s"),
        ("lead_silence_s", ">", 0.30, "repair", "lead_silence", "orig", LEAD),
        ("phone_dist_db", ">", -14.0, "reject", "phone_distortion", "phone", DIST),
        ("phone_drop_db", ">", 3.6, "repair", "phone_drop", "phone", DROP),
    ],
}
SEVERITY_W = {"repair": 0.3, "reject": 0.6}


def _exceed(op: str, v: float, t: float) -> bool:
    return v > t if op == ">" else v < t


def features(x: np.ndarray, sr: int, ir: str | None = None) -> dict:
    f = extract(x, sr)
    f["dc_offset_abs"] = abs(f["dc_offset"])
    f["dur_live_s"] = max(0.0, f["dur_s"] - f["lead_silence_s"] - f["tail_silence_s"])
    m = x.mean(axis=1) if x.ndim == 2 else x
    f.update(simulate(m.astype(np.float32), sr, ir, tp=10 ** (f["true_peak_dbtp"] / 20), l0=f["lufs"]))
    return f


def judge(f: dict, kind: str = "sfx") -> dict:
    rules = THRESHOLDS.get(kind, THRESHOLDS["sfx"])
    hits: dict[tuple, dict] = {}
    for metric, op, t, sev, code, stage, msg in rules:
        v = f.get(metric)
        if v is None or not _exceed(op, v, t):
            continue
        key = (code, stage)
        prev = hits.get(key)
        if prev and prev["severity"] == "reject":
            continue  # přísnější pravidlo téhož kódu už platí
        hits[key] = {"code": code, "metric": metric, "value": round(float(v), 5), "threshold": t,
                     "stage": stage, "severity": sev, "message": msg.format(v=v)}
    reasons = list(hits.values())
    if any(r["severity"] == "reject" for r in reasons):
        verdict = "reject"
    elif reasons:
        verdict = "repair"
    else:
        verdict = "pass"
    score = 1.0 - float(np.prod([1 - SEVERITY_W[r["severity"]] for r in reasons])) if reasons else 0.0
    return {"pass": verdict == "pass", "verdict": verdict, "score": round(score, 4), "reasons": reasons}


def gate(x: np.ndarray, sr: int, kind: str = "sfx", ir: str | None = None) -> dict:
    t0 = time.perf_counter()
    f = features(x, sr, ir)
    out = judge(f, kind)
    out["features"] = {k: round(v, 5) for k, v in f.items()}
    out["kind"] = kind
    out["ms"] = round((time.perf_counter() - t0) * 1000, 3)
    out["version"] = VERSION
    return out


def gate_file(path: str, kind: str = "sfx", ir: str | None = None) -> dict:
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    return gate(x, sr, kind, ir)
