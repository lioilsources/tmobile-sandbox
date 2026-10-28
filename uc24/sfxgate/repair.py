"""UC24 krok 5 — oprava místo zahození (hraniční klipy, verdikt "repair").

  ffmpeg HP 120 Hz (ambient/hudba) | 200 Hz (SFX), ořez ticha na začátku (numpy)
  → noisereduce spektrální gate (profil z nejtiššího 0.5s úseku; jen při důvodu "hiss")
  → ffmpeg loudnorm -16 LUFS, true-peak -1 dBTP (+ alimiter jako pojistka)
  → znovu gate; přijmout jen když projde a CLAP neklesl o > 0.05 (čištění nesmí sebrat obsah).

CLAP je volitelný (`clap` = callable(audio, sr, prompt) → cos); bez něj clap_* = None a rozhoduje
jen gate.
"""
from __future__ import annotations

import subprocess

import numpy as np

from .gate import gate

CLAP_MAX_DROP = 0.05


def _ffmpeg(x: np.ndarray, sr: int, af: str) -> np.ndarray:
    p = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
         "-af", af, "-ar", str(sr), "-f", "f32le", "-ac", "1", "pipe:1"],
        input=np.ascontiguousarray(x, dtype=np.float32).tobytes(), capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def _quietest(x: np.ndarray, sr: int, win_s: float = 0.5) -> np.ndarray:
    w = int(win_s * sr)
    if len(x) <= 2 * w:
        return x
    hop = w // 2
    e = [np.mean(x[i:i + w] ** 2) for i in range(0, len(x) - w, hop)]
    i = int(np.argmin(e)) * hop
    return x[i:i + w]


def clean(x: np.ndarray, sr: int, kind: str = "sfx", codes: set | None = None) -> tuple[np.ndarray, list]:
    codes = codes or set()
    steps = []
    m = x.mean(axis=1) if x.ndim == 2 else x
    m = m - m.mean()  # DC pryč předem, jinak skok na začátku rozkmitá HP a ořez ticha selže
    hp = 200 if kind == "sfx" else 120
    af = [f"highpass=f={hp}:poles=2", f"highpass=f={hp}:poles=2"]  # 4. řád (2× biquad)
    y = _ffmpeg(m, sr, ",".join(af))
    if "lead_silence" in codes:  # ořez ticha: první vzorek nad -50 dB pod špičkou, 10 ms rezerva
        idx = np.flatnonzero(np.abs(y) > np.abs(y).max() * 10 ** (-50 / 20))
        if len(idx):
            y = y[max(0, idx[0] - int(0.01 * sr)):]
    steps.append(f"HP {hp} Hz 4. řád" + (", ořez ticha" if "lead_silence" in codes else ""))
    if "hiss" in codes:
        import noisereduce as nr
        y = nr.reduce_noise(y=y, sr=sr, y_noise=_quietest(y, sr), stationary=True, prop_decrease=0.8).astype(np.float32)
        steps.append("noisereduce (stacionární, profil z nejtiššího 0.5 s)")
    y = _ffmpeg(y, sr, "loudnorm=I=-16:TP=-1.5:LRA=11,alimiter=limit=0.84:level=false")
    steps.append("loudnorm -16 LUFS / -1.5 dBTP + alimiter")
    return y, steps


def repair(x: np.ndarray, sr: int, kind: str = "sfx", prompt: str | None = None, clap=None, first: dict | None = None) -> dict:
    g0 = first or gate(x, sr, kind)
    codes = {r["code"] for r in g0["reasons"]}
    y, steps = clean(x, sr, kind, codes)
    g1 = gate(y, sr, kind)
    if g1["verdict"] == "repair":  # 2. průchod: vada skrytá pod jinou (DC "zaplní" ticho na začátku)
        y, more = clean(y, sr, kind, {r["code"] for r in g1["reasons"]})
        steps += more
        g1 = gate(y, sr, kind)
    cb = ca = None
    if clap is not None and prompt:
        m = x.mean(axis=1) if x.ndim == 2 else x
        cb, ca = float(clap(m, sr, prompt)), float(clap(y, sr, prompt))
    content_ok = cb is None or (cb - ca) <= CLAP_MAX_DROP
    return {"audio": y, "sr": sr, "accepted": bool(g1["pass"] and content_ok), "steps": steps,
            "clap_before": cb, "clap_after": ca, "content_ok": content_ok, "gate_before": g0, "gate": g1}
