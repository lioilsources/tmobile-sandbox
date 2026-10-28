"""UC24 — syntetická sada se známou pravdou: čisté procedurální SFX/ambient + řezy ACE-Step
klipů z UC5 (hudba), na každý base klip 7 vad × 2 síly (mild/strong) + "mismatch" páry
(klip × cizí popis) pro sémantické soudce.

  python synth.py --out /cold/uc24/synth --ace /srv/work/comfy-out/uc5
  → <out>/wav/*.wav (16-bit, 44.1 kHz, mono) + <out>/manifest.jsonl

Řádek manifestu: {id, path, base, kind, type, prompt, defect, severity, split, seed}
split podle seedu base klipu (dev = 0–2, test = 3–5; hudba ace 1–3 dev, 4–6 test) — prahy se
ladí na dev, čísla se reportují na test. Žádná data zvenku, žádné hlasy.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import zlib

import numpy as np
import soundfile as sf
from scipy import signal

SR = 44100
DEFECTS = ["rumble", "clipping", "hiss", "haze", "clicks", "dc_offset", "lead_silence"]


def env_ad(n, a=0.005, d=0.3, sr=SR):
    t = np.arange(n) / sr
    return np.minimum(1, t / a) * np.exp(-t / d)


def bp(x, lo, hi, order=2):
    sos = signal.butter(order, [lo, min(hi, 0.45 * SR)], "bandpass", fs=SR, output="sos")
    return signal.sosfilt(sos, x)


def place(dst, src, at):
    at = int(at)
    n = min(len(src), len(dst) - at)
    if n > 0:
        dst[at:at + n] += src[:n]


# ---------- procedurální zdroje (popis = prompt pro CLAP/soudce) ----------
def s_bell(r, n):
    f0 = r.uniform(700, 1500); t = np.arange(n) / SR
    y = sum(a * np.sin(2 * np.pi * f0 * k * t + r.uniform(0, 6)) * np.exp(-t / (1.2 / k ** 0.5))
            for k, a in [(1, 1), (2.76, .5), (5.4, .3), (8.93, .15)])
    return y * np.minimum(1, t / 0.002)


def s_knock(r, n):
    y = np.zeros(n)
    for i in range(r.integers(2, 4)):
        m = int(0.25 * SR); t = np.arange(m) / SR
        k = sum(np.sin(2 * np.pi * f * t) * np.exp(-t / 0.025) for f in (r.uniform(330, 420), r.uniform(700, 900), 1500))
        k += bp(r.standard_normal(m), 800, 4000) * np.exp(-t / 0.004) * 0.5
        place(y, k, 0.3 * SR + i * r.uniform(0.25, 0.35) * SR)
    return y


def s_footsteps(r, n):
    y = np.zeros(n); step = r.uniform(0.45, 0.6)
    for i in range(int(2.8 / step)):
        m = int(0.18 * SR)
        place(y, bp(r.standard_normal(m), 500, 6000) * env_ad(m, 0.004, 0.04), (0.1 + i * step) * SR)
    return y


def s_bird(r, n):
    y = np.zeros(n)
    for i in range(r.integers(3, 7)):
        m = int(r.uniform(0.06, 0.15) * SR); t = np.arange(m) / SR
        f = r.uniform(2500, 4500) + r.uniform(-1500, 1500) * t / t[-1]
        ph = 2 * np.pi * np.cumsum(f) / SR
        place(y, np.sin(ph) * np.sin(np.pi * t / t[-1]) ** 2, r.uniform(0.05, 2.7) * SR)
    return y


def s_owl(r, n):
    y = np.zeros(n); f0 = r.uniform(380, 480)
    for i, (st, du) in enumerate([(0.2, 0.35), (0.75, 0.25), (1.15, 0.7)]):
        m = int(du * SR); t = np.arange(m) / SR
        ph = 2 * np.pi * np.cumsum(f0 * (1 + 0.02 * np.sin(2 * np.pi * 5 * t))) / SR
        h = np.sin(ph) + 0.3 * np.sin(2 * ph) + 0.1 * np.sin(3 * ph)
        place(y, h * np.sin(np.pi * t / t[-1]), st * SR)
    return y


def s_drip(r, n):
    y = np.zeros(n)
    for i in range(r.integers(3, 6)):
        m = int(0.08 * SR); t = np.arange(m) / SR
        f = r.uniform(700, 1100) * (1 + 2 * t / t[-1])
        place(y, np.sin(2 * np.pi * np.cumsum(f) / SR) * env_ad(m, 0.001, 0.02), r.uniform(0.1, 2.8) * SR)
    return y


def s_whoosh(r, n):
    m = int(r.uniform(0.8, 1.4) * SR); t = np.arange(m) / SR
    x = r.standard_normal(m); out = np.zeros(m); w = 2048; win = np.hanning(w)
    for i in range(0, m - w, w // 2):  # časově proměnný pásmový filtr přes OLA (bez skoků na hranách)
        fc = 400 + 3000 * np.sin(np.pi * (i + w / 2) / m)
        out[i:i + w] += bp(x[i:i + w] * win, fc / 1.5, fc * 1.5)
    y = np.zeros(n); place(y, out * np.sin(np.pi * t / t[-1]) ** 2, 0.4 * SR)
    return y


def s_coins(r, n):
    y = np.zeros(n)
    for i in range(r.integers(8, 16)):
        m = int(0.12 * SR); t = np.arange(m) / SR
        c = sum(np.sin(2 * np.pi * r.uniform(3000, 9000) * t) for _ in range(3)) * np.exp(-t / 0.03)
        place(y, c, r.uniform(0.2, 1.6) * SR)
    return y


def s_creak(r, n):
    m = int(r.uniform(1.2, 2.0) * SR); t = np.arange(m) / SR
    rate = r.uniform(25, 60) * (1 + 0.5 * np.sin(2 * np.pi * 0.7 * t))
    pulses = (np.diff(np.floor(np.cumsum(rate) / SR), prepend=0) > 0).astype(float)
    c = bp(pulses, 600, 3500, 4) * np.sin(np.pi * t / t[-1])
    y = np.zeros(n); place(y, c, 0.3 * SR)
    return y


def s_chime(r, n):
    y = np.zeros(n); base = r.uniform(900, 1300)
    for i, k in enumerate([1, 1.25, 1.5, 2, 2.5, 3]):
        m = int(0.9 * SR); t = np.arange(m) / SR
        place(y, np.sin(2 * np.pi * base * k * t) * env_ad(m, 0.002, 0.25), (0.1 + 0.12 * i) * SR)
    return y


def s_bark(r, n):
    y = np.zeros(n)
    for i in range(r.integers(2, 4)):
        m = int(0.18 * SR); t = np.arange(m) / SR
        f0 = r.uniform(450, 650) * (1 - 0.3 * t / t[-1])
        ph = 2 * np.pi * np.cumsum(f0) / SR
        h = sum(np.sin(k * ph) / k for k in range(1, 10))
        h = bp(h, 500, 3000) + 0.3 * bp(r.standard_normal(m), 1000, 4000)
        place(y, h * env_ad(m, 0.01, 0.08), (0.2 + i * 0.5) * SR)
    return y


def s_meow(r, n):
    m = int(r.uniform(0.6, 0.9) * SR); t = np.arange(m) / SR
    f0 = 600 + 300 * np.sin(np.pi * t / t[-1]) ** 2
    ph = 2 * np.pi * np.cumsum(f0) / SR
    h = sum(np.sin(k * ph) * (1 / k) for k in range(1, 8))
    y = np.zeros(n); place(y, bp(h, 500, 5000) * np.sin(np.pi * t / t[-1]), 0.3 * SR)
    return y


def a_wind(r, n):
    t = np.arange(n) / SR
    x = bp(np.cumsum(r.standard_normal(n)) * 0.02 + r.standard_normal(n) * 0.3, 250, 1800)
    return x * (0.6 + 0.4 * np.sin(2 * np.pi * r.uniform(0.1, 0.3) * t + r.uniform(0, 6)))


def a_rain(r, n):
    y = bp(r.standard_normal(n), 1500, 12000) * 0.25
    drops = np.zeros(n); idx = r.integers(0, n, int(r.uniform(150, 300) * n / SR)); drops[idx] = r.uniform(0.2, 1, len(idx))
    return y + bp(drops, 2000, 8000)


def a_stream(r, n):
    y = bp(r.standard_normal(n), 400, 4000) * 0.2
    for _ in range(int(40 * n / SR)):
        m = int(0.03 * SR); t = np.arange(m) / SR
        place(y, np.sin(2 * np.pi * r.uniform(400, 2000) * t) * np.sin(np.pi * t / t[-1]) * r.uniform(0.1, 0.4), r.integers(0, n))
    return y


def a_fire(r, n):
    y = bp(r.standard_normal(n), 300, 3000) * 0.08
    for _ in range(int(r.uniform(4, 10) * n / SR)):
        m = int(0.01 * SR)
        place(y, bp(r.standard_normal(m), 1000, 8000) * env_ad(m, 0.0005, 0.002) * r.uniform(0.3, 1.0), r.integers(0, n))
    return y


def a_forest(r, n):
    y = a_wind(r, n) * 0.15
    for k in range(0, n - 3 * SR, int(1.5 * SR)):
        place(y, s_bird(r, 3 * SR) * 0.5, k + r.integers(0, SR))
    return y


def a_crickets(r, n):
    t = np.arange(n) / SR; f = r.uniform(4000, 5000)
    gate = (np.sin(2 * np.pi * r.uniform(2, 4) * t) > 0.3) * (np.sin(2 * np.pi * 30 * t) > 0)
    return np.sin(2 * np.pi * f * t) * signal.sosfilt(signal.butter(2, 200, fs=SR, output="sos"), gate.astype(float))


def a_waves(r, n):
    t = np.arange(n) / SR
    x = signal.sosfilt(signal.butter(2, [300, 2500], "bandpass", fs=SR, output="sos"), r.standard_normal(n))
    return x * (0.3 + 0.7 * np.sin(np.pi * t * r.uniform(0.12, 0.2) + r.uniform(0, 3)) ** 4)


SOURCES = {
    "bell": ("sfx", s_bell, "a small bell ringing"), "knock": ("sfx", s_knock, "knocking on a wooden door"),
    "footsteps": ("sfx", s_footsteps, "footsteps on gravel"), "bird": ("sfx", s_bird, "a bird chirping"),
    "owl": ("sfx", s_owl, "an owl hooting"), "drip": ("sfx", s_drip, "water dripping"),
    "whoosh": ("sfx", s_whoosh, "a fast whoosh"), "coins": ("sfx", s_coins, "coins jingling"),
    "creak": ("sfx", s_creak, "a creaking wooden door"), "chime": ("sfx", s_chime, "a magical sparkle chime"),
    "bark": ("sfx", s_bark, "a dog barking"), "meow": ("sfx", s_meow, "a cat meowing"),
    "wind": ("ambient", a_wind, "wind blowing"), "rain": ("ambient", a_rain, "rain falling"),
    "stream": ("ambient", a_stream, "a babbling brook"), "fire": ("ambient", a_fire, "a crackling campfire"),
    "forest": ("ambient", a_forest, "birds singing in a forest"), "crickets": ("ambient", a_crickets, "crickets chirping at night"),
    "waves": ("ambient", a_waves, "ocean waves on a beach"),
}
ACE_PROMPT = "chiptune, upbeat, retro arcade, 140 bpm, synth lead"  # tagy z uc5/comfy_bench.py


def normalize(y, r):
    y = y - y.mean()
    return y / (np.abs(y).max() + 1e-12) * 10 ** (r.uniform(-12, -3) / 20)


# ---------- vady se známou silou ----------
def rms(y):
    return np.sqrt(np.mean(y ** 2) + 1e-12)


def add_defect(y, d, sev, r):
    s = sev == "strong"
    n = len(y)
    if d == "rumble":  # 30–90 Hz šum + 45 Hz tón, -6 dB (mild) / +4 dB (strong) vůči RMS klipu
        lo = signal.sosfilt(signal.butter(4, 90, fs=SR, output="sos"), r.standard_normal(n))
        lo = lo / rms(lo) * 0.7 + np.sin(2 * np.pi * r.uniform(35, 55) * np.arange(n) / SR)
        z = y + lo / rms(lo) * rms(y) * 10 ** ((4 if s else -6) / 20)
        return z / max(1, np.abs(z).max() / 0.7)
    if d == "clipping":  # ořez na plné škále (gain 3× / 8×); polovina pak normalizována dolů
        g = (8 if s else 3) * 0.95 / (np.abs(y).max() + 1e-12)
        z = np.clip(y * g, -1, 1)
        return z * 0.7 if r.random() < 0.5 else z
    if d == "hiss":  # bílý šum SNR 30 / 15 dB
        return y + r.standard_normal(n) * rms(y) * 10 ** (-(15 if s else 30) / 20)
    if d == "haze":  # šum jen nad 8 kHz SNR 25 / 12 dB ("mlha" difuzního modelu)
        h = signal.sosfilt(signal.butter(4, 8000, "highpass", fs=SR, output="sos"), r.standard_normal(n))
        return y + h / rms(h) * rms(y) * 10 ** (-(12 if s else 25) / 20)
    if d == "clicks":  # impulzy 2/s nebo 8/s, amplituda 0.3–0.8 × peak
        z = y.copy(); k = int((8 if s else 2) * n / SR)
        for i in r.integers(100, n - 100, k):
            z[i] += np.sign(r.standard_normal()) * r.uniform(0.3, 0.8) * np.abs(y).max()
        return z
    if d == "dc_offset":
        return y + (0.05 if s else 0.02)
    if d == "lead_silence":
        return np.concatenate([np.zeros(int((1.5 if s else 0.5) * SR)), y])[:n + int(1.5 * SR)]
    raise ValueError(d)


def ace_slices(ace_dir, n_per=4, dur=8.0):
    out = []
    for i, p in enumerate(sorted(glob.glob(os.path.join(ace_dir, "ace_step_*.flac")))):
        x, sr = sf.read(p, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != SR:
            x = signal.resample_poly(x, SR, sr)
        L = int(dur * SR)
        for j in range(n_per):
            st = int((5 + j * (len(x) / SR - 10) / n_per) * SR)
            out.append((i + 1, j, x[st:st + L].astype(np.float64)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/cold/uc24/synth")
    ap.add_argument("--ace", default="/srv/work/comfy-out/uc5")
    ap.add_argument("--seeds", type=int, default=6)
    a = ap.parse_args()
    wd = os.path.join(a.out, "wav"); os.makedirs(wd, exist_ok=True)
    rows = []
    bases = []
    for typ, (kind, fn, prompt) in SOURCES.items():
        for s in range(a.seeds):
            r = np.random.default_rng(zlib.crc32(typ.encode()) * 100 + s)
            n = int((3.0 if kind == "sfx" else 8.0) * SR)
            bases.append((f"{typ}-{s}", kind, typ, prompt, "dev" if s < 3 else "test", s, normalize(fn(r, n), r)))
    for ace_i, j, x in ace_slices(a.ace):
        r = np.random.default_rng(9000 + ace_i * 10 + j)
        bases.append((f"ace{ace_i}-{j}", "music", "ace_music", ACE_PROMPT, "dev" if ace_i <= 3 else "test", ace_i, normalize(x, r)))
    types = sorted({b[2] for b in bases if b[2] != "ace_music"})
    for bid, kind, typ, prompt, split, s, y in bases:
        r = np.random.default_rng(zlib.crc32(bid.encode()))
        variants = [("clean", "none", y)] + [(d, sev, add_defect(y, d, sev, r)) for d in DEFECTS for sev in ("mild", "strong")]
        for d, sev, z in variants:
            cid = f"{bid}.{d}" + ("" if d == "clean" else f".{sev}")
            path = os.path.join(wd, cid + ".wav")
            sf.write(path, np.clip(z, -1, 1).astype(np.float32), SR, subtype="PCM_16")
            rows.append({"id": cid, "path": path, "base": bid, "kind": kind, "type": typ, "prompt": prompt,
                         "defect": d, "severity": sev, "split": split, "seed": s})
        if typ != "ace_music":  # mismatch: čistý klip × popis jiného typu stejného druhu
            other = [t for t in types if t != typ and SOURCES[t][0] == kind]
            wrong = other[int(r.integers(len(other)))]
            rows.append({"id": f"{bid}.mismatch", "path": os.path.join(wd, f"{bid}.clean.wav"), "base": bid, "kind": kind,
                         "type": typ, "prompt": SOURCES[wrong][2], "prompt_type": wrong, "defect": "mismatch",
                         "severity": "strong", "split": split, "seed": s})
    with open(os.path.join(a.out, "manifest.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"{len(bases)} base klipů, {len(rows)} řádků manifestu → {a.out}")


if __name__ == "__main__":
    main()
