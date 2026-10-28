"""Simulace telefonního reproduktoru — příznaky "po telefonu".

Model (fallback, dokud není změřené phone_ir.wav):
  z   = x · g        telefon na max: soubor hraje tak, jak je, jen dorovnaný na true-peak -1 dBTP
                     (aplikace i přehrávač dělají peak, ne loudness normalizaci)
  u   = softclip(z)  limiter / ochrana "smart amp" čipu: lineární do 0.8 FS, nad tím tanh koleno
                     → hutný horký signál (ořezaný, překomprimovaný) se zkreslí, řídké špičky málo
  u2  = u - e + Xe·softclip(e/Xe)   e = LP 250 Hz(u): mez výchylky membrány. Výchylku malého
                     reproduktoru dělá hlavně basa, kterou nevyzáří; nelinearita tam vzniká a
                     intermodulací leze do slyšitelného pásma (rumble → "chrčení")
  out = LP12k(HP300(u2))  (4. řád Butterworth; s phone_ir.wav konvoluce s IR)
  lin = LP12k(HP300(z))   lineární reference
  phone_dist_db = 10·log10(|out-lin|² / |lin|²),  phone_drop_db = K-vážená energie z / lin
Reproduktor je lineární, takže energie se počítají ve frekvenční oblasti (Parseval):
|out-lin|² = Σ|H|²|FFT(u2-z)|², |lin|² = Σ|H|²|FFT(z)|² — dvě FFT místo 2× sosfilt 8 sekcí
(8.7 → ~3 ms na 3s klip). Simulace běží na polovičním vzorkování (průměr dvojic vzorků),
LP pak 0.45·fs ≈ 9.9 kHz místo 12 kHz. phone_drop je bez BS.1770 gatingu (rozdíl vůči gated LUFS < 0.5 dB
u hutných klipů; u řídkých SFX je to stejně jen relativní příznak).

phone_ir.wav (sweep ze 3 telefonů mikrofonem Mac Mini) je TODO uživatele; cesta přes
env SFXGATE_PHONE_IR nebo parametr `ir`. Xe a koleno jsou odhad, ne měření.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from scipy import signal

from .features import EPS, true_peak

KNEE = 0.8
XE = 0.12  # mez "výchylky" basové složky (≈ -18 dBFS)


@lru_cache(maxsize=8)
def _sos(sr: int):
    hp = signal.butter(4, 300, "highpass", fs=sr, output="sos")
    lp = signal.butter(4, min(12000, 0.45 * sr), "lowpass", fs=sr, output="sos")
    return np.vstack([hp, lp])


@lru_cache(maxsize=8)
def _lp250(sr: int):
    return signal.butter(2, 250, "lowpass", fs=sr, output="sos")


@lru_cache(maxsize=4)
def _ir(path: str, sr: int) -> np.ndarray:
    import soundfile as sf
    h, hsr = sf.read(path, dtype="float32", always_2d=True)
    h = h.mean(axis=1)
    if hsr != sr:
        h = signal.resample_poly(h, sr, hsr).astype(np.float32)
    return h / (np.sqrt(np.sum(h * h)) + EPS)


def softclip(z: np.ndarray, knee: float = KNEE) -> np.ndarray:
    a = np.abs(z)
    over = a > knee
    if not over.any():
        return z
    y = z.copy()
    y[over] = np.sign(z[over]) * (knee + (1 - knee) * np.tanh((a[over] - knee) / (1 - knee)))
    return y


def speaker(x: np.ndarray, sr: int, ir: str | None = None) -> np.ndarray:
    ir = ir or os.environ.get("SFXGATE_PHONE_IR")
    if ir and os.path.exists(ir):
        return signal.fftconvolve(x, _ir(ir, sr), mode="full")[: len(x)].astype(np.float32)
    return signal.sosfilt(_sos(sr), x).astype(np.float32)


@lru_cache(maxsize=8)
def _grid(sr: int, ir: str | None):
    """|H|² reproduktoru a |K|² (BS.1770 K-filtr) na pevné mřížce 4097 bodů 0..sr/2."""
    from .features import _kweight
    w = np.linspace(0, sr / 2, 4097)
    if ir:
        h = _ir(ir, sr)
        H2 = np.abs(np.fft.rfft(h, 8192)) ** 2
    else:
        H2 = np.abs(signal.sosfreqz(_sos(sr), worN=w, fs=sr)[1]) ** 2
    K2 = np.abs(signal.sosfreqz(_kweight(sr), worN=w, fs=sr)[1]) ** 2
    return w, H2.astype(np.float64), K2.astype(np.float64)


def simulate(x: np.ndarray, sr: int, ir: str | None = None, tp: float | None = None, l0: float | None = None) -> dict:
    import scipy.fft as sfft
    ir = ir or os.environ.get("SFXGATE_PHONE_IR")
    ir = ir if ir and os.path.exists(ir) else None
    tp = tp if tp is not None else true_peak(x)
    if tp < 1e-6:
        return {"phone_gain_db": 0.0, "phone_dist_db": -120.0, "phone_drop_db": 0.0}
    g = 10 ** (-1 / 20) / tp
    if sr >= 44100 and len(x) >= 2:  # telefon nad ~11 kHz nic podstatného nevyzáří → půl rychlosti
        h = len(x) // 2
        x, sr = 0.5 * (x[0:2 * h:2] + x[1:2 * h:2]), sr // 2
    z = (x * g).astype(np.float32)
    u = softclip(z)
    e = signal.sosfilt(_lp250(sr), u).astype(np.float32)
    d = (u - z) + (XE * softclip(e / XE) - e)  # nelineární zbytek (řídký)
    n = sfft.next_fast_len(len(z), real=True)
    w, H2g, K2g = _grid(sr, ir)
    fq = np.fft.rfftfreq(n, 1.0 / sr)
    H2 = np.interp(fq, w, H2g)
    Z2 = np.abs(sfft.rfft(z, n)) ** 2
    D2 = np.abs(sfft.rfft(d, n)) ** 2
    lin_e = float(np.dot(H2, Z2)) + EPS
    err = float(np.dot(H2, D2))
    K2 = np.interp(fq, w, K2g)
    drop = 10 * np.log10((float(np.dot(K2, Z2)) + EPS) / (float(np.dot(K2 * H2, Z2)) + EPS))
    return {
        "phone_gain_db": float(20 * np.log10(g)),
        "phone_dist_db": float(10 * np.log10(err / lin_e + EPS)),
        # kolik hlasitosti zmizí, protože ji telefon nevyzáří (energie pod 300 Hz)
        "phone_drop_db": float(drop),
    }
