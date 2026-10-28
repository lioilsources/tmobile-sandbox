"""UC24 krok 1 — signálové příznaky klipu (CPU, numpy/scipy, cíl <10 ms/klip).

Chrčení v telefonu má tři měřitelné příčiny (plán §9c):
  (a) energie pod ~250 Hz — malý reproduktor ji nevyzáří, ale budí v něm intermodulaci,
  (b) clipping / vysoká crest-factor špička — limiter v telefonu pak "prská",
  (c) širokopásmový šum / difuzní "mlha" nad 8 kHz (plochost spektra blízko 1).
Všechny spektrální příznaky jdou z jednoho STFT. Rychlost (první verze s pyloudnorm a
per-peak resample_poly měla 24 ms/3s klip):
  - LUFS = BS.1770-4 gating vlastní vektorovou implementací (K-filtr = koeficienty pyloudnorm;
    test drží shodu s pyloudnorm.integrated_loudness na ±0.05 LU),
  - true-peak = 4× FFT převzorkování jen okének okolo nejvyšších vzorků (najednou, 2D),
  - percentily přes np.partition, vše ve float32.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.fft as sfft
from scipy import signal

N_FFT = 2048
HOP = 1024
EPS = 1e-12
BANDS = {"low": (20, 250), "mid": (250, 2000), "high": (2000, 8000), "air": (8000, 16000)}


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.mean(axis=1) if x.ndim == 2 else x


@lru_cache(maxsize=8)
def _kweight(sr: int) -> np.ndarray:
    import pyloudnorm as pyln
    f = pyln.Meter(sr)._filters
    return np.vstack([signal.tf2sos(f["high_shelf"].b, f["high_shelf"].a),
                      signal.tf2sos(f["high_pass"].b, f["high_pass"].a)])


def lufs(x: np.ndarray, sr: int) -> float:
    """Integrated loudness (BS.1770-4: bloky 400 ms, krok 100 ms, gate -70 LUFS a -10 LU).
    Energie po 100ms pod-blocích (reshape) → 400ms bloky = součet 4 sousedních; bez cumsum."""
    y = signal.sosfilt(_kweight(sr), np.asarray(x, dtype=np.float32))
    step = int(0.1 * sr)
    nsub = max(4, -(-len(y) // step))
    if nsub * step != len(y):
        y = np.pad(y, (0, nsub * step - len(y)))
    sub = np.einsum("ij,ij->i", y.reshape(nsub, step), y.reshape(nsub, step), dtype=np.float64)
    z = (sub[:-3] + sub[1:-2] + sub[2:-1] + sub[3:]) / (4 * step)
    with np.errstate(divide="ignore"):
        lz = -0.691 + 10 * np.log10(z)
    z = z[lz >= -70.0]
    if len(z) == 0:
        return -120.0
    rel = -0.691 + 10 * np.log10(z.mean()) - 10.0
    z = z[-0.691 + 10 * np.log10(z) >= rel]
    return float(-0.691 + 10 * np.log10(z.mean())) if len(z) else -120.0


def true_peak(x: np.ndarray, top: int = 16, half: int = 32) -> float:
    """Max |x| po 4× (FFT) převzorkování okének ±half okolo `top` nejvyšších vzorků."""
    n = len(x)
    if n == 0:
        return 0.0
    if n <= 4 * half:
        return float(np.abs(signal.resample(x, 4 * n)).max())
    # kandidáti = maxima 64vzorkových bloků (argpartition nad celým klipem je na VM ~5 ms)
    nb = n // 64
    blk = np.abs(x[: nb * 64]).reshape(nb, 64)
    bmax = blk.max(axis=1)
    top = min(top, nb)
    bi = np.argpartition(bmax, -top)[-top:]
    idx = bi * 64 + blk[bi].argmax(axis=1)
    a_top = float(bmax[bi].max())
    idx = np.clip(idx, half, n - half)
    W = x[idx[:, None] + np.arange(-half, half)[None, :]] * np.hanning(2 * half + 2)[1:-1]
    S = np.fft.rfft(W, axis=1)
    up = np.fft.irfft(S, 8 * half, axis=1) * 4
    c = 4 * half
    return max(a_top, float(np.abs(up[:, c - 16:c + 16]).max()))


def _frames(x: np.ndarray) -> np.ndarray:
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    n = 1 + (len(x) - N_FFT) // HOP
    return np.lib.stride_tricks.as_strided(x, (n, N_FFT), (x.strides[0] * HOP, x.strides[0]))


@lru_cache(maxsize=1)
def _hann() -> np.ndarray:
    return np.hanning(N_FFT).astype(np.float32)


def _pct(v: np.ndarray, q: float, axis=None):
    """Rychlý percentil (nejbližší pořadí) přes np.partition."""
    v = np.asarray(v)
    if axis is None:
        v = v.ravel()
        k = int(round(q / 100 * (len(v) - 1)))
        return np.partition(v, k)[k]
    k = int(round(q / 100 * (v.shape[axis] - 1)))
    return np.take(np.partition(v, k, axis=axis), k, axis=axis)


def clicks(x: np.ndarray, sr: int, k: float = 8.0, blk: int = 512) -> int:
    """Počet impulzů: max |2. diference| v bloku 512 vz. > k × RMS 2. diference bloku a sousedů
    (rozumné zvuky mají 2. diferenci hladkou, klik je širokopásmový impulz). Klik = blok;
    sousední zasažené bloky = jedna událost."""
    d2 = np.diff(x, 2)
    nb = len(d2) // blk
    if nb < 3:
        return 0
    B = d2[: nb * blk].reshape(nb, blk)
    e = np.einsum("ij,ij->i", B, B, dtype=np.float64)
    loc = np.sqrt((e + np.concatenate([[0], e[:-1]]) + np.concatenate([e[1:], [0]])) / (3 * blk)) + 1e-9
    hit = np.abs(B).max(axis=1) > k * loc
    return int(np.count_nonzero(hit[1:] & ~hit[:-1]) + hit[0])


def spectral(x: np.ndarray, sr: int) -> dict:
    fr = _frames(x)
    S = sfft.rfft(fr * _hann(), axis=1)
    P = (S.real * S.real + S.imag * S.imag).astype(np.float32)  # (frames, bins)
    f = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    Ef = P.sum(axis=1)  # výkon rámce
    tot = Ef.sum() - P[:, 0].sum() + EPS  # bez DC binu (DC offset má vlastní příznak)
    fe_db = 10 * np.log10(Ef / (N_FFT * N_FFT * 0.375 / 2) + EPS)  # ≈ dBFS RMS rámce (Hann)
    live = fe_db > max(-60.0, fe_db.max() - 50.0)
    cum = np.cumsum(P.sum(axis=0))

    def below(hz):
        return float((cum[np.searchsorted(f, hz) - 1] - cum[0]) / tot)

    out = {"low_ratio_250": below(250), "sub_ratio_80": below(80), "hf_ratio_8k": 1.0 - below(8000)}
    Pl = P[live] if live.any() else P
    for name, (lo, hi) in BANDS.items():
        i0, i1 = np.searchsorted(f, lo), np.searchsorted(f, min(hi, sr / 2))
        if i1 - i0 < 2:
            out[f"flat_{name}"] = 0.0
            out[f"floor_{name}_db"] = -120.0
            continue
        B = Pl[:, i0:i1] + EPS
        flat = np.exp(np.log(B).mean(axis=1)) / B.mean(axis=1)
        out[f"flat_{name}"] = float(np.median(flat))
        # stacionární podlaha pásma vůči hlasitým rámcům celého klipu: p10(výkon pásma) / p95(výkon
        # rámce). Čisté SFX s tichem mezi údery → -80 dB a níž; hiss v SFX → -20 až -50; déšť → ~-5.
        # (Relativně k vlastnímu pásmu by to nešlo: pásmo bez obsahu má "podlahu" z kvantizace.)
        Eb = P[:, i0:i1].sum(axis=1)
        out[f"floor_{name}_db"] = float(10 * np.log10((_pct(Eb, 10) + EPS) / (_pct(Ef, 95) + EPS)))
    valid = fe_db[fe_db > -100]
    out["noise_floor_db"] = float(_pct(valid, 5)) if len(valid) else -120.0
    out["dyn_range_db"] = float(_pct(valid, 95) - _pct(valid, 5)) if len(valid) else 0.0
    q = np.flatnonzero(live)
    hop_s = HOP / sr
    out["lead_silence_s"] = float(q[0] * hop_s) if len(q) else len(x) / sr
    out["tail_silence_s"] = float((len(fe_db) - 1 - q[-1]) * hop_s) if len(q) else len(x) / sr
    return out


def extract(x: np.ndarray, sr: int) -> dict:
    """Všechny příznaky kroku 1 nad originálem (x: float, mono nebo (n, ch))."""
    xs = np.asarray(x, dtype=np.float32)
    m = to_mono(xs)
    a_all = np.abs(xs)
    peak = float(a_all.max()) if a_all.size else 0.0
    rms = float(np.sqrt(np.mean(m.astype(np.float64) ** 2) + EPS))
    tp = true_peak(m)
    feats = {
        "dur_s": len(m) / sr,
        "peak_dbfs": 20 * np.log10(peak + EPS),
        "true_peak_dbtp": 20 * np.log10(tp + EPS),
        "rms_dbfs": 20 * np.log10(rms),
        "lufs": lufs(m, sr),
        "crest_db": 20 * np.log10((peak + EPS) / rms),
        "dc_offset": float(m.mean()),
    }
    # clipping: vzorky na plné škále + "plochý vrchol" (ořez a pak normalizace dolů)
    full = float(np.count_nonzero(a_all >= 0.999)) / max(a_all.size, 1)
    if peak > 0.1:
        near = np.abs(m) >= 0.98 * peak
        flat = near[1:] & near[:-1] & (np.abs(np.diff(m)) < 1e-6 * peak + 1e-9)
        flat_frac = float(np.count_nonzero(flat)) / len(m)
    else:
        flat_frac = 0.0
    feats["clip_frac"] = max(full, flat_frac)
    feats["clicks_per_s"] = clicks(m, sr) / max(feats["dur_s"], 1e-3)
    feats.update(spectral(m, sr))
    return {k: float(v) for k, v in feats.items()}
