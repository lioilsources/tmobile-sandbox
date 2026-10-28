"""CLAP (LAION) text↔audio kosinus — sémantická shoda klipu s popisem (krok 2) a embedding
pro učený gate (krok 3). Volitelná závislost (torch + transformers), samotná brána ji nepotřebuje.

  c = Clap("laion/larger_clap_general", "cpu"); c.score(audio, sr, "a dog barking") → 0.31
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

CLAP_SR = 48000


class Clap:
    def __init__(self, model_id: str = "laion/larger_clap_general", device: str = "cpu"):
        import torch
        from transformers import ClapModel, ClapProcessor
        self.torch = torch
        self.device = device
        self.model = ClapModel.from_pretrained(model_id).to(device).eval()
        self.proc = ClapProcessor.from_pretrained(model_id)
        self.model_id = model_id
        self._text = lru_cache(maxsize=4096)(self._text_emb)

    def _resample(self, x: np.ndarray, sr: int) -> np.ndarray:
        from scipy import signal
        x = x.mean(axis=1) if x.ndim == 2 else x
        if sr != CLAP_SR:
            from math import gcd
            g = gcd(sr, CLAP_SR)
            x = signal.resample_poly(x, CLAP_SR // g, sr // g)
        return x.astype(np.float32)

    def _text_emb(self, text: str) -> np.ndarray:
        with self.torch.inference_mode():
            t = self.proc(text=[text], return_tensors="pt", padding=True).to(self.device)
            e = self.model.get_text_features(**t)
            e = e / e.norm(dim=-1, keepdim=True)
        return e[0].float().cpu().numpy()

    def audio_emb(self, xs: list, srs: list) -> np.ndarray:
        """Normalizované audio embeddingy (batch)."""
        aud = [self._resample(x, s) for x, s in zip(xs, srs)]
        with self.torch.inference_mode():
            a = self.proc(audio=aud, sampling_rate=CLAP_SR, return_tensors="pt").to(self.device)
            e = self.model.get_audio_features(**a)
            e = e / e.norm(dim=-1, keepdim=True)
        return e.float().cpu().numpy()

    def text_emb(self, text: str) -> np.ndarray:
        return self._text(text)

    def score(self, x: np.ndarray, sr: int, prompt: str) -> float:
        return float(self.audio_emb([x], [sr])[0] @ self.text_emb(prompt))
