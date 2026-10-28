"""Učený gate (krok 3) za běhu: ONNX (LightGBM) nad příznaky z gate.features → p(nepříjemné).

  lg = LearnedGate("/cold/uc24/learned/sfxgate_signal.onnx")   # + .json se seznamem příznaků a prahem
  lg.predict(gate_result["features"], kind="sfx") → {"p_unpleasant": 0.97, "flag": True, "threshold": 0.9967}

Jen onnxruntime + numpy (CPU) — tohle je varianta pro audio službu AiStacku na Sparku. Práh je konformní
(recall ≥ 0.95 na kalibrační sadě); dokud model nebyl přetrénován na Olinových hodnoceních, platí jen
pro syntetické vady (viz learned.py).
"""
from __future__ import annotations

import json

import numpy as np


class LearnedGate:
    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        meta = json.load(open(onnx_path.replace(".onnx", ".json")))
        self.features, self.threshold, self.labels = meta["features"], meta["threshold"], meta.get("labels")

    def predict(self, feats: dict, kind: str = "sfx") -> dict:
        row = [feats[f] if not f.startswith("kind_") else float(f == f"kind_{kind}") for f in self.features]
        out = self.sess.run(None, {"x": np.asarray([row], np.float32)})
        p = float(np.asarray(out[1])[0, 1]) if len(out) > 1 else float(np.asarray(out[0]).ravel()[0])
        return {"p_unpleasant": round(p, 4), "flag": p >= self.threshold, "threshold": self.threshold, "labels": self.labels}
