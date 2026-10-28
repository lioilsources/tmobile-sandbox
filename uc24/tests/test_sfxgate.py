"""sfxgate — unit testy nad syntetickými klipy se známou vadou (pytest, CPU, ~2 s).

  cd uc24 && python -m pytest -q tests
"""
import base64
import io
import json
import os
import shutil
import sys
import threading
import urllib.request

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sfxgate import features as F  # noqa: E402
from sfxgate.gate import gate  # noqa: E402

SR = 44100


def bell(dur=3.0, seed=0, peak=0.5):
    """Čistý SFX: nesouměrné parciály s exp. doznějí, začíná hned, ticho na konci."""
    r = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    f0 = r.uniform(800, 1200)
    y = sum(a * np.sin(2 * np.pi * f0 * k * t) * np.exp(-t / (0.8 / k ** 0.5)) for k, a in [(1, 1), (2.76, .5), (5.4, .3)])
    y *= np.minimum(1, t / 0.003)
    return (y / np.abs(y).max() * peak).astype(np.float32)


def codes(g):
    return {r["code"] for r in g["reasons"]}


def test_clean_passes():
    for s in range(3):
        g = gate(bell(seed=s), SR, "sfx")
        assert g["verdict"] == "pass", g["reasons"]
        assert g["score"] == 0.0


def test_lufs_matches_pyloudnorm():
    pyln = pytest.importorskip("pyloudnorm")
    r = np.random.default_rng(1)
    for x in [bell(), (r.standard_normal(SR * 2) * 0.1).astype(np.float32), bell(dur=0.3)]:
        ref = pyln.Meter(SR).integrated_loudness(np.pad(x, (0, max(0, int(0.4 * SR) + 1 - len(x)))).astype(np.float64))
        assert abs(F.lufs(x, SR) - ref) < 0.05


def test_true_peak_sees_intersample_peak():
    t = np.arange(SR) / SR
    x = (0.9 * np.sin(2 * np.pi * (SR / 4) * t + np.pi / 4)).astype(np.float32)  # vzorky na 0.636
    assert np.abs(x).max() < 0.65
    assert F.true_peak(x) > 0.85


def test_rumble_detected():
    x = bell()
    t = np.arange(len(x)) / SR
    g = gate((x + 0.3 * np.sin(2 * np.pi * 45 * t)).astype(np.float32) * 0.7, SR, "sfx")
    assert "rumble" in codes(g)
    assert any(r["message"].startswith("rumble <250Hz") for r in g["reasons"])


def test_clipping_rejected_even_after_normalising_down():
    x = np.clip(bell() * 8, -1, 1) * 0.7  # ořez a pak zeslabení → plochý vrchol
    g = gate(x.astype(np.float32), SR, "sfx")
    assert g["verdict"] == "reject" and "clipping" in codes(g)


def test_hiss_detected():
    x = bell()
    n = np.random.default_rng(2).standard_normal(len(x)) * np.sqrt(np.mean(x ** 2)) * 10 ** (-30 / 20)
    assert "hiss" in codes(gate((x + n).astype(np.float32), SR, "sfx"))


def test_clicks_detected():
    x = bell().copy()
    for i in np.random.default_rng(3).integers(1000, len(x) - 1000, 8):
        x[i] += 0.3
    assert "clicks" in codes(gate(x, SR, "sfx"))


def test_dc_offset_and_lead_silence_are_repairable():
    g = gate(bell() + 0.03, SR, "sfx")
    assert "dc_offset" in codes(g)
    g = gate(np.concatenate([np.zeros(SR), bell()]).astype(np.float32), SR, "sfx")
    assert g["verdict"] == "repair" and codes(g) == {"lead_silence"}


def test_phone_sim_flags_bass_heavy_clip():
    x = bell(peak=0.3)
    t = np.arange(len(x)) / SR
    heavy = (x + 0.6 * np.sin(2 * np.pi * 50 * t)).astype(np.float32)
    clean_f, heavy_f = gate(x, SR)["features"], gate(heavy, SR)["features"]
    assert heavy_f["phone_drop_db"] > clean_f["phone_drop_db"] + 3
    assert heavy_f["phone_dist_db"] > clean_f["phone_dist_db"] + 10


def test_gate_output_is_json_and_fast():
    gs = [gate(bell(), SR, "sfx") for _ in range(5)]  # 1. volání plní cache (K-filtr, mřížka |H|²)
    g = min(gs, key=lambda q: q["ms"])
    json.dumps(g)
    assert {"pass", "verdict", "score", "reasons", "features", "ms"} <= set(g)
    assert g["ms"] < 250  # cíl <10 ms (měří eval_gate.py); tady jen hrubá pojistka i pod zátěží VM


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg")
def test_repair_fixes_dc_and_lead_silence():
    from sfxgate.repair import repair
    x = np.concatenate([np.zeros(SR), bell()]).astype(np.float32) + 0.03
    r = repair(x, SR, "sfx")
    assert r["accepted"], r["gate"]["reasons"]
    assert abs(F.lufs(r["audio"], SR) + 16) < 1.5


def _wav_b64(x):
    b = io.BytesIO()
    sf.write(b, x, SR, format="WAV", subtype="PCM_16")
    return base64.b64encode(b.getvalue()).decode()


def test_http_contract():
    from http.server import ThreadingHTTPServer
    from sfxgate.server import H
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert json.load(urllib.request.urlopen(base + "/healthz"))["ok"]
        x = np.clip(bell() * 8, -1, 1).astype(np.float32)
        req = urllib.request.Request(base + "/v1/gate", json.dumps({"audio_b64": _wav_b64(x), "prompt": "bell", "kind": "sfx"}).encode(),
                                     {"Content-Type": "application/json"})
        g = json.load(urllib.request.urlopen(req))
        assert g["verdict"] == "reject" and g["pass"] is False
        assert {"code", "metric", "value", "threshold", "stage", "message"} <= set(g["reasons"][0])
    finally:
        srv.shutdown()


# --- κ-gate: stejné tabulky jako FinetuneGallery server/kappa_test.go ---
def test_kappa_mirrors_kappa_go():
    from kappa import table
    a = table(36, 0, 0, 8)
    assert abs(a.kappa - 1) < 5e-4 and a.gate == "pass"
    a = table(72, 0, 8, 0)  # always-up soudce: 90 % shoda, κ = 0
    assert abs(a.po - 0.9) < 5e-4 and abs(a.kappa) < 5e-4 and a.downAgree == 0 and a.gate == "fail"
    a = table(70, 5, 5, 20)
    assert abs(a.pe - 0.625) < 5e-4 and abs(a.kappa - 0.7333) < 5e-4 and abs(a.lower - 0.5765) < 5e-4 and a.gate == "pass"
    a = table(28, 2, 3, 7)  # κ .655, ale dolní mez .372 → fail
    assert abs(a.kappa - 0.6552) < 5e-4 and abs(a.lower - 0.3724) < 5e-4 and a.gate == "fail"
    assert table(31, 0, 0, 8).gate == "uncalibrated" and table(40, 0, 0, 7).gate == "uncalibrated"
    a = table(50, 0, 0, 0)
    assert a.kappa is None and a.downAgree is None and a.gate == "uncalibrated"
    assert table(0, 0, 0, 0).po is None
