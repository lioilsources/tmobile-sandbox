"""sfxgate jako HTTP sidecar (stdlib, bez frameworku) — kontrakt, který volá Storyteller
`internal/sfxgate` (Go) a později audio služba AiStacku:

  GET  /healthz    → {"ok": true, "version": "..."}
  POST /v1/gate    {"audio_b64": <WAV/FLAC bytes b64>, "prompt": "...", "kind": "sfx"|"music"|"ambient"}
                   → {"pass","verdict","score","reasons":[...],"features":{...},"ms"}
  POST /v1/repair  stejné tělo → {"audio_b64": WAV, "accepted", "clap_before", "clap_after", "gate": {...}}

  python -m sfxgate.server --port 8096   (8095 = AiStack audio-sfx, nekolidovat) [--clap laion/larger_clap_general] [--device cpu]

S --clap se do /v1/gate přidá důvod "clap_mismatch" (cos < --clap-min) a /v1/repair hlídá,
že čištění nesebralo obsah. Bez --clap je to čistě signálová brána (CPU, ms/klip).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf

from .gate import VERSION, gate
from .repair import repair

STATE = {"clap": None, "clap_min": 0.15}


def _decode(b64: str):
    x, sr = sf.read(io.BytesIO(base64.b64decode(b64)), dtype="float32", always_2d=False)
    return x, sr


def _wav_b64(y: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def do_gate(body: dict) -> dict:
    x, sr = _decode(body["audio_b64"])
    kind = body.get("kind") or "sfx"
    g = gate(x, sr, kind)
    clap, prompt = STATE["clap"], body.get("prompt")
    if clap is not None and prompt:
        t0 = time.perf_counter()
        m = x.mean(axis=1) if x.ndim == 2 else x
        c = float(clap(m, sr, prompt))
        g["features"]["clap"] = round(c, 4)
        if c < STATE["clap_min"]:
            g["reasons"].append({"code": "clap_mismatch", "metric": "clap", "value": round(c, 4),
                                 "threshold": STATE["clap_min"], "stage": "semantic", "severity": "reject",
                                 "message": f"CLAP {c:.2f}: zvuk neodpovídá popisu '{prompt[:60]}'"})
            g["verdict"], g["pass"] = "reject", False
            g["score"] = round(1 - (1 - g["score"]) * 0.4, 4)
        g["ms"] = round(g["ms"] + (time.perf_counter() - t0) * 1000, 3)
    return g


def do_repair(body: dict) -> dict:
    x, sr = _decode(body["audio_b64"])
    r = repair(x, sr, body.get("kind") or "sfx", body.get("prompt"), STATE["clap"])
    return {"audio_b64": _wav_b64(r["audio"], sr), "accepted": r["accepted"], "steps": r["steps"],
            "clap_before": r["clap_before"], "clap_after": r["clap_after"], "gate": r["gate"]}


class H(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/healthz":
            return self._send(200, {"ok": True, "version": VERSION, "clap": STATE["clap"] is not None})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        fn = {"/v1/gate": do_gate, "/v1/repair": do_repair}.get(self.path)
        if fn is None:
            return self._send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            self._send(200, fn(body))
        except (KeyError, ValueError, RuntimeError) as e:
            self._send(400, {"error": str(e)})

    def log_message(self, fmt, *args):  # tichý access log
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8096)
    ap.add_argument("--clap", default=None, help="HF id CLAP modelu (volitelné)")
    ap.add_argument("--clap-min", type=float, default=0.15)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.clap:
        from .clap import Clap
        STATE["clap"] = Clap(a.clap, a.device).score
        STATE["clap_min"] = a.clap_min
    print(f"sfxgate {VERSION} na http://{a.host}:{a.port} (clap={a.clap})", flush=True)
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
