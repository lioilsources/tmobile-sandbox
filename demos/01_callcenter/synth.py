#!/usr/bin/env python3
"""Namluví hovory (Piper: operátorka = kasandra, zákazník = jirka) → stereo WAV jako z call centra
(L = operátor, R = zákazník), telefonní kvalita: 8 kHz, pásmo 300–3400 Hz, G.711 μ-law.

  synth.py --calls /srv/work/demo1/calls --out /srv/work/demo1/audio
"""
import argparse, concurrent.futures as cf, io, json, pathlib, random, subprocess, tempfile
import httpx, numpy as np, soundfile as sf
from common import TTS, VOICE

def tts(text, voice):
    r = httpx.post(TTS, files={"request": (None, json.dumps({"text": text, "language": "cs", "model": voice}))}, timeout=300)
    r.raise_for_status(); a, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    return (a if a.ndim == 1 else a.mean(1)), sr

def one(cf_, out, seed):
    call = json.loads(cf_.read_text()); rng = random.Random(seed)
    segs, sr0, t = [], None, 0.6
    for rep in call["repliky"]:
        if len(rep["text"].strip()) < 2: continue
        a, sr = tts(rep["text"], VOICE[rep["mluvci"]]); sr0 = sr0 or sr
        segs.append((rep["mluvci"], t, a)); t += len(a) / sr + rng.uniform(0.25, 0.9)
    n = int((t + 0.5) * sr0); st = np.zeros((n, 2), np.float32)
    for who, start, a in segs:
        i = int(start * sr0); st[i:i + len(a), 0 if who == "operator" else 1] += a
    st += np.random.default_rng(seed).normal(0, 0.003, st.shape).astype(np.float32)   # šum linky
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, st, sr0)
        wav = out / f"{call['id']}.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp.name, "-af", "highpass=f=300,lowpass=f=3400",
                        "-ar", "8000", "-c:a", "pcm_mulaw", str(wav)], check=True)
    meta = {"id": call["id"], "delka_s": round(n / sr0, 1),
            "segmenty": [{"mluvci": w, "start": round(s, 2), "konec": round(s + len(a) / sr0, 2)} for w, s, a in segs]}
    (out / f"{call['id']}.json").write_text(json.dumps(meta, ensure_ascii=False))
    return call["id"], meta["delka_s"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--calls", default="/srv/work/demo1/calls"); ap.add_argument("--out", default="/srv/work/demo1/audio")
    a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(pathlib.Path(a.calls).glob("call_*.json")); total = 0
    with cf.ThreadPoolExecutor(4) as ex:
        for cid, d in ex.map(lambda p: one(p[1], out, p[0]), enumerate(files)):
            total += d; print(f"{cid} {d:6.1f} s", flush=True)
    print(f"{len(files)} hovorů, {total / 60:.1f} min audia")

if __name__ == "__main__": main()
