#!/usr/bin/env python3
"""Zpracování nahrávek jako v produkci: stereo WAV → Whisper po kanálech (operátor / zákazník) → dialog podle
časových značek → LLM analýza (json_schema) → JSON na hovor.

  process.py --audio /srv/work/demo1/audio --out /srv/work/demo1/runs/<run> --concurrency 8
"""
import argparse, asyncio, io, json, pathlib, subprocess, time
import httpx
from common import LLM, STT, WHISPER_MODEL, LABELS, BOOLS, DEF_SENTIMENT, DEF_KROK

SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    **{k: {"enum": v} for k, v in LABELS.items()}, **{k: {"type": "boolean"} for k in BOOLS},
    "shrnuti": {"type": "string"}}}
SCHEMA["required"] = list(SCHEMA["properties"])
PROMPT = f"""Jsi analytik kvality zákaznické linky mobilního operátora. Dostaneš automatický přepis hovoru
(může obsahovat chyby rozpoznávání řeči). Vyplň:
- kategorie: hlavní téma hovoru
  (fakturace = vyúčtování, platby, nečekané položky; zmena_tarifu; roaming = služby v zahraničí; technicka_zavada = signál, výpadek, internet;
   vypoved_smlouvy = zákazník chce ukončit smlouvu; prenos_cisla = přenos čísla, OKU; zarizeni_a_pojisteni = telefon, reklamace, pojištění, splátky;
   sim_a_esim = SIM karta, eSIM, PUK)
- sentiment_zakaznika: nálada zákazníka NA KONCI hovoru – {"; ".join(f"{k} = {v}" for k, v in DEF_SENTIMENT.items())}
- vyreseno: je zákazníkův problém během hovoru vyřešen (ne jen předán dál nebo odložen)?
- riziko_odchodu: zmínil zákazník odchod ke konkurenci nebo zrušení smlouvy?
- overeni_identity: true JEN tehdy, když operátorka výslovně požádá o ověřovací údaj (datum narození, heslo, PIN, část čísla dokladu)
  a zákazník ho sdělí. Pouhé vyhledání účtu nebo telefonního čísla, nahlédnutí do smlouvy či oslovení jménem NENÍ ověření identity.
- dalsi_krok: jediný dohodnutý další krok – {"; ".join(f"{k} = {v}" for k, v in DEF_KROK.items())}
- shrnuti: 1–2 věty česky pro CRM."""

def channel(wav, ch):  # 0 = operátor (L), 1 = zákazník (R) → mono WAV 16 kHz v paměti
    return subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(wav), "-af", f"pan=mono|c0=c{ch}", "-ar", "16000", "-f", "wav", "-"],
                          check=True, capture_output=True).stdout

async def stt(cl, audio):
    r = await cl.post(STT, files={"file": ("a.wav", audio, "audio/wav")},
                      data={"model": WHISPER_MODEL, "language": "cs", "response_format": "verbose_json", "temperature": "0",
                            "vad_filter": "true"})   # bez VAD Whisper v tichu druhého kanálu halucinuje opakující se fráze
    r.raise_for_status(); return r.json()["segments"]

async def one(cl, sem_stt, sem_llm, wav, out, model):
    t0 = time.time()
    async with sem_stt:
        a = await asyncio.to_thread(channel, wav, 0); b = await asyncio.to_thread(channel, wav, 1)
        so, sz = await asyncio.gather(stt(cl, a), stt(cl, b))
    t1 = time.time()
    seg = sorted([("Operátor", s["start"], s["text"].strip()) for s in so] + [("Zákazník", s["start"], s["text"].strip()) for s in sz], key=lambda x: x[1])
    dialog, prev = [], None
    for who, _, txt in seg:   # sloučit navazující segmenty stejného mluvčího
        if dialog and who == prev: dialog[-1] += " " + txt
        else: dialog.append(f"{who}: {txt}")
        prev = who
    transcript = "\n".join(dialog)
    async with sem_llm:
        r = await cl.post(f"{LLM}/chat/completions", json={"model": model, "temperature": 0, "max_tokens": 600,
            "response_format": {"type": "json_schema", "json_schema": {"name": "analyza", "schema": SCHEMA}},
            "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": transcript}]})
        r.raise_for_status(); j = r.json()
    t2 = time.time()
    res = {"id": wav.stem, "stt_s": round(t1 - t0, 2), "llm_s": round(t2 - t1, 2), "usage": j["usage"],
           "prepis": {"operator": " ".join(s["text"].strip() for s in so), "zakaznik": " ".join(s["text"].strip() for s in sz)},
           "dialog": transcript, "analyza": json.loads(j["choices"][0]["message"]["content"])}
    (out / f"{wav.stem}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"{wav.stem}  stt {t1 - t0:5.1f}s  llm {t2 - t1:4.1f}s  {res['analyza']['kategorie']}", flush=True)

async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--audio", default="/srv/work/demo1/audio"); ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=8); a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    wavs = sorted(pathlib.Path(a.audio).glob("call_*.wav"))
    async with httpx.AsyncClient(timeout=900) as cl:
        model = (await cl.get(f"{LLM}/models")).json()["data"][0]["id"]
        s1, s2 = asyncio.Semaphore(a.concurrency), asyncio.Semaphore(a.concurrency * 2); t0 = time.time()
        await asyncio.gather(*(one(cl, s1, s2, w, out, model) for w in wavs))
        wall = time.time() - t0
    audio_s = sum(json.loads((pathlib.Path(a.audio) / f"{w.stem}.json").read_text())["delka_s"] for w in wavs)
    (out / "_run.json").write_text(json.dumps({"model": model, "whisper": WHISPER_MODEL, "hovoru": len(wavs), "souběh": a.concurrency,
                                               "wall_s": round(wall, 1), "audio_s": round(audio_s, 1)}, indent=1))
    print(f"\n{len(wavs)} hovorů, {audio_s / 60:.1f} min audia za {wall:.0f} s → {audio_s / wall:.1f}× reálný čas")

if __name__ == "__main__": asyncio.run(main())
