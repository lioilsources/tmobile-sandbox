#!/usr/bin/env python3
"""UC6: příprava reference a vyhodnocení klonovaného hlasu (eval-venv).

  eval_tts.py ref  --ref ref/voice16k.wav --out ref/        → ref.json (přepis), clip24k.wav (≤12 s na hranici slova pro F5)
  eval_tts.py eval --ref ref/voice16k.wav --runs runs/ xtts f5 pocket  → runs/eval.json

Metriky: podobnost mluvčího (ECAPA-TDNN VoxCeleb, kosinus věty vs. celá reference), WER/CER zpětným přepisem
(Whisper large-v3, cs) proti vstupnímu textu. Strop = reference proti sobě (dvě poloviny nahrávky),
podlaha = cizí český mluvčí (vzorek z ParCzech u Pocket TTS).
"""
import argparse, glob, json, os, pathlib, re, unicodedata

import jiwer
import numpy as np
import soundfile as sf
import torch
import torchaudio


def norm(t):
    t = unicodedata.normalize("NFC", t.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()


def load16(p):
    w, sr = torchaudio.load(str(p))
    return torchaudio.functional.resample(w.mean(0, keepdim=True), sr, 16000)


def whisper():  # audio předáváme jako pole 16 kHz: faster-whisper volá PyAV s argumentem, který novější av nezná
    from faster_whisper import WhisperModel
    if os.environ.get("UC6_CPU"):  # když GPU drží jiná úloha
        return WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=4)
    return WhisperModel("large-v3", device="cuda", compute_type="float16")


def cmd_ref(a):
    segs, _ = whisper().transcribe(load16(a.ref).squeeze(0).numpy(), language="cs", word_timestamps=True, beam_size=5)
    words = [w for s in segs for w in s.words]
    full = "".join(w.word for w in words).strip()
    # F5 chce krátkou referenci: konec po slově, které končí větu nebo čárkou, mezi 6 a 12 s
    cut = max((w for w in words if 6 <= w.end <= 12 and w.word.strip()[-1:] in ".,!?"), key=lambda w: w.end, default=None) \
        or max((w for w in words if w.end <= 12), key=lambda w: w.end)
    clip_text = "".join(w.word for w in words if w.end <= cut.end).strip()
    w, sr = torchaudio.load(str(a.ref))
    clip = torchaudio.functional.resample(w[:, : int((cut.end + 0.15) * sr)], sr, 24000)
    torchaudio.save(str(a.out / "clip24k.wav"), clip, 24000)
    r = {"text": full, "clip_text": clip_text, "clip_s": round(cut.end + 0.15, 2), "ref_s": round(w.shape[1] / sr, 2)}
    (a.out / "ref.json").write_text(json.dumps(r, ensure_ascii=False, indent=1)); print(r)


@torch.inference_mode()
def cmd_eval(a):
    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cuda"},
                                           savedir="/srv/work/.hf/ecapa")
    emb = lambda w: torch.nn.functional.normalize(ecapa.encode_batch(w.cuda()).squeeze(), dim=0)
    ref = load16(a.ref); e_ref = emb(ref)
    half = ref.shape[1] // 2
    other = glob.glob("/srv/work/.hf/hub/models--vvolhejn--pocket-tts-czech/snapshots/*/voices/cs_m_zenisek.wav")[0]
    res = {"strop_reference_polovina_vs_polovina": round(float(emb(ref[:, :half]) @ emb(ref[:, half:])), 4),
           "podlaha_cizi_cesky_mluvci": round(float(emb(load16(other)) @ e_ref), 4), "engines": {}}
    wm = whisper()
    for name in a.engines:
        g = json.loads((a.runs / name / "gen.json").read_text())
        sims, hyp = [], []
        for r in g["vety"]:
            p = a.runs / name / "s" / f"{r['i']:03d}.wav"
            w = load16(p)
            if w.shape[1] > 16000 * 0.8:  # ECAPA na velmi krátkých úsecích (Hú.) je šum
                sims.append(float(emb(w) @ e_ref))
            segs, _ = wm.transcribe(w.squeeze(0).numpy(), language="cs", beam_size=5, condition_on_previous_text=False)
            hyp.append(" ".join(s.text for s in segs))
        refs = [norm(r["text"]) for r in g["vety"]]; hyps = [norm(h) for h in hyp]
        whole = float(emb(load16(a.runs / name / "meditace.wav")) @ e_ref)
        per = [{"i": r["i"], "wer": round(jiwer.wer(x, y), 3), "text": r["text"], "prepis": h}
               for r, x, y, h in zip(g["vety"], refs, hyps, hyp)]
        res["engines"][name] = {
            "spk_sim_vety_prumer": round(float(np.mean(sims)), 4), "spk_sim_vety_min": round(float(np.min(sims)), 4),
            "spk_sim_cela_meditace": round(whole, 4),
            "wer": round(jiwer.wer(" ".join(refs), " ".join(hyps)), 4), "cer": round(jiwer.cer(" ".join(refs), " ".join(hyps)), 4),
            "rtf": g["rtf"], "audio_s": g["audio_s"], "meditace_s": g["meditace_s"], "gen_s": g["gen_s"],
            "vram_peak_gb": g["vram_peak_gb"], "nejhorsi_vety": sorted(per, key=lambda x: -x["wer"])[:8]}
        print(name, {k: v for k, v in res["engines"][name].items() if k != "nejhorsi_vety"}, flush=True)
    (a.runs / "eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("ref"); r.add_argument("--ref", required=True); r.add_argument("--out", type=pathlib.Path, required=True)
    e = sp.add_parser("eval"); e.add_argument("--ref", required=True); e.add_argument("--runs", type=pathlib.Path, required=True)
    e.add_argument("engines", nargs="+")
    a = ap.parse_args(); {"ref": cmd_ref, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
