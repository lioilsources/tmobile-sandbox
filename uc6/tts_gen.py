#!/usr/bin/env python3
"""UC6: usínací meditace hlasem rodiče — jeden engine, věta po větě, s měřením času.

Každý engine běží ve svém venv (XTTS-v2 / F5-TTS česky / Pocket TTS česky):
  xtts-venv/bin/python tts_gen.py --engine xtts --ref ref/voice24k.wav --out runs/xtts
Výstup: <out>/s/<i>.wav (věty, 24 kHz), <out>/meditace.wav (s pauzami), <out>/gen.json (čas generace, délka audia).
"""
import argparse, json, pathlib, re, time

import numpy as np
import soundfile as sf
import torch

SR = 24000
HERE = pathlib.Path(__file__).parent


def sentences(text):
    """[(odstavec, věta)] — věty dělí na .!? a dlouhé ještě na čárce, aby se vešly do limitu XTTS (~180 znaků)."""
    out = []
    for p, par in enumerate(t for t in text.split("\n\n") if t.strip()):
        for s in re.split(r"(?<=[.!?])\s+", par.strip()):
            while len(s) > 170 and "," in s[:170]:
                cut = s[:170].rindex(",") + 1
                out.append((p, s[:cut].strip())); s = s[cut:].strip()
            out.append((p, s))
    return out


class Xtts:
    def __init__(self, ref, ref_text):
        from huggingface_hub import snapshot_download
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts as M
        d = snapshot_download("coqui/XTTS-v2")
        cfg = XttsConfig(); cfg.load_json(f"{d}/config.json")
        self.m = M.init_from_config(cfg); self.m.load_checkpoint(cfg, checkpoint_dir=d, use_deepspeed=False); self.m.cuda().eval()
        self.lat, self.spk = self.m.get_conditioning_latents(audio_path=[ref], gpt_cond_len=30, max_ref_length=30)

    def __call__(self, text):
        o = self.m.inference(text, "cs", self.lat, self.spk, temperature=0.65, repetition_penalty=5.0, enable_text_splitting=False)
        return np.asarray(o["wav"], dtype=np.float32)


class F5:
    def __init__(self, ref, ref_text):
        from f5_tts.api import F5TTS
        # chosenek/f5-tts-czech-model, jen EMA váhy (originál nese i stav optimizéru a na GPU by zabral 3× víc)
        self.m = F5TTS(model="F5TTS_v1_Base", ckpt_file="/srv/work/uc6/f5cs_ema.pt", vocab_file="/srv/work/uc6/f5cs_vocab.txt", device="cuda")
        self.ref, self.ref_text = ref, ref_text

    def __call__(self, text):
        wav, sr, _ = self.m.infer(self.ref, self.ref_text, text, show_info=lambda *a: None, progress=None, seed=0, nfe_step=32)
        assert sr == SR
        return np.asarray(wav, dtype=np.float32)


class Pocket:
    def __init__(self, ref, ref_text):
        from pocket_tts import TTSModel
        self.m = TTSModel.load_model(config="hf://vvolhejn/pocket-tts-czech/czech.yaml@7c1fbd0acba765617749dd17f3dbddc2be791cc7")
        self.st = self.m.get_state_for_audio_prompt(ref)

    def __call__(self, text):
        return self.m.generate_audio(self.st, text).detach().float().cpu().numpy().reshape(-1)


ENGINES = {"xtts": Xtts, "f5": F5, "pocket": Pocket}


@torch.no_grad()  # ne inference_mode: Pocket TTS si do stavu (KV cache) zapisuje in-place
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=ENGINES, required=True)
    ap.add_argument("--ref", required=True, help="referenční nahrávka hlasu (wav)")
    ap.add_argument("--ref-text", default="", help="přepis reference (F5 ho potřebuje)")
    ap.add_argument("--text", default=str(HERE / "meditace.txt"))
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    (a.out / "s").mkdir(parents=True, exist_ok=True)
    t0 = time.time(); eng = ENGINES[a.engine](a.ref, a.ref_text); load_s = time.time() - t0
    eng("Dobrou noc.")  # zahřátí (kompilace kernelů, cache), do měření nejde
    rows, parts = [], []
    for i, (p, s) in enumerate(sentences(pathlib.Path(a.text).read_text())):
        torch.cuda.synchronize(); t = time.time()
        w = eng(s)
        torch.cuda.synchronize(); g = time.time() - t
        sf.write(a.out / "s" / f"{i:03d}.wav", w, SR)
        rows.append({"i": i, "odstavec": p, "text": s, "gen_s": round(g, 3), "audio_s": round(len(w) / SR, 3)})
        if parts:  # pauza mezi větami 0,6 s, mezi odstavci 1,4 s — meditace má být pomalá
            parts.append(np.zeros(int(SR * (1.4 if p != rows[-2]["odstavec"] else 0.6)), np.float32))
        parts.append(w)
        print(f"{i:3d} {g:6.2f}s → {len(w) / SR:6.2f}s  {s[:60]}", flush=True)
    sf.write(a.out / "meditace.wav", np.concatenate(parts), SR)
    gen, aud = sum(r["gen_s"] for r in rows), sum(r["audio_s"] for r in rows)
    summ = {"engine": a.engine, "vet": len(rows), "load_s": round(load_s, 1), "gen_s": round(gen, 1), "audio_s": round(aud, 1),
            "meditace_s": round(sum(len(x) for x in parts) / SR, 1), "rtf": round(gen / aud, 4),
            "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "vety": rows}
    (a.out / "gen.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "vety"}))


if __name__ == "__main__":
    main()
