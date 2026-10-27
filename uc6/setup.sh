#!/usr/bin/env bash
# UC6: prostředí pro klonování hlasu (XTTS-v2, F5-TTS česky, Pocket TTS česky) + vyhodnocení (Whisper, ECAPA).
# Každý engine má vlastní venv (konfliktní závislosti); uv sdílí cache, takže torch se na disk nekopíruje víckrát.
set -euo pipefail
W=/srv/work/uc6; mkdir -p $W; cd $W
[ -x $W/uvenv/bin/uv ] || { python3 -m venv $W/uvenv && $W/uvenv/bin/pip -q install uv; }
UV=$W/uvenv/bin/uv; export UV_CACHE_DIR=/srv/work/.uv-cache UV_LINK_MODE=hardlink
mk() { [ -x $W/$1/bin/python ] || $UV venv -q -p 3.12 $W/$1; VIRTUAL_ENV=$W/$1 $UV pip install -q "${@:2}"; }
mk xtts-venv coqui-tts "transformers<4.58" torch==2.8.0 torchaudio==2.8.0
mk f5-venv f5-tts
mk pocket-venv pocket-tts
mk eval-venv faster-whisper speechbrain jiwer soundfile torch==2.8.0 torchaudio==2.8.0
# modely
export HF_HOME=/srv/work/.hf
$W/eval-venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download as s
for r in ["coqui/XTTS-v2", "chosenek/f5-tts-czech-model", "vvolhejn/pocket-tts-czech", "SWivid/F5-TTS",
          "speechbrain/spkrec-ecapa-voxceleb", "Systran/faster-whisper-large-v3"]:
    print(r, s(r, allow_patterns=None if r != "SWivid/F5-TTS" else ["F5TTS_v1_Base/*"]))
PY
echo UC6_SETUP_DONE
