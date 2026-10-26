#!/usr/bin/env bash
# Prostředí pro demo 1 (call centrum). TTS služba: ../../55_tts/up.sh, LLM + Whisper: profil B.
set -euo pipefail
D=/srv/work/demo1; mkdir -p $D
[ -d $D/venv ] || python3 -m venv $D/venv
$D/venv/bin/pip install -q -U httpx numpy soundfile jiwer
echo ok
