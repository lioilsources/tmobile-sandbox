#!/usr/bin/env bash
# Build AiStack audio-tts (CPU) pro x86, stažení českých hlasů, start. Idempotentní.
set -euo pipefail
cd "$(dirname "$0")"
A=/srv/work/AiStack
[ -d $A/.git ] || git clone -q https://github.com/lioilsources/AiStack.git $A
git -C $A pull -q --ff-only || true
docker build -q -f $A/services/audio/runtime/Dockerfile.tts-cpu -t aistack/audio-tts:latest $A/services/audio
[ -d /srv/audio ] || { sudo mkdir -p /srv/audio && sudo chown "$USER:$USER" /srv/audio; }
[ -f /srv/audio/models/piper-voices/cs/cs_CZ/kasandra/medium/cs_CZ-kasandra-medium.onnx ] || AUDIO_ROOT=/srv/audio $A/services/audio/scripts/download.sh tts-cpu
docker network inspect tms >/dev/null 2>&1 || docker network create tms >/dev/null
docker compose up -d --wait
curl -s localhost:8102/health
