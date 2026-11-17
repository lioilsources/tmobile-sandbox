#!/usr/bin/env bash
# Přepnutí VRAM profilu: A (LLM) | B (API mix) | C (dokumenty) | D (media: ComfyUI + Whisper) | O (OCR engine, OCR_MODEL=…) | L (batch 30B-A3B, L_SEQS=…) | K (30B-A3B 32k, velká KV) | W (coder 30B-A3B, :8003) | S (Qwen3-8B) | off
set -euo pipefail
cd "$(dirname "$0")"
P=${1:?použití: switch.sh A|B|C|D|off}
docker network inspect tms >/dev/null 2>&1 || docker network create tms >/dev/null
docker compose --profile '*' down --remove-orphans
systemctl --user stop comfyui 2>/dev/null || true
case $P in
  A|B|C|K|L|O|W|S) docker compose --profile "$P" up -d --wait --wait-timeout 900 ;;
  D)     docker compose --profile D up -d; systemctl --user start comfyui ;;
  off)   ;;
  *)     echo "neznámý profil $P"; exit 1 ;;
esac
echo "$P${OCR_MODEL:+ $OCR_MODEL}${L_SEQS:+ seqs=$L_SEQS}" > /srv/work/.gpu_profile
"$(dirname "$0")/../70_monitoring/gpu_profile.sh" || true   # textfile metrika pro witness (ServiceDown jen u aktivního profilu)
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
