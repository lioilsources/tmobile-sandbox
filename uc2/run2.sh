#!/usr/bin/env bash
# UC2 kolo 2: evoluce shaderů po rodinách (evolve2.py). LLM Qwen3-30B-A3B (profil L) + ShaderBattle evaluátor.
#   [UC2_SET=r3] [CHILDREN_MAP=malba=200,...] [EXPORT_N=25] [EXPORT_CAP=0] [FACE_FLOOR=] [MAX_FLICKER=0.6] run2.sh <rodiny čárkou> <dětí na rodinu> [out=/srv/work/uc2/runs/r2]
# Zkouška: run2.sh mozaika,optika 100. Navazuje z checkpointu, když už out/<rodina>/state.json existuje.
set -euo pipefail
cd "$(dirname "$0")"; W=/srv/work/uc2; FAM=${1:?rodiny}; N=${2:-100}; OUT=${3:-$W/runs/r2}; PY=/srv/work/uc1/ft-venv/bin/python
docker start swarm-evaluator >/dev/null 2>&1 || docker run -d --name swarm-evaluator --restart unless-stopped --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute -e EVALUATOR_LOG_FORMAT=text -p 127.0.0.1:8081:8081 swarm-evaluator >/dev/null
# vLLM nejvýš 84 % VRAM (≈ 7 GB zbývá pro PickScore, render a evaluátor); dispatcher profil L spouští s vyšším výchozím podílem
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if ! curl -sf localhost:8000/v1/models >/dev/null || [ "$used" -gt 40000 ]; then
  L_GPU=0.84 L_MAXLEN=12288 L_SEQS=64 ../50_vllm/switch.sh L >/dev/null
fi
until curl -sf localhost:8000/v1/models >/dev/null; do sleep 5; done
until curl -sf localhost:8081/healthz >/dev/null; do sleep 2; done
mkdir -p "$OUT"
../bench/memguard.sh evolve2.py 1500 &
MG=$!; trap 'kill $MG 2>/dev/null' EXIT
$PY evolve2.py --mirrorbooth $W/MirrorBooth --selfies $W/selfies --flutter-root /srv/work/uc4/flutter \
  --seeds $W/MirrorBooth/mirrorbooth/shaders --seeds $W/roundb --seeds $W/seeds-kaleido ${UC2_SET:+--seeds $W/seeds-$UC2_SET} --out "$OUT" --families "$FAM" --children "$N" --children-map "${CHILDREN_MAP:-}" --export-n "${EXPORT_N:-25}" --export-cap "${EXPORT_CAP:-0}" ${FACE_FLOOR:+--face-floor "$FACE_FLOOR"} --max-flicker "${MAX_FLICKER:-0.6}" --concurrency 32 2>&1 \
  | grep --line-buffered -v -E "Warning|W1026|warn|Loading weights" | tee -a "$OUT/run.log"
