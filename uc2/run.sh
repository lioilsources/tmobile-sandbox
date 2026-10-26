#!/usr/bin/env bash
# UC2: evoluce shaderů MirrorBooth. LLM Qwen3-30B-A3B (profil L, 84 % VRAM ≈ 30 GB váhy + 8 GB KV; zbylých ~7 GB pro PickScore, render
# a ShaderBattle evaluátor přes EGL) → galerie nejlepších s ověřením impellerc. GPU log přes bench/gpu_log.sh.
#   run.sh [dětí na téma=2000]
set -euo pipefail
cd "$(dirname "$0")"; W=/srv/work/uc2; N=${1:-2000}; PY=/srv/work/uc1/ft-venv/bin/python
[ -d $W/MirrorBooth ] || git clone -q --depth 1 https://github.com/lioilsources/MirrorBooth.git $W/MirrorBooth
[ -d $W/selfies ] || { echo "chybí $W/selfies (make_selfies.py, profil D)"; exit 1; }
docker start swarm-evaluator >/dev/null 2>&1 || docker run -d --name swarm-evaluator --restart unless-stopped --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute -e EVALUATOR_LOG_FORMAT=text -p 127.0.0.1:8081:8081 swarm-evaluator >/dev/null
L_GPU=0.84 L_MAXLEN=12288 L_SEQS=64 ../50_vllm/switch.sh L >/dev/null
until curl -sf localhost:8000/v1/models >/dev/null; do sleep 5; done
until curl -sf localhost:8081/healthz >/dev/null; do sleep 2; done
read -r PID CSV < <(../bench/gpu_log.sh uc2_evolve)
$PY evolve.py --mirrorbooth $W/MirrorBooth --selfies $W/selfies --out $W/runs/evo1 --children $N --concurrency 48 2>&1 \
  | grep --line-buffered -v -E "Warning|W1026|warn|Loading weights" | tee $W/evo1.log
kill $PID
../bench/summarize.py "$CSV" --label UC2_evolve --extra "{\"deti\": $((N * 3))}" | tee $W/runs/evo1/gpu.json
$PY gallery.py --mirrorbooth $W/MirrorBooth --selfies $W/selfies --run $W/runs/evo1 --flutter-root /srv/work/uc4/flutter --top 8
