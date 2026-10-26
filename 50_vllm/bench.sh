#!/usr/bin/env bash
# Benchmark běžícího vLLM (profil A/B/C) → /srv/work/bench/
set -euo pipefail
M=$(curl -s localhost:8000/v1/models | jq -r '.data[0].id'); R=$(curl -s localhost:8000/v1/models | jq -r '.data[0].root')
mkdir -p /srv/work/bench
docker exec vllm vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
  --model "$R" --served-model-name "$M" --dataset-name random --random-input-len 1024 --random-output-len 256 \
  --num-prompts ${N:-200} --request-rate ${RATE:-8} --save-result --result-dir /tmp/bench
docker cp vllm:/tmp/bench/. /srv/work/bench/
ls -t /srv/work/bench | head -1
