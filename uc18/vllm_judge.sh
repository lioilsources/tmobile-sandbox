#!/usr/bin/env bash
# UC18/UC19: dedicated vLLM Qwen3-VL-8B-Instruct for the judge — like uc25/vllm_vl8b.sh but 2 images/request
# (pairwise A vs B). SPOUŠTÍ DIRECTOR / score.sh. Same container name + port as UC25 (vllm-vl8b, :8004) so the
# two can never run at once.
#
#   ./vllm_judge.sh up [GPU_UTIL]   # default 0.80 = alone on the GPU (~37 GB; BF16 weights ~17 GB + KV)
#   ./vllm_judge.sh logs | down     # docker logs → $UC18_OUT/vllm-judge.<time>.log
#
# 2 × 768 px images ≈ 2 × 576 visual tokens + ~400 text → 8k context is plenty; answers ≤ ~150 tokens.
set -euo pipefail
O=${UC18_OUT:-/cold/w2/uc18}
save_logs() {
  mkdir -p "$O"
  docker inspect vllm-vl8b >/dev/null 2>&1 || { echo "vllm-vl8b does not exist — no logs"; return 0; }
  local f="$O/vllm-judge.$(date +%Y%m%dT%H%M%S).log"
  docker logs --timestamps vllm-vl8b > "$f" 2>&1 || true
  ln -sfn "$(basename "$f")" "$O/vllm-judge.log"; echo "docker logs → $f"
}
case "${1:-up}" in
  up)
    if docker inspect vllm-vl8b >/dev/null 2>&1; then
      [ "$(docker inspect -f '{{.State.Running}}' vllm-vl8b)" = true ] && { echo "vllm-vl8b already running"; exit 0; }
      save_logs; docker rm vllm-vl8b >/dev/null
    fi
    docker run -d --name vllm-vl8b --gpus all --ipc host --restart no \
      -p 127.0.0.1:8004:8000 -v /srv/hf:/root/.cache/huggingface -v /cold:/cold:ro -e HF_HUB_OFFLINE=1 \
      vllm/vllm-openai:latest \
      --model Qwen/Qwen3-VL-8B-Instruct --served-model-name qwen3-vl-8b \
      --max-model-len 8192 --gpu-memory-utilization "${2:-0.80}" --max-num-seqs 64 \
      --limit-mm-per-prompt '{"image":2,"video":0}' --mm-processor-kwargs '{"max_pixels":802816}' \
      --enable-prefix-caching
    t0=$SECONDS
    until curl -sf localhost:8004/health >/dev/null; do
      [ "$(docker inspect -f '{{.State.Running}}' vllm-vl8b 2>/dev/null)" != true ] && { echo "vllm-vl8b died at start"; save_logs; exit 3; }
      (( SECONDS - t0 > ${UP_TIMEOUT:-900} )) && { echo "health timeout"; save_logs; exit 4; }
      sleep 10
    done
    echo "health OK after $((SECONDS - t0)) s" ;;
  logs) save_logs ;;
  down) save_logs; docker rm -f vllm-vl8b ;;
  *) echo "up|logs|down"; exit 1 ;;
esac
