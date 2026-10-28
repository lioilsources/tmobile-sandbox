#!/usr/bin/env bash
# UC25: vlastní vLLM instance Qwen3-VL-8B-Instruct pro strukturovanou extrakci (guided JSON).
# SPOUŠTÍ DIRECTOR (vlastník GPU), ne sub-agent. Port 8004 (8000 vllm, 8001 whisper, 8002 embed, 8003 coder).
#
#   ./vllm_vl8b.sh up [GPU_UTIL]     # default 0.80 = sám na GPU (~37 GB; váhy BF16 ~17 GB, zbytek KV)
#   ./vllm_vl8b.sh down
#
# Guided JSON: eval_vlm.py posílá OpenAI `response_format: {"type":"json_schema", ...}` se schématem
# z taxo/schema/vlm_extract.schema.json; vLLM ho vynutí přes structured outputs (backend auto → xgrammar),
# nic dalšího se nezapíná. Obrázek ≤ 896 px (~1k vizuálních tokenů), max_tokens 512 → 8k kontext stačí.
# Ekvivalent pro 50_vllm/compose.yaml (profil V):
#   vl8b-v: { <<: *vllm, profiles: [V], container_name: vllm-vl8b, ports: ["127.0.0.1:8004:8000"],
#             command: >- --model Qwen/Qwen3-VL-8B-Instruct --served-model-name qwen3-vl-8b
#             --max-model-len 8192 --gpu-memory-utilization ${V_GPU:-0.80} --max-num-seqs 64
#             --limit-mm-per-prompt '{"image":1,"video":0}' --mm-processor-kwargs '{"max_pixels":802816}'
#             --enable-prefix-caching }
set -euo pipefail
case "${1:-up}" in
  up)
    docker run -d --name vllm-vl8b --gpus all --ipc host --restart no \
      -p 127.0.0.1:8004:8000 -v /srv/hf:/root/.cache/huggingface -v /cold:/cold:ro -e HF_HUB_OFFLINE=1 \
      vllm/vllm-openai:latest \
      --model Qwen/Qwen3-VL-8B-Instruct --served-model-name qwen3-vl-8b \
      --max-model-len 8192 --gpu-memory-utilization "${2:-0.80}" --max-num-seqs 64 \
      --limit-mm-per-prompt '{"image":1,"video":0}' --mm-processor-kwargs '{"max_pixels":802816}' \
      --enable-prefix-caching
    until curl -sf localhost:8004/health >/dev/null; do sleep 10; done
    curl -s localhost:8004/v1/models | head -c 300; echo ;;
  down) docker rm -f vllm-vl8b ;;
  *) echo "up|down"; exit 1 ;;
esac
