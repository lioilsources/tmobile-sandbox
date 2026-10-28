#!/usr/bin/env bash
# UC25: vlastní vLLM instance Qwen3-VL-8B-Instruct pro strukturovanou extrakci (guided JSON).
# SPOUŠTÍ DIRECTOR (vlastník GPU), ne sub-agent. Port 8004 (8000 vllm, 8001 whisper, 8002 embed, 8003 coder).
#
#   ./vllm_vl8b.sh up [GPU_UTIL]     # default 0.80 = sám na GPU (~37 GB; váhy BF16 ~17 GB, zbytek KV)
#   ./vllm_vl8b.sh logs              # docker logs → $UC25_OUT/vllm-vl8b.<čas>.log (default out/gpu)
#   ./vllm_vl8b.sh down              # uloží docker logs a smaže kontejner
#
# Guided JSON: eval_vlm.py posílá OpenAI `response_format: {"type":"json_schema", ...}` se schématem
# z taxo/schema/vlm_extract.schema.json; vLLM ho vynutí přes structured outputs (backend auto → xgrammar),
# nic dalšího se nezapíná. Obrázek ≤ 896 px = 784 vizuálních tokenů; celý prompt změřen 1 506 tokenů (Qwen3-VL procesor),
# max_tokens 1024 (nejdelší možná odpověď ~350) → 8k kontext stačí.
# Ekvivalent pro 50_vllm/compose.yaml (profil V):
#   vl8b-v: { <<: *vllm, profiles: [V], container_name: vllm-vl8b, ports: ["127.0.0.1:8004:8000"],
#             command: >- --model Qwen/Qwen3-VL-8B-Instruct --served-model-name qwen3-vl-8b
#             --max-model-len 8192 --gpu-memory-utilization ${V_GPU:-0.80} --max-num-seqs 64
#             --limit-mm-per-prompt '{"image":1,"video":0}' --mm-processor-kwargs '{"max_pixels":802816}'
#             --enable-prefix-caching }
set -euo pipefail
O=${UC25_OUT:-/srv/work/w2/uc25/out/gpu}
# docker logs kontejneru → $O/vllm-vl8b.<čas>.log (+ symlink vllm-vl8b.log na poslední); před každým down
save_logs() {
  mkdir -p "$O"
  docker inspect vllm-vl8b >/dev/null 2>&1 || { echo "vllm-vl8b neexistuje — žádné logy"; return 0; }
  local f="$O/vllm-vl8b.$(date +%Y%m%dT%H%M%S).log"
  docker logs --timestamps vllm-vl8b > "$f" 2>&1 || true
  ln -sfn "$(basename "$f")" "$O/vllm-vl8b.log"
  echo "docker logs → $f ($(wc -l < "$f") řádků)"
}
case "${1:-up}" in
  up)
    if docker inspect vllm-vl8b >/dev/null 2>&1; then
      [ "$(docker inspect -f '{{.State.Running}}' vllm-vl8b)" = true ] && { echo "vllm-vl8b už běží"; exit 0; }
      save_logs; docker rm vllm-vl8b >/dev/null   # zastavený zbytek po minulém běhu
    fi
    docker run -d --name vllm-vl8b --gpus all --ipc host --restart no \
      -p 127.0.0.1:8004:8000 -v /srv/hf:/root/.cache/huggingface -v /cold:/cold:ro -e HF_HUB_OFFLINE=1 \
      vllm/vllm-openai:latest \
      --model Qwen/Qwen3-VL-8B-Instruct --served-model-name qwen3-vl-8b \
      --max-model-len 8192 --gpu-memory-utilization "${2:-0.80}" --max-num-seqs 64 \
      --limit-mm-per-prompt '{"image":1,"video":0}' --mm-processor-kwargs '{"max_pixels":802816}' \
      --enable-prefix-caching
    # čekej na health, ale skonči, když kontejner spadne nebo to trvá > UP_TIMEOUT s (default 900)
    t0=$SECONDS
    until curl -sf localhost:8004/health >/dev/null; do
      if [ "$(docker inspect -f '{{.State.Running}}' vllm-vl8b 2>/dev/null)" != true ]; then
        echo "vllm-vl8b neběží (spadl při startu)"; save_logs; exit 3; fi
      (( SECONDS - t0 > ${UP_TIMEOUT:-900} )) && { echo "health timeout ${UP_TIMEOUT:-900} s"; save_logs; exit 4; }
      sleep 10
    done
    echo "health OK po $((SECONDS - t0)) s"
    curl -s localhost:8004/v1/models | head -c 300; echo ;;
  logs) save_logs ;;
  down) save_logs; docker rm -f vllm-vl8b ;;
  *) echo "up|logs|down"; exit 1 ;;
esac
