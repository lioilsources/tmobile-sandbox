#!/usr/bin/env bash
# UC1: přepne GPU na daný OCR engine, změří přesnost + propustnost + GPU (gpu_log/summarize).
#   bench_engine.sh <label> <hf_model|profil:C> [souběh=16] [extra vLLM args] [greedy|fix]
set -euo pipefail
cd "$(dirname "$0")"; L=$1; M=$2; C=${3:-16}; X=${4:-}; DEC=${5:-greedy}
CUR=$(cat /srv/work/.gpu_profile 2>/dev/null || true)
if [ "$M" = profil:C ]; then [ "$CUR" = C ] || ../50_vllm/switch.sh C >/dev/null; URL=http://127.0.0.1:8000/v1/chat/completions; MODEL_NAME=qwen3-vl-32b
else [ "$CUR" = "O $M" ] || OCR_MODEL=$M OCR_ARGS="$X" ../50_vllm/switch.sh O >/dev/null; MODEL_NAME=ocr; fi
RUN=/srv/work/uc1/runs/$L-$DEC-c$C-$(date +%m%d-%H%M); mkdir -p $RUN
read -r PID CSV < <(../bench/gpu_log.sh uc1_$L)
/srv/work/uc1/venv/bin/python run_ocr.py --engine $L --out $RUN --concurrency $C --decoding $DEC ${MODEL_NAME:+--model-name $MODEL_NAME} > $RUN/stdout.json
kill $PID
N=$(jq .vzorku $RUN/report.json)
../bench/summarize.py "$CSV" --label UC1_${L}_${DEC}_c$C --extra "{\"pulstran\": $N}" --merge $RUN/report.json
