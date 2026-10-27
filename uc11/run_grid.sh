#!/usr/bin/env bash
# UC11: spustí grid runner s GPU logem (nvidia-smi dmon) a vzorkováním vLLM /metrics.
#   run_grid.sh <label> <specs.jsonl> <results.jsonl> [parallel] [limit]
# Pouštět v tmux na VM z adresáře /srv/work/w2/uc11 (bin/grid, prompts/, env.sh, venv/).
set -euo pipefail
L=${1:?label}; IN=${2:?specs}; OUT=${3:?results}; PAR=${4:-32}; LIM=${5:-0}
cd "$(dirname "$0")/.."
. ./env.sh
mkdir -p logs
read -r GPID GCSV < <(~/tmobile-sandbox/bench/gpu_log.sh "uc11_$L")
VLLM=${LLM_BASE_URL:-http://127.0.0.1:8000}
MET=logs/${L}_vllm_metrics.csv
echo "ts,running,waiting,kv_usage,gen_tokens_total,prompt_tokens_total,preemptions" > "$MET"
( while true; do
    m=$(curl -s "$VLLM/metrics" || true)
    g() { echo "$m" | awk -v k="$1" '$1 ~ "^"k"[{ ]" {print $2; exit}'; }
    echo "$(date +%s),$(g vllm:num_requests_running),$(g vllm:num_requests_waiting),$(g vllm:kv_cache_usage_perc),$(g vllm:generation_tokens_total),$(g vllm:prompt_tokens_total),$(g vllm:num_preemptions_total)" >> "$MET"
    sleep 10
  done ) &
MPID=$!
trap 'kill $GPID $MPID 2>/dev/null || true' EXIT
echo "gpu_log: $GCSV  vllm metrics: $MET"
./bin/grid -in "$IN" -out "$OUT" -parallel "$PAR" -limit "$LIM" -prompts "${GRID_PROMPTS:-prompts}" -stats "logs/${L}_stats.json" 2>&1 | tee -a "logs/${L}.log"
kill $GPID $MPID 2>/dev/null || true
python3 ~/tmobile-sandbox/bench/summarize.py "$GCSV" --label "uc11_$L" --merge "logs/${L}_stats.json" > "logs/${L}_gpu_summary.txt" 2>&1 || true
cp "$GCSV" "logs/${L}_gpu.csv"
cat "logs/${L}_gpu_summary.txt"
