#!/usr/bin/env bash
# Spustí na pozadí `nvidia-smi dmon` (1 s) → /srv/work/bench/logs/<label>_<ts>.csv a vypíše PID.
#   pid=$(bench/gpu_log.sh uc1_qwen3vl8b) ; …benchmark… ; kill $pid ; bench/summarize.py <csv> --label uc1_qwen3vl8b
set -euo pipefail
L=${1:?label}; D=/srv/work/bench/logs; mkdir -p $D
F=$D/${L}_$(date +%Y%m%d-%H%M%S).csv
nohup nvidia-smi dmon -s pucvmt -d 1 -o T > "$F" 2>/dev/null &
echo "$! $F"
