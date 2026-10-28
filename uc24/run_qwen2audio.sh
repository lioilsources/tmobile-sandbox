#!/usr/bin/env bash
# UC24 krok 2 — Qwen2-Audio-7B soudce na GPU (queue-ready, spouští Director v okně).
# Potřeba: ≥ 21 GB volné VRAM (bf16 16.5 GB + batch 8). Neřeší profily — jen běží, když je místo.
#   bash ~/tmobile-sandbox/uc24/run_qwen2audio.sh [limit=400]
# → /cold/uc24/judges/qwen2audio.jsonl + /srv/work/bench/results/UC24_qwen2audio.json (s Wh z gpu_log)
set -euo pipefail
cd "$(dirname "$0")"
N=${1:-400}
export HF_HOME=/srv/hf HF_HUB_OFFLINE=1
PY=/cold/venvs/uc24/bin/python; W=/srv/work/w2/uc24; mkdir -p $W /srv/work/bench/results
read -r PID CSV < <(../bench/gpu_log.sh uc24_qwen2audio)
$PY qwen2audio_judge.py --split test --limit "$N" --out /cold/uc24/judges/qwen2audio.jsonl | tee $W/qwen2audio_run.json || { kill $PID; exit 1; }
kill $PID
../bench/summarize.py "$CSV" --label UC24_qwen2audio --merge $W/qwen2audio_run.json > /srv/work/bench/results/UC24_qwen2audio.json
$PY eval_judges.py --qwen /cold/uc24/judges/qwen2audio.jsonl --out /srv/work/bench/results/UC24_judges.json >/dev/null
echo "HOTOVO (uc24 qwen2audio)"
