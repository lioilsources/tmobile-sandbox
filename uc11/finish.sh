#!/usr/bin/env bash
# UC11: po doběhnutí (nebo kdykoli nad částečnými výsledky — specs jsou zamíchané) spočítá
# souhrn, Sobola, surrogate a throughput. Spouštět na VM z /srv/work/w2/uc11.
#   code/finish.sh [results=runs/grid30b_results.jsonl] [tag=qwen3-30b-a3b]
# Výstup: results/UC11_*.json + results/recommend_model_<tag>.json (malé, bez dat T-Mobile).
# Na Macu pak: rsync -a tcloud:/srv/work/w2/uc11/results/UC11_\* tmobile-sandbox/bench/results/
set -euo pipefail
cd "$(dirname "$0")/.."
R=${1:-runs/grid30b_results.jsonl}; TAG=${2:-qwen3-30b-a3b}
mkdir -p results
venv/bin/python code/analyze.py code/grid.yaml "$R" --out-dir results --tag "$TAG" \
  --model-json "results/recommend_model_${TAG}.json"
labels=$(ls logs/ | sed -n 's/^\(grid30b_[a-z]*\)_stats.json$/\1/p' | sort -u | tr '\n' ' ')
venv/bin/python code/throughput.py --logs logs --labels $labels --results "$R" \
  --out "results/UC11_throughput_${TAG}.json" --extra runs/ab_prompt_compare.json runs/ab_prompt_noise.json
ls -la results/
