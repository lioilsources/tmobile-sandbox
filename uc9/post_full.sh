#!/usr/bin/env bash
# UC9 po ostré dávce: rozhodčí Qwen3-32B (profil A) na vzorku → compare.py → leads.sqlite + summary.json.
set -euo pipefail
cd "$(dirname "$0")"; W=/srv/work/uc9
../50_vllm/switch.sh A >/dev/null
$W/venv/bin/python extract.py --out $W/runs/judge32b --ids $W/judge_ids.txt --model qwen3-32b --guided schema --no-think --concurrency 64 | tail -1
./compare.py $W/runs/full/extracted.jsonl $W/runs/judge32b/extracted.jsonl --inp $W/contracts.jsonl --out /srv/work/bench/results/UC9_judge.json
python3 build_db.py --ext $W/runs/full/extracted.jsonl
