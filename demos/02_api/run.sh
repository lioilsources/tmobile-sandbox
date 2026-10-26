#!/usr/bin/env bash
# Demo 2: AI API pro 3 tenanty — load test (3 fáze) + vyúčtování z knihy gateway.  ./run.sh [fáze_s=120]
set -euo pipefail
cd "$(dirname "$0")"; D=/srv/work/demo2; mkdir -p $D/runs
[ -d $D/venv ] || { python3 -m venv $D/venv; $D/venv/bin/pip install -q httpx; }
[ "$(cat /srv/work/.gpu_profile 2>/dev/null)" = B ] || ../../50_vllm/switch.sh B
curl -sf localhost:8080/health >/dev/null || ../../56_api/up.sh
RUN=$D/runs/$(date +%Y%m%d-%H%M%S)
$D/venv/bin/python loadtest.py --tenants ../../56_api/tenants.json --out $RUN --phase ${1:-120}
$D/venv/bin/python billing.py --ledger $D/usage.jsonl --since $(jq .start $RUN/loadtest.json) --until $(jq .konec $RUN/loadtest.json) --out $RUN
echo "run: $RUN"
