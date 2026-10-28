#!/usr/bin/env bash
# UC24 krok 4 — OpenEvolve nad šablonou promptu. Bez GPU pro sim; pro skutečné zvuky UC24_GEN=aistack|comfy-sao.
#   UC24_GEN=sim bash run_evolve.sh 40          # suchý běh harnessu (LLM přes vLLM :8000)
#   UC24_GEN=aistack UC24_AUDIO_URL=http://127.0.0.1:8093 bash run_evolve.sh 40   # MOSS: 50 klipů × iterace
set -euo pipefail
cd "$(dirname "$0")/evolve"
IT=${1:-40}; OUT=/cold/uc24/evolve_${UC24_GEN:-sim}
M=$(curl -s 127.0.0.1:8000/v1/models | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
sed "s/name: \"qwen3-8b\"/name: \"$M\"/" config.yaml > /tmp/uc24_evolve_$$.yaml
/cold/venvs/uc24/bin/python nothink_proxy.py --port 8009 & PROXY=$!; trap 'kill $PROXY' EXIT; sleep 1
/cold/venvs/uc24/bin/openevolve-run initial_program.py evaluator.py --config /tmp/uc24_evolve_$$.yaml \
  --iterations "$IT" --output "$OUT"
echo "HOTOVO (uc24 evolve) → $OUT"
