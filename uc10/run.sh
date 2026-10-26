#!/usr/bin/env bash
# UC10: galerie stylů storytelleru (FLUX-schnell, 354 motivů × 7 stylů × 4 varianty) jako dlouhý soak test GPU.
#   run.sh [variants=4]
set -euo pipefail
cd "$(dirname "$0")"; W=/srv/work/uc10; V=${1:-4}; ST=$W/storyteller
[ -d $ST ] || git clone -q --depth 1 https://github.com/lioilsources/storyteller.git $ST
[ "$(cat /srv/work/.gpu_profile 2>/dev/null)" = D ] || ../50_vllm/switch.sh D >/dev/null
until curl -sf localhost:8188/system_stats >/dev/null; do sleep 5; done
read -r PID CSV < <(../bench/gpu_log.sh uc10_styles)
/srv/work/uc1/venv/bin/python render_motifs.py --cards $ST/rag/data/cz_cards.json $ST/rag/data/cz_tpe_cards.json \
  --styles $ST/infra/seed/models_styles.sql --variants $V --out $W/styles | tee $W/styles.log | tail -1
kill $PID
N=$(jq .obrazku $W/styles/report.json)
../bench/summarize.py "$CSV" --label UC10_styles --extra "{\"obrazku\": $N}" --merge $W/styles/report.json
rm -rf /srv/work/comfy-out/uc10   # PNG originály; JPG kopie jsou v $W/styles/img
