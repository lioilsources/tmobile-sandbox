#!/usr/bin/env bash
# UC9 ostrá dávka: profil L (512 sekvencí) → extract.py nad všemi smlouvami, s GPU logem. Lze pustit znovu (navazuje).
#   run_full.sh [run=full]
set -euo pipefail
cd "$(dirname "$0")"; R=${1:-full}
export L_SEQS=512 L_BATCHTOK=16384 L_GPU=0.90
[ "$(cat /srv/work/.gpu_profile 2>/dev/null)" = "L seqs=512" ] || ../50_vllm/switch.sh L >/dev/null
read -r PID CSV < <(../bench/gpu_log.sh uc9_$R)
/srv/work/uc9/venv/bin/python extract.py --out /srv/work/uc9/runs/$R --concurrency 768 | tee -a /srv/work/uc9/runs/$R.log | tail -1
kill $PID
N=$(jq .smluv /srv/work/uc9/runs/$R/report.json)
../bench/summarize.py "$CSV" --label UC9_$R --extra "{\"smluv\": $N}" --merge /srv/work/uc9/runs/$R/report.json
