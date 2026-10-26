#!/usr/bin/env bash
# Noc po UC2: DreamerV3 (size50m) na Kirianu, kolo 2 „vystřílej všechno“, celá GPU, 10 h.
set -uo pipefail
R=~/tmobile-sandbox; Q=/srv/work/queue; D=/srv/work/uc4/dreamer
log() { echo "$(date -Is) $*" | tee -a $Q/night4.log; }
until grep -q DAY3_DONE $Q/day3.log 2>/dev/null; do sleep 60; done
$R/50_vllm/switch.sh off >/dev/null; echo dreamer > /srv/work/.gpu_profile
cd $R/uc4
read -r PID CSV < <(../bench/gpu_log.sh uc4_dreamer)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONPATH=$D/dreamerv3:$PWD timeout 10h $D/venv/bin/python kirian_dreamer.py \
  --configs kirian size50m --logdir $D/runs/d1 > $D/d1.log 2>&1
log "DreamerV3 exit $?"
kill $PID
N=$(cat $D/runs/d1/env*/episodes.jsonl 2>/dev/null | wc -l)
../bench/summarize.py "$CSV" --label UC4_dreamer --extra "{\"epizod\": $N}" | tee $D/runs/d1/gpu.json
echo off > /srv/work/.gpu_profile
log NIGHT4_DONE
