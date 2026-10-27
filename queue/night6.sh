#!/usr/bin/env bash
# DreamerV3 d1, 3. běh: v noci 5 ho po 3,5 h ukončil memguard (RSS 16 GB i s replay 5e5). Pokračuje z checkpointu s replay 2e5.
set -uo pipefail
R=~/tmobile-sandbox; Q=/srv/work/queue; D=/srv/work/uc4/dreamer
log() { echo "$(date -Is) $*" | tee -a $Q/night6.log; }
$R/50_vllm/switch.sh off >/dev/null; echo dreamer > /srv/work/.gpu_profile
cd $R/uc4
read -r PID CSV < <(../bench/gpu_log.sh uc4_dreamer_d1c)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONPATH=$D/dreamerv3:$PWD timeout 4h $D/venv/bin/python kirian_dreamer.py \
  --configs kirian size50m --replay.size 2e5 --logdir $D/runs/d1 >> $D/d1.log 2>&1
log "DreamerV3 exit $?"
kill $PID
N=$(cat $D/runs/d1/env*/episodes.jsonl 2>/dev/null | wc -l)
../bench/summarize.py "$CSV" --label UC4_dreamer --extra "{\"epizod\": $N}" | tee $D/runs/d1/gpu.json
echo off > /srv/work/.gpu_profile
log NIGHT6_DONE
