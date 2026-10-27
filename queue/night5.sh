#!/usr/bin/env bash
# Pokračování noci 4: DreamerV3 spadl na OOM (replay 5e6 kroků s latenty size50m ≈ 17,5 GB RAM při 23 GB VM).
# Po dokončení vyhodnocení UC7 pokračuje z checkpointu v runs/d1 s replay 5e5.
set -uo pipefail
R=~/tmobile-sandbox; Q=/srv/work/queue; D=/srv/work/uc4/dreamer
log() { echo "$(date -Is) $*" | tee -a $Q/night5.log; }
while pgrep -f "^/srv/work/uc1/ft-venv/bin/python eval_lora.py" >/dev/null; do sleep 30; done
$R/50_vllm/switch.sh off >/dev/null; echo dreamer > /srv/work/.gpu_profile
cd $R/uc4
read -r PID CSV < <(../bench/gpu_log.sh uc4_dreamer_d1b)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONPATH=$D/dreamerv3:$PWD timeout 8h $D/venv/bin/python kirian_dreamer.py \
  --configs kirian size50m --replay.size 5e5 --logdir $D/runs/d1 >> $D/d1.log 2>&1
log "DreamerV3 exit $?"
kill $PID
N=$(cat $D/runs/d1/env*/episodes.jsonl 2>/dev/null | wc -l)
../bench/summarize.py "$CSV" --label UC4_dreamer --extra "{\"epizod\": $N}" | tee $D/runs/d1/gpu.json
echo off > /srv/work/.gpu_profile
log NIGHT5_DONE
