#!/usr/bin/env bash
# UC3: benchmark lc0 sítí (200k pozic) → ostrá analýza celé Lichess puzzle DB nejsilnější sítí (BT4)
# s počtem shardů, který v benchmarku dal nejvíc pozic/s. GPU musí být volná (switch.sh off).
#   run_lc0.sh
set -euo pipefail
cd "$(dirname "$0")"; W=/srv/work/uc3
[ "$(cat /srv/work/.gpu_profile 2>/dev/null)" = off ] || ../50_vllm/switch.sh off >/dev/null
echo "lc0" > /srv/work/.gpu_profile
run() {  # run <label> <net> <shards> [limit]
  local L=$1 NET=$2 S=$3 LIM=${4:-0} OUT=$W/runs/$1
  mkdir -p $OUT
  read -r PID CSV < <(../bench/gpu_log.sh uc3_$L)
  for i in $(seq 0 $((S-1))); do
    docker run --rm --gpus all -v $W:/work -v $PWD:/code tms/lc0 python3 /code/analyze.py --inp positions.jsonl \
      --out runs/$L/analysis.s$i.jsonl --net nets/$NET.pb.gz --batch 1024 --shard $i/$S --limit $LIM \
      > $OUT/shard$i.log 2>&1 &
  done
  wait; kill $PID
  N=$(cat $OUT/analysis.s*.jsonl | wc -l)
  ../bench/summarize.py "$CSV" --label UC3_$L --extra "{\"pozic\": $N}" | tee $OUT/summary.json
}
run t1-256-s2 t1-256x10-distilled-swa-2432500 2 100000
run t1-512-s2 t1-512x15x8h-distilled-swa-3395000 2 100000
for S in 1 2 3; do run bt4-s$S BT4-1024x15x32h-swa-6147500 $S $((150000 / S)); done
BEST=$(for S in 1 2 3; do echo "$S $(jq '.pozic / .sekund' $W/runs/bt4-s$S/summary.json)"; done | sort -k2 -nr | head -1 | cut -d' ' -f1)
echo "BT4 nejlepší shardů: $BEST"
run full-bt4 BT4-1024x15x32h-swa-6147500 $BEST 0
echo off > /srv/work/.gpu_profile
echo UC3_LC0_DONE
