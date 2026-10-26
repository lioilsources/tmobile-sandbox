#!/usr/bin/env bash
# Stav stroje každé 2 s, okamžitě na disk (sync) — pro případ tvrdého pádu VM.
while true; do
  printf "%s %s %s\n" "$(date +%T)" "$(free -m | awk "/Mem/{printf \"ram=%d\", \$3} /Swap/{printf \" swap=%d\", \$3}")"     "$(nvidia-smi --query-gpu=utilization.gpu,power.draw,temperature.gpu,memory.used,clocks_throttle_reasons.active --format=csv,noheader,nounits | tr -d " " | sed "s/^/gpu=/")" >> /srv/work/demo2/crashwatch.log
  sync /srv/work/demo2/crashwatch.log; sleep 2
done
