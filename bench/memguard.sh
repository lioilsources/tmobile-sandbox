#!/usr/bin/env bash
# Pojistka RAM (VM má 23 GB): když MemAvailable klesne pod práh, ukončí proces podle vzoru dřív, než VM zamrzne
# při swapování / OOM (stalo se 27. 10. s DreamerV3). Zápis do /srv/work/bench/memguard.log.
#   memguard.sh "<vzor pro pgrep -f>" [práh_MB=1500]
PAT=${1:?vzor}; MIN=${2:-1500}; L=/srv/work/bench/memguard.log
while true; do
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  if [ "$avail" -lt "$MIN" ]; then
    pids=$(pgrep -f "^[^ ]*python[^ ]* $PAT")
    [ -n "$pids" ] && { echo "$(date -Is) MemAvailable ${avail} MB < ${MIN} → kill $pids ($PAT)" >> $L; kill $pids; sleep 30; }
  fi
  sleep 5
done
