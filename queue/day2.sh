#!/usr/bin/env bash
set -uo pipefail
R=~/tmobile-sandbox; Q=/srv/work/queue
log() { echo "$(date -Is) $*" | tee -a $Q/day2.log; }
until grep -q DAY1_DONE $Q/day1.log 2>/dev/null; do sleep 60; done
bash $R/uc10/run.sh > $Q/uc10.log 2>&1; log "UC10 exit $?"
log DAY2_DONE
