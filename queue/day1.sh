#!/usr/bin/env bash
set -uo pipefail
R=~/tmobile-sandbox; Q=/srv/work/queue
log() { echo "$(date -Is) $*" | tee -a $Q/day1.log; }
bash $R/uc8/run.sh > $Q/uc8.log 2>&1; log "UC8 exit $?"
bash $R/uc5/run.sh > $Q/uc5.log 2>&1; log "UC5 exit $?"
log DAY1_DONE
