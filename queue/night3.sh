#!/usr/bin/env bash
# Třetí fronta: po night2 → UC5(a) benchmark ComfyUI (profil D).
set -uo pipefail
R=~/tmobile-sandbox; Q=/srv/work/queue
log() { echo "$(date -Is) $*" | tee -a $Q/night3.log; }
until grep -q NIGHT2_DONE $Q/night2.log 2>/dev/null; do sleep 60; done
bash $R/uc5/run.sh > $Q/uc5.log 2>&1; log "UC5 exit $?"
log NIGHT3_DONE
