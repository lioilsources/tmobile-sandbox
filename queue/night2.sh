#!/usr/bin/env bash
# Druhá fronta: po night1 → UC8 obrázek→3D (profil D).
set -uo pipefail
R=~/tmobile-sandbox; Q=/srv/work/queue; mkdir -p $Q
log() { echo "$(date -Is) $*" | tee -a $Q/night2.log; }
until grep -q NIGHT1_DONE $Q/night1.log 2>/dev/null; do sleep 60; done
bash $R/uc8/run.sh 200 > $Q/uc8.log 2>&1; log "UC8 exit $? $(tail -1 $Q/uc8.log)"
log NIGHT2_DONE
