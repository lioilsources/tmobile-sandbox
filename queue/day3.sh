#!/usr/bin/env bash
# Večer po UC10: UC7 LoRA RealEyes (celé GPU, bez služeb) → UC2 evoluce shaderů (profil L + EGL evaluátor).
set -uo pipefail
R=~/tmobile-sandbox; Q=/srv/work/queue
log() { echo "$(date -Is) $*" | tee -a $Q/day3.log; }
until grep -q DAY2_DONE $Q/day2.log 2>/dev/null; do sleep 60; done
if [ -f /srv/work/uc7/dataset/summary.json ]; then bash $R/uc7/run.sh > $Q/uc7.log 2>&1; log "UC7 exit $?"; fi
bash $R/uc2/run.sh > $Q/uc2.log 2>&1; log "UC2 exit $?"
log DAY3_DONE
