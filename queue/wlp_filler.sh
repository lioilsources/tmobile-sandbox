#!/usr/bin/env bash
# Filler window for an external client (qctl windows cannot be fillers): run as a filler cmd job
# on the client's profile. While it runs, READY exists; on preemption (SIGTERM) READY goes away and
# STOP appears, the client stops and resumes on the next READY. The client touches DONE when all
# its work is finished, which ends the filler for good.
#   wlp_filler.sh <PREFIX>   →  /srv/work/queue/<PREFIX>_READY | _STOP | _DONE
set -u
Q=/srv/work/queue; P=${1:?prefix}
rm -f "$Q/${P}_STOP"
trap 'rm -f "$Q/${P}_READY"; touch "$Q/${P}_STOP"; exit 143' TERM INT
touch "$Q/${P}_READY"
while [ ! -e "$Q/${P}_DONE" ]; do sleep 20 & wait $!; done
rm -f "$Q/${P}_READY"
