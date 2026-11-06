#!/usr/bin/env bash
# UC4 iterace 3 (6. 11.). Sweep startovního kreditu (0–1 M) ani delší limit sektoru (300/600 s) kampaň neposunul:
# v sektoru 2 zůstává falconx (1 000 HP), který neodlétá, a pilot do něj nestřílí. Proto kampaň s odměnou
# za poškození nepřátel a limitem 300 s, init pilot iterace 2.
set -uo pipefail
export PATH=/srv/work/uc4/flutter/bin:$PATH
cd ~/tmobile-sandbox/uc4; PY=/srv/work/uc1/ft-venv/bin/python; R=/srv/work/uc4/runs; O=$R/it3; K=/srv/work/uc4/kiran
mkdir -p $O/campaign; log(){ echo "$(date "+%F %T") $*" | tee -a $O/it3.log; }
log "1: kampaň s odměnou za poškození, limit 300 s, 16 M kroků"
$PY ppo_campaign.py --repo $K --init $R/it2/campaign/agent.pt --out $O/campaign --steps 16000000 --damage-reward --max-t 300 \
  --base-port 7000 > $O/campaign.log 2>&1; log "kampaň exit $?"
for s in learn greedy none; do
  $PY ppo_campaign.py --repo $K --eval $O/campaign --shop $s --runs 30 --max-t 300 --base-port 7100 2>&1 | tail -1 | tee -a $O/it3.log
done
log IT3_DONE
