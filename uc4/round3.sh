#!/usr/bin/env bash
# UC4 kolo 3: kampaň s obchodem (start z pilota kola 2) → vyhodnocení obchodník / pravidlo / bez nákupů
set -uo pipefail
export PATH=/srv/work/uc4/flutter/bin:$PATH
cd ~/tmobile-sandbox/uc4; PY=/srv/work/uc1/ft-venv/bin/python; R=/srv/work/uc4/runs
until grep -q ROUND2_DONE $R/ppo2.log 2>/dev/null; do sleep 60; done
$PY ppo_campaign.py --repo /srv/work/uc4/kiran --init $R/ppo2/agent.pt --out $R/ppo3 --steps 16000000 --base-port 7000 2>&1 | tee $R/ppo3.log
for s in learn greedy none; do
  $PY ppo_campaign.py --repo /srv/work/uc4/kiran --eval $R/ppo3 --shop $s --runs 30 --base-port 7100 2>&1 | tail -1 | tee -a $R/ppo3_eval.log
done
echo ROUND3_DONE >> $R/ppo3.log
