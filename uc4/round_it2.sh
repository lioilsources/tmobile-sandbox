#!/usr/bin/env bash
# UC4 iterace 2 (4. 11.): „vystřílej všechno“ + nákupy. Pilot se učí s náhodnou výbavou (od základní po silnou),
# pak obchodník nad ním v kampani. Krok 0 ověří, jestli se sektory s maximální výbavou vůbec dají vyčistit.
set -uo pipefail
export PATH=/srv/work/uc4/flutter/bin:$PATH
cd ~/tmobile-sandbox/uc4; PY=/srv/work/uc1/ft-venv/bin/python; R=/srv/work/uc4/runs; O=$R/it2; K=/srv/work/uc4/kiran
mkdir -p $O; log(){ echo "$(date "+%F %T") $*" | tee -a $O/it2.log; }
log "0: pilot kola 3 s max. výbavou (strop)"
$PY evaluate.py --repo $K --agent $R/ppo3/agent.pt --v3 --loadout max --no-random --episodes 10 --out $O/eval_max_ppo3 2>&1 | tail -1 | tee -a $O/it2.log
log "1: pilot, náhodná výbava, 20 M kroků"
$PY ppo.py --repo $K --envs 8 --steps 20000000 --mode clear --v3 --loadout random --init $R/ppo3/agent.pt \
  --focus 2,4,5,9,10,11,12,13,17 --out $O/pilot --base-port 7000 > $O/pilot.log 2>&1; log "pilot exit $?"
log "2: vyhodnocení pilota (základní a max. výbava)"
$PY evaluate.py --repo $K --agent $O/pilot/agent.pt --v3 --episodes 20 --out $O/pilot/eval 2>&1 | tail -1 | tee -a $O/it2.log
$PY evaluate.py --repo $K --agent $O/pilot/agent.pt --v3 --loadout max --no-random --episodes 10 --out $O/pilot/eval_max 2>&1 | tail -1 | tee -a $O/it2.log
log "3: kampaň s obchodem, 16 M kroků"
$PY ppo_campaign.py --repo $K --init $O/pilot/agent.pt --out $O/campaign --steps 16000000 --base-port 7000 > $O/campaign.log 2>&1; log "kampaň exit $?"
for s in learn greedy none; do
  $PY ppo_campaign.py --repo $K --eval $O/campaign --shop $s --runs 30 --base-port 7100 2>&1 | tail -1 | tee -a $O/it2.log
done
log IT2_DONE
