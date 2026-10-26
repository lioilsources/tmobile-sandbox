#!/usr/bin/env bash
# UC4 kolo 2: „vystřílej všechno“ → evaluace kola 2 i kola 1 (se sestřely) → GIFy
set -uo pipefail
export PATH=/srv/work/uc4/flutter/bin:$PATH
cd ~/tmobile-sandbox/uc4; PY=/srv/work/uc1/ft-venv/bin/python; R=/srv/work/uc4/runs
$PY ppo.py --repo /srv/work/uc4/kiran --envs 8 --steps 16000000 --mode clear --init $R/ppo1/agent.pt \
  --focus 2,4,5,10,11,12,13,17 --out $R/ppo2 --base-port 7000 2>&1 | tee $R/ppo2.log
echo PPO2_DONE >> $R/ppo2.log
$PY evaluate.py --repo /srv/work/uc4/kiran --agent $R/ppo2/agent.pt --out $R/ppo2/eval --episodes 30 --mode clear 2>&1 | tail -2
$PY evaluate.py --repo /srv/work/uc4/kiran --agent $R/ppo1/agent.pt --out $R/ppo1/eval_k --episodes 30 2>&1 | tail -2
mkdir -p $R/ppo2/eval/gif
for s in 04 08 10 13; do /srv/work/uc1/venv/bin/python replay_gif.py $R/ppo2/eval/replays/sector_$s.json --out $R/ppo2/eval/gif/sector_$s.gif --every 3 --max-frames 250; done
echo ROUND2_DONE >> $R/ppo2.log
