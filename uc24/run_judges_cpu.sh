#!/usr/bin/env bash
# UC24 krok 2 — CPU soudci nad celou syntetickou sadou (bez GPU; ~2 h na 8 vCPU VM při zátěži UC11).
#   tmux new -d -s uc24j 'bash ~/tmobile-sandbox/uc24/run_judges_cpu.sh'
set -uo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES= HF_HOME=/srv/hf HF_HUB_OFFLINE=1
PY=/cold/venvs/uc24/bin/python; LOG=/cold/uc24/judges_cpu.log
for j in panns clap-unfused clap-general aesthetics; do
  nice -n 19 $PY judges.py --judge $j --threads 4 2>/dev/null | tee -a $LOG || echo "$j selhal" | tee -a $LOG
done
echo "HOTOVO (uc24 judges cpu)" | tee -a $LOG
