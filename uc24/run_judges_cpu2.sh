#!/usr/bin/env bash
# UC24 — zkrácení CPU soudců: larger_clap_general a Audiobox Aesthetics jen nad test částí + čistými dev
# klipy (prahy), protože na CPU dělají ~1100 resp. ~520 klipů/h (celá sada by trvala ~6 h).
# Čeká na clap-unfused.npz z run_judges_cpu.sh, pak ho ukončí a pokračuje tady.
set -uo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES= HF_HOME=/srv/hf HF_HUB_OFFLINE=1
PY=/cold/venvs/uc24/bin/python; LOG=/cold/uc24/judges_cpu.log
until grep -q '"judge": "clap-unfused"' $LOG 2>/dev/null; do sleep 20; done  # řádek se tiskne až po np.savez
tmux kill-session -t uc24j 2>/dev/null
python3 -c "
import json
with open('/cold/uc24/synth/manifest_eval.jsonl','w') as f:
    for l in open('/cold/uc24/synth/manifest.jsonl'):
        r=json.loads(l)
        if r['split']=='test' or r['defect'] in ('clean','mismatch'): f.write(l)
"
for j in clap-general aesthetics; do
  nice -n 19 $PY judges.py --judge $j --manifest /cold/uc24/synth/manifest_eval.jsonl --threads 4 2>/dev/null | tee -a $LOG || echo "$j selhal" | tee -a $LOG
done
echo "HOTOVO (uc24 judges cpu)" | tee -a $LOG
