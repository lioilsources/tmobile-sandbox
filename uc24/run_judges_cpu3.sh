#!/usr/bin/env bash
# UC24 — Audiobox Aesthetics na CPU dělá ~520 klipů/h, proto jen vzorek: test seed 3 (+ ACE 4) + čisté dev klipy.
# Čeká na larger_clap_general z run_judges_cpu2.sh, ukončí ho a pustí aesthetics nad menším manifestem.
set -uo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES= HF_HOME=/srv/hf HF_HUB_OFFLINE=1
PY=/cold/venvs/uc24/bin/python; LOG=/cold/uc24/judges_cpu.log
until grep -q '"judge": "clap-general"' $LOG 2>/dev/null; do sleep 20; done
tmux kill-session -t uc24j2 2>/dev/null
python3 -c "
import json
with open('/cold/uc24/synth/manifest_aes.jsonl','w') as f:
    for l in open('/cold/uc24/synth/manifest.jsonl'):
        r=json.loads(l)
        if (r['split']=='test' and r['seed'] in (3,4) and (r['type']=='ace_music') == (r['seed']==4)) or r['defect'] in ('clean','mismatch'): f.write(l)
"
nice -n 19 $PY judges.py --judge aesthetics --manifest /cold/uc24/synth/manifest_aes.jsonl --threads 4 2>/dev/null | tee -a $LOG || echo "aesthetics selhal" | tee -a $LOG
echo "HOTOVO (uc24 judges cpu)" | tee -a $LOG
