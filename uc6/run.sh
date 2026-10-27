#!/usr/bin/env bash
# UC6 Hajánek: 30 s hlasu rodiče → ~9 min usínací meditace. Čeká na konec DreamerV3 (NIGHT6_DONE), pak engine po enginu
# s gpu_log a nakonec vyhodnocení. XTTS-v2 jen se souborem xtts.ok (licenci CPML musí odsouhlasit uživatel).
set -uo pipefail
R=~/tmobile-sandbox; W=/srv/work/uc6; Q=/srv/work/queue; export HF_HOME=/srv/work/.hf
log() { echo "$(date -Is) $*" | tee -a $Q/uc6.log; }
until grep -q NIGHT6_DONE $Q/night6.log 2>/dev/null; do sleep 30; done
log "disk: $(df -h /srv | awk 'NR==2{print $4}') free"
echo uc6 > /srv/work/.gpu_profile; cd $W
REFTXT=$(eval-venv/bin/python -c 'import json;print(json.load(open("ref/ref.json"))["clip_text"])')
gen() {  # engine venv ref [ref_text]
  read -r PID CSV < <($R/bench/gpu_log.sh uc6_$1)
  $2/bin/python $R/uc6/tts_gen.py --engine $1 --ref $3 --ref-text "${4:-}" --out runs/$1 > runs/$1.log 2>&1
  log "$1 exit $?"; kill $PID; $R/bench/summarize.py "$CSV" --label UC6_$1 > runs/$1.gpu.json
}
mkdir -p runs
gen f5 f5-venv ref/clip24k.wav "$REFTXT"
gen pocket pocket-venv ref/voice24k.wav
E="f5 pocket"
[ -f xtts.ok ] && { COQUI_TOS_AGREED=1 gen xtts xtts-venv ref/voice24k.wav; E="xtts $E"; }
eval-venv/bin/python $R/uc6/eval_tts.py eval --ref ref/voice16k.wav --runs runs $E > runs/eval.log 2>&1; log "eval exit $?"
echo off > /srv/work/.gpu_profile; log UC6_DONE
