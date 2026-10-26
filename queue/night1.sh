#!/usr/bin/env bash
# Noční fronta GPU (2026-10-26): po UC3 lc0 → UC10 galerie stylů (profil D) a souběžně na CPU výběr
# kvízů UC3 (Stockfish v kontejneru) → UC3 texty Qwen3-32B (profil A). Každý krok loguje do /srv/work/queue/.
set -uo pipefail
R=~/tmobile-sandbox; Q=/srv/work/queue; mkdir -p $Q
log() { echo "$(date -Is) $*" | tee -a $Q/night1.log; }
until grep -q UC3_LC0_DONE /srv/work/uc3/run_lc0.log 2>/dev/null; do sleep 60; done
log "UC3 lc0 hotovo → výběr kvízů (CPU) + UC10 (GPU)"
docker run --rm -v /srv/work/uc3:/work -v $R/uc3:/code tms/lc0 python3 /code/select_quizzes.py --pos positions.jsonl \
  --ana 'runs/full-bt4/analysis.s*.jsonl' --out quizzes.json --per-lesson 400 > $Q/uc3_select.log 2>&1 &
SEL=$!
bash $R/uc10/run.sh 4 > $Q/uc10.log 2>&1; log "UC10 exit $?"
wait $SEL; log "UC3 výběr exit $? $(tail -1 $Q/uc3_select.log)"
[ -x /srv/work/uc3/venv/bin/python ] || { python3 -m venv /srv/work/uc3/venv && /srv/work/uc3/venv/bin/pip -q install httpx chess; }
$R/50_vllm/switch.sh A > /dev/null
/srv/work/uc3/venv/bin/python $R/uc3/texts.py --inp /srv/work/uc3/quizzes.json --out /srv/work/uc3/quizzes_cs.json > $Q/uc3_texts.log 2>&1
log "UC3 texty exit $? $(tail -1 $Q/uc3_texts.log)"
log NIGHT1_DONE
