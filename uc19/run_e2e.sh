#!/usr/bin/env bash
# UC18+19 finish (plan 08 P3, filler F3) — one qctl cmd job, profile L; switches profiles itself between phases.
#
#   cd ~/tmobile-sandbox && uc19/run_e2e.sh            # PHASES=TGSIR (default)
#
# T  profile L (vLLM :8000): translate the 50 held-out CZ prompts, t1 + t2 × 4 dialects = 250 requests, ~2 min
# G  profile D (ComfyUI): e2e.py jobs (router on CPU) + gen_matrix, 150 images ≈ 13 min
# S  profile off: score.sh A (PickScore, SigLIP, WD14, ImageReward) + B (Qwen3-VL-8B single judge, own vLLM :8004)
# I  profile off: ImageReward over the existing UC18 (6 300) and UC19 (1 600) matrices — the other scorers are
#    resumable and skip done images; est. 3–6 k img/h → 1.5–2.5 h. Then UC18 phase C (router/report) + UC19 report.
# R  CPU: e2e report → /cold/w2/uc19e2e/report_UC19_e2e.md + bench/results/UC19_e2e.json
set -uo pipefail
R=$(cd "$(dirname "$0")" && pwd); U18=$R/../uc18; SW=$R/../50_vllm/switch.sh; O=${E2E_OUT:-/cold/w2/uc19e2e}
mkdir -p "$O"
log() { echo "$(date -Is) run_e2e: $*" | tee -a "$O/run.log"; }
PY=/cold/venvs/uc25/bin/python
export HF_HOME=/srv/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PHASES=${PHASES:-TGSIR}
profile() { [ "$(awk '{print $1}' /srv/work/.gpu_profile 2>/dev/null)" = "$1" ] || bash "$SW" "$1" >> "$O/switch.log" 2>&1; }

if [[ $PHASES == *T* ]]; then
  curl -sf 127.0.0.1:8000/v1/models >/dev/null || profile L
  $PY "$R/translate.py" --cz "$R/prompts/e2e_cz50.jsonl" --fewshot "$R/prompts/fewshot.jsonl" --out "$O/translations.jsonl" \
      --techniques t1,t2 2> "$O/translate.err"
  rc=$?; log "T translate rc=$rc $(tail -1 "$O/translate.err" | cut -c1-300)"; [ $rc = 0 ] || exit $rc
fi

if [[ $PHASES == *G* ]]; then
  profile D
  $PY "$R/e2e.py" jobs --dir "$O" > "$O/jobs_summary.json" || { log "e2e jobs failed"; exit 2; }
  log "G jobs: $(tr -d '\n ' < "$O/jobs_summary.json" | cut -c1-400)"
  for i in $(seq 60); do curl -sf 127.0.0.1:8188/system_stats >/dev/null && break; sleep 5; done
  python3 "$U18/gen_matrix.py" --jobs "$O/jobs.jsonl" --out-dir "$O/img" --manifest "$O/manifest.jsonl" --comfy-prefix uc19e2e
  rc=$?; log "G gen_matrix rc=$rc $(cat "$O/progress.json" 2>/dev/null)"; [ $rc = 0 ] || exit $rc
fi

if [[ $PHASES == *S* ]]; then
  profile off
  UC18_OUT=$O REFS=$R/prompts/e2e_cz50.jsonl JUDGE=single PHASES=AB TAG=UC19_e2e bash "$U18/score.sh"
  log "S score rc=$?"
fi

if [[ $PHASES == *I* ]]; then
  profile off
  if [ -f /cold/hf/imagereward/ImageReward.pt ]; then
    UC18_OUT=/cold/w2/uc18 PHASES=A bash "$U18/score.sh"; log "I uc18 A rc=$?"
    UC18_OUT=/cold/w2/uc18 PHASES=C bash "$U18/score.sh"; log "I uc18 C rc=$?"
    UC18_OUT=/cold/w2/uc19 REFS=$R/prompts/cz_prompts.jsonl JUDGE=single PHASES=A TAG=UC19_dialects bash "$U18/score.sh"; log "I uc19 A rc=$?"
    PHASES=R bash "$R/run_uc19.sh"; log "I uc19 report rc=$?"
  else
    log "I: ImageReward not set up (uc18/setup_imagereward.sh) — skipped"
  fi
fi

if [[ $PHASES == *R* ]]; then
  $PY "$R/e2e.py" report --dir "$O" --json ~/tmobile-sandbox/bench/results/UC19_e2e.json
  log "R report rc=$?"
fi
log HOTOVO
