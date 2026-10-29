#!/usr/bin/env bash
# UC19 dialect benchmark — QUEUE-READY (Director). Each phase needs a different GPU profile → queue separately.
#
#   PHASES=T  timeout 40m bash ~/tmobile-sandbox/uc19/run_uc19.sh   # profile A (qwen3-32b) or S (qwen3-8b) on :8000
#   PHASES=G  timeout 4h  bash ~/tmobile-sandbox/uc19/run_uc19.sh   # profile D (ComfyUI): build jobs + gen_matrix
#   PHASES=S  timeout 1h  bash ~/tmobile-sandbox/uc19/run_uc19.sh   # scorers (GPU ≥ 12 GB) + VLM judge (≥ 38 GB free, own vLLM :8004)
#   PHASES=R  bash ~/tmobile-sandbox/uc19/run_uc19.sh               # CPU report
#
# T: t1 = 200 requests, t2 = 200 × 4 dialects = 800 (15-shot prefix ~2.5k tokens, prefix-cached; out ≤ 150 tok).
#    Estimate qwen3-32b at concurrency 16 (~600–800 out tok/s aggregate) → 5–10 min; qwen3-8b 2–4 min.
#    Both models? run T twice with OUT_TAG=… (translations_<model>.jsonl) — default: whatever :8000 serves.
# G: 200 × {t1,t2} × {pony, illustrious, juggernaut, flux_dev} = 1 600 images
#    ≈ 400×4.3 + 400×4.7 + 400×5.0 + 400×10 s ≈ 2.7 h (+ T1P=1 adds t1+prefix on the 3 SDXL models: +1 200 img, +1.6 h).
#    VRAM as UC18 (ComfyUI ≤ ~20 GB).
# S: A ≈ 5 min for 1 600 images; B single-image judge 1 600 req at ~13.8 k/h (UC25) ≈ 7 min + vLLM start 2–5 min.
# Env: UC19_OUT (/cold/w2/uc19) TECHS (t1,t2) ALL_TARGETS (0) T1P (0) LLM_URL (http://127.0.0.1:8000/v1) LIMIT (0)
set -uo pipefail
R=$(cd "$(dirname "$0")" && pwd); U18=$R/../uc18; O=${UC19_OUT:-/cold/w2/uc19}
mkdir -p "$O"
log() { echo "$(date -Is) run_uc19: $*" | tee -a "$O/run.log"; }
PY=/cold/venvs/uc25/bin/python   # requests, jsonschema, transformers (CLIP tokenizer), tokenizers
export HF_HOME=/srv/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PHASES=${PHASES:-R}
LIM=(); [ "${LIMIT:-0}" != 0 ] && LIM=(--limit "$LIMIT")

if [[ $PHASES == *T* ]]; then
  curl -sf "${LLM_URL:-http://127.0.0.1:8000/v1}/models" >/dev/null || { log "T: no LLM on ${LLM_URL:-:8000} — switch.sh A or S first"; exit 3; }
  $PY "$R/translate.py" --cz "$R/prompts/cz_prompts.jsonl" --fewshot "$R/prompts/fewshot.jsonl" --out "$O/translations.jsonl" \
      --techniques "${TECHS:-t1,t2}" --base-url "${LLM_URL:-http://127.0.0.1:8000/v1}" "${LIM[@]}" 2> "$O/translate.err"
  rc=$?; log "T translate rc=$rc $(tail -1 "$O/translate.err" | cut -c1-300)"; grep -m1 '^first error' "$O/translate.err" | cut -c1-800 | tee -a "$O/run.log"
  [ $rc = 0 ] || exit $rc
fi

if [[ $PHASES == *G* ]]; then
  extra=(); [ "${ALL_TARGETS:-0}" = 1 ] && extra+=(--all-targets); [ "${T1P:-0}" = 1 ] && extra+=(--t1-prefix)
  python3 "$R/build_jobs.py" --translations "$O/translations.jsonl" --out "$O/jobs.jsonl" --techniques "${TECHS:-t1,t2}" "${extra[@]}" \
      > "$O/jobs_summary.json" || { log "build_jobs failed"; exit 2; }
  log "G jobs: $(tr -d '\n ' < "$O/jobs_summary.json" | cut -c1-400)"
  vllm_mib=$(nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader,nounits 2>/dev/null \
             | awk -F', ' 'tolower($1) ~ /vllm/ {s+=$2} END {print s+0}')
  [ "${vllm_mib:-0}" -gt 0 ] && { log "G: vLLM still holds ${vllm_mib} MiB — not starting"; exit 2; }
  curl -sf 127.0.0.1:8188/system_stats >/dev/null || { log "G: ComfyUI :8188 not answering — switch.sh D first"; exit 3; }
  python3 "$U18/gen_matrix.py" --jobs "$O/jobs.jsonl" --out-dir "$O/img" --manifest "$O/manifest.jsonl" --comfy-prefix uc19 --dry-run \
      > "$O/dryrun.json" || { log "G dry run INVALID"; exit 2; }
  python3 "$U18/gen_matrix.py" --jobs "$O/jobs.jsonl" --out-dir "$O/img" --manifest "$O/manifest.jsonl" --comfy-prefix uc19 "${LIM[@]}"
  rc=$?; log "G gen_matrix rc=$rc $(cat "$O/progress.json" 2>/dev/null)"; [ $rc = 0 ] || exit $rc
fi

if [[ $PHASES == *S* ]]; then
  UC18_OUT=$O REFS=$R/prompts/cz_prompts.jsonl JUDGE=single PHASES=AB TAG=UC19_dialects bash "$U18/score.sh"
  log "S score rc=$?"
fi

if [[ $PHASES == *R* ]]; then
  python3 "$R/report.py" --dir "$O" --md "$O/report_UC19.md" --json ~/tmobile-sandbox/bench/results/UC19_dialects.json
  log "R report rc=$?"
fi
log HOTOVO
