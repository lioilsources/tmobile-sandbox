#!/usr/bin/env bash
# UC18 (and UC19) scoring pass — QUEUE-READY (Director). Runs over whatever gen_matrix has produced so far.
#
#   PHASES=A  timeout 2h bash ~/tmobile-sandbox/uc18/score.sh   # PickScore, SigLIP 2, WD14 (+ ImageReward if set up)
#   PHASES=B  timeout 3h bash ~/tmobile-sandbox/uc18/score.sh   # Qwen3-VL-8B judge: starts/stops its own vLLM :8004
#   PHASES=C  bash ~/tmobile-sandbox/uc18/score.sh               # CPU: Bradley–Terry, TrueSkill, router, report (minutes)
#   PHASES=ABC …                                                 # all (B skipped with a log line if VRAM < 38 GB)
#
# Phase A — GPU ≥ 12 GB free (ComfyUI may stay up; vLLM must be down). Models are loaded one after another:
#   PickScore CLIP-H fp16 ~4 GB, SigLIP 2 so400m fp16 ~2.5 GB, WD14 ONNX ~1 GB (+ ImageReward ~2 GB).
#   Speed estimate for 4 200 images: WD14 71 851/h and SigLIP 54 801/h (measured UC25) → ~5 min each; PickScore
#   ≈ SigLIP-class → ~5–10 min; ImageReward (BLIP, 1 image/call) ~3–6 k/h estimated → 45–80 min → it runs last.
# Phase B — needs ≥ 38 GB free (profile off; ComfyUI stopped). vllm-vl8b 0.80 × 46 GB ≈ 37 GB.
#   UC18 items: 300 prompts × C(7,2)=21 pairs (seed 1) = 6 300 + 15 % swapped re-asks ≈ 7 250 requests.
#   UC25 measured 13 843 single-image requests/h (1 500 prompt tokens, ≤ 344 out); pairs ≈ 1 700 prompt tokens,
#   ≤ 150 out → estimate 9–14 k/h → 35–50 min + 2–5 min vLLM start. Probe request first; abort on >50 % errors.
# Phase C — CPU, 4 threads: rank.py (200× bootstrap) + report.py, estimated 2–5 min.
#
# Env: UC18_OUT (/cold/w2/uc18) REFS (uc18 prompt set) JUDGE (pair|single) SEED_IDX (1) SWAP_FRAC (0.15)
#      VLM_CONC (48) GROUP_KEY (ckpt|variant_ckpt) TAG (UC18_matrix) SKIP_IR (0)
# Exit: 0 ok · 2 not enough VRAM · 3 vLLM did not start · 4 probe failed · 5 judge produced nothing
set -uo pipefail
R=$(cd "$(dirname "$0")" && pwd); O=${UC18_OUT:-/cold/w2/uc18}; S=$O/scores
REFS=${REFS:-$R/prompts/uc18_prompts.jsonl}; JUDGE=${JUDGE:-pair}; TAG=${TAG:-UC18_matrix}
export HF_HOME=/srv/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export FTG_TAXO=${FTG_TAXO:-/srv/work/w2/uc25/ftg/taxo}
PY=/cold/venvs/uc25/bin/python; IR_PY=${IR_PY:-/cold/venvs/uc18-ir/bin/python}
mkdir -p "$S"
log() { echo "$(date -Is) score: $*" | tee -a "$O/run.log"; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
PHASES=${PHASES:-ABC}
[ -s "$O/manifest.jsonl" ] || { log "no $O/manifest.jsonl — run gen_matrix first"; exit 1; }
log "phases $PHASES, $(grep -vc '"error"' "$O/manifest.jsonl") images in manifest, refs $(basename "$REFS")"

if [[ $PHASES == *A* ]]; then
  f=$(free_mib); [ "$f" -lt 12000 ] && { log "phase A: only $f MiB VRAM free — skipped"; exit 2; }
  bash -c "while sleep 15; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader >> $O/gpu_score.csv; done" & MON=$!
  for sc in wd14 siglip pick; do
    extra=(); [ $sc = siglip ] && extra=(--text-emb-out "$S/text_emb.npz")
    t0=$SECONDS
    $PY "$R/score_images.py" --manifest "$O/manifest.jsonl" --refs "$REFS" --scorer $sc --out "$S/$sc.jsonl" \
        --device cuda --batch 16 "${extra[@]}" 2> "$S/$sc.err"
    log "A $sc rc=$? $((SECONDS - t0)) s $(tail -1 "$S/$sc.err" | cut -c1-200)"
    grep -m1 '^first error' "$S/$sc.err" | cut -c1-600 | tee -a "$O/run.log"
  done
  if [ "${SKIP_IR:-0}" = 0 ] && [ -x "$IR_PY" ] && [ -f /cold/hf/imagereward/ImageReward.pt ]; then
    t0=$SECONDS
    HF_HOME=/cold/hf $IR_PY "$R/score_imagereward.py" --manifest "$O/manifest.jsonl" --refs "$REFS" --out "$S/ir.jsonl" 2> "$S/ir.err"
    log "A imagereward rc=$? $((SECONDS - t0)) s $(tail -1 "$S/ir.err" | cut -c1-200)"
  else
    log "A imagereward skipped (not set up — uc18/setup_imagereward.sh)"
  fi
  kill $MON 2>/dev/null
fi

if [[ $PHASES == *B* ]]; then
  f=$(free_mib)
  if [ "$f" -lt "${NEED_MIB:-38000}" ]; then
    log "phase B: only $f MiB VRAM free (< ${NEED_MIB:-38000}; ComfyUI/vLLM still up?) — skipped"
  else
    started=1
    cleanup() { [ "${started:-0}" = 1 ] && { UC18_OUT=$O bash "$R/vllm_judge.sh" down >/dev/null 2>&1; log "vllm-vl8b down"; }; }
    trap cleanup EXIT; trap 'exit 143' TERM INT
    UC18_OUT=$O bash "$R/vllm_judge.sh" up "${V_GPU:-0.80}" || { log "vLLM did not start — $O/vllm-judge.log"; exit 3; }
    $PY "$R/vlm_judge.py" $JUDGE --manifest "$O/manifest.jsonl" --refs "$REFS" --out "$S/vlm_probe.jsonl" \
        --seed-idx "${SEED_IDX:-1}" --limit 1 --concurrency 1 2> "$S/vlm_probe.err"
    if grep -q '"error"' "$S/vlm_probe.jsonl" 2>/dev/null || [ ! -s "$S/vlm_probe.jsonl" ]; then
      log "probe FAILED: $(grep -m1 '^first error' "$S/vlm_probe.err" | cut -c1-1500)"; exit 4
    fi
    log "probe OK: $(tail -1 "$S/vlm_probe.jsonl" | cut -c1-300)"; rm -f "$S/vlm_probe.jsonl"
    out=$S/vlm_pairs.jsonl; [ "$JUDGE" = single ] && out=$S/vlm_single.jsonl
    t0=$SECONDS
    $PY "$R/vlm_judge.py" $JUDGE --manifest "$O/manifest.jsonl" --refs "$REFS" --out "$out" \
        --seed-idx "${SEED_IDX:-1}" --swap-frac "${SWAP_FRAC:-0.15}" --concurrency "${VLM_CONC:-48}" 2> "$S/vlm.err"
    rc=$?; log "B judge $JUDGE rc=$rc $((SECONDS - t0)) s $(tail -1 "$S/vlm.err" | cut -c1-300)"
    grep -m1 '^first error' "$S/vlm.err" | cut -c1-800 | tee -a "$O/run.log"
    UC18_OUT=$O bash "$R/vllm_judge.sh" down >/dev/null 2>&1; started=0; log "vllm-vl8b down"
    ok=$(grep -vc '"error"' "$out" 2>/dev/null); [ "${ok:-0}" -gt 0 ] || { log "judge produced no answers"; exit 5; }
  fi
fi

if [[ $PHASES == *C* ]]; then
  if [ "$JUDGE" = pair ]; then
    t0=$SECONDS
    timeout 20m nice -n 10 $PY "$R/rank.py" --dir "$O" --out "$O/rank.json" --router-out "$O/router.json" \
        --group-key "${GROUP_KEY:-ckpt}" 2>> "$O/run.log"
    log "C rank rc=$? $((SECONDS - t0)) s"
    timeout 5m python3 "$R/report.py" --rank "$O/rank.json" --dir "$O" --md "$O/report_${TAG}.md" \
        --json ~/tmobile-sandbox/bench/results/${TAG}.json 2>> "$O/run.log"
    log "C report rc=$?"
  else
    log "C: JUDGE=single → UC19 report is uc19/report.py (run_uc19.sh calls it)"
  fi
fi
log "score HOTOVO"
