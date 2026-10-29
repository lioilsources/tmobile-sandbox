#!/usr/bin/env bash
# UC18 matrix 300 prompts × 7 checkpoints × seeds through ComfyUI — QUEUE-READY (Director starts it).
#
#   bash 60_comfy/switch.sh D   (Director)  → then:
#   SEEDS=1   timeout 5h  bash ~/tmobile-sandbox/uc18/run_matrix.sh >> /srv/work/queue/<noc>.log 2>&1   # pass 1
#   SEEDS=1,2 timeout 5h  bash ~/tmobile-sandbox/uc18/run_matrix.sh >> …                                # adds seed 2 (resume)
#
# Estimate (models.est_s; SDXL measured 5.03 s @30 steps, FLUX.1-dev 20 steps ESTIMATED 10 s):
#   per seed: 6 SDXL ckpts × 300 × ~4.7 s ≈ 2.35 h + FLUX 300 × ~10 s ≈ 0.85 h + 7 model loads → ~3.3 h
#   both seeds ≈ 6.5 h. The health check (5 min) prints the measured s/img and the ×1.5 timeout to use.
# VRAM: ComfyUI SDXL ~17 GB peak, FLUX.1-dev fp8 ~20 GB (as schnell; to be measured) — profile D incl. Whisper.
# Disk: ~1.5 MB/PNG → 4 200 images ≈ 6–7 GB on /cold (ComfyUI's copy in /srv/work/comfy-out is deleted).
#
# Env: SEEDS (1) CKPTS (all | pony,illustrious,…) STYLES ("" = all) LIMIT (0) DEPTH (6) UC18_OUT (/cold/w2/uc18)
# Exit: 0 ok · 2 GPU held by vLLM / invalid workflow · 3 ComfyUI down · 4 disk · 6 health check · 7 stall · 8 ComfyUI lost
set -uo pipefail
R=$(cd "$(dirname "$0")" && pwd); O=${UC18_OUT:-/cold/w2/uc18}
mkdir -p "$O"
log() { echo "$(date -Is) run_matrix: $*" | tee -a "$O/run.log"; }

# 1) ComfyUI up (profile D), nothing else on the GPU
for i in $(seq 1 24); do curl -sf 127.0.0.1:8188/system_stats >/dev/null && break; sleep 5; done
curl -sf 127.0.0.1:8188/system_stats >/dev/null || { log "ComfyUI :8188 not answering — switch.sh D first"; exit 3; }
vllm_mib=$(nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader,nounits 2>/dev/null \
           | awk -F', ' 'tolower($1) ~ /vllm/ {s+=$2} END {print s+0}')
[ "${vllm_mib:-0}" -gt 0 ] && { log "vLLM still holds ${vllm_mib} MiB of VRAM — not starting"; exit 2; }

# 2) disk
cold_gb=$(df -BG --output=avail /cold | tail -1 | tr -dc 0-9); srv_gb=$(df -BG --output=avail /srv | tail -1 | tr -dc 0-9)
[ "$cold_gb" -lt 20 ] && { log "/cold only ${cold_gb} GB free"; exit 4; }
[ "$srv_gb" -lt 5 ] && { log "/srv only ${srv_gb} GB free (ComfyUI writes there first)"; exit 4; }

ARGS=(--prompts "$R/prompts/uc18_prompts.jsonl" --ckpts "${CKPTS:-all}" --seeds "${SEEDS:-1}"
      --out-dir "$O/img" --manifest "$O/manifest.jsonl" --depth "${DEPTH:-6}")
[ -n "${STYLES:-}" ] && ARGS+=(--styles "$STYLES")
[ "${LIMIT:-0}" != 0 ] && ARGS+=(--limit "$LIMIT")

# 3) dry run against the LIVE /object_info (catches a missing checkpoint file before any GPU time)
python3 "$R/gen_matrix.py" "${ARGS[@]}" --dry-run > "$O/dryrun.json" || { log "dry run INVALID: $(cat "$O/dryrun.json")"; exit 2; }
log "dry run OK: $(tr -d '\n ' < "$O/dryrun.json")"

# 4) the run (python3 = stdlib only)
python3 "$R/gen_matrix.py" "${ARGS[@]}"
rc=$?
log "gen_matrix exit $rc; $(grep -vc '"error"' "$O/manifest.jsonl" 2>/dev/null) ok rows, $(grep -c '"error"' "$O/manifest.jsonl" 2>/dev/null) error rows; $(cat "$O/progress.json" 2>/dev/null)"
exit $rc
