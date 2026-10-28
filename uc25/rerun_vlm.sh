#!/usr/bin/env bash
# UC25: opakování fáze B (VLM Qwen3-VL-8B) + fáze C s VLM — QUEUE-READY, spouští Director do fronty.
#
#   timeout 90m bash ~/tmobile-sandbox/uc25/rerun_vlm.sh >> /srv/work/queue/<noc>.log 2>&1
#
# Předpoklad: GPU volná (profil off, ≥ 38 GB volné VRAM) — vllm-vl8b bere 0,80 × 46 GB ≈ 37 GB
# (váhy BF16 ~17 GB + KV cache). Výstupy fáze A (wd14/nsfw/siglip .jsonl, emb_gpu.npz) se znovu nepočítají.
# Doba: start vLLM 2–5 min (váhy z /srv/hf, compile) + B ≈ 3 037 obr. 10–20 min (odhad, B ještě nikdy
# neproběhla) + C ≈ 2–3 min (změřeno) → ~20–30 min, limit 90 min stačí s rezervou.
#
# Kroky: up → health → JEDEN zkušební request (při chybě vypíše tělo odpovědi vLLM a skončí) →
# fáze B → docker logs do out/gpu → down → fáze C s VLM → bench/results/UC25_gpu.json.
# Kontejner se zastaví i při chybě/timeoutu (trap). Exit: 0 OK, 2 málo VRAM, 3 vLLM nenastartoval,
# 4 zkušební request selhal, 5 fáze B bez jediné odpovědi.
set -uo pipefail
R=~/tmobile-sandbox/uc25; W=/srv/work/w2/uc25; T=$W/ftg/taxo; O=$W/out/gpu
export HF_HOME=/srv/hf HF_HUB_OFFLINE=1
. /cold/venvs/uc25/bin/activate
mkdir -p $O
log() { echo "$(date -Is) rerun_vlm: $*" | tee -a $O/run.log; }

need=${NEED_MIB:-38000}
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
[ "$free" -lt "$need" ] && { log "volná VRAM $free MiB < $need — GPU drží něco jiného, končím"; exit 2; }

started=0
cleanup() { [ $started = 1 ] && { bash $R/vllm_vl8b.sh down >/dev/null 2>&1; log "vllm-vl8b down (logy v $O/vllm-vl8b.log)"; }; }
trap cleanup EXIT
trap 'exit 143' TERM INT   # timeout z fronty → EXIT trap → down

log "start vllm-vl8b (volná VRAM $free MiB)"
started=1
UP_TIMEOUT=${UP_TIMEOUT:-900} bash $R/vllm_vl8b.sh up ${V_GPU:-0.80} || { log "vLLM nenastartoval — viz $O/vllm-vl8b.log"; exit 3; }

# zkušební request: přesně tvar dávky (schéma, obrázek, max_tokens), 1 obrázek
head -1 $O/manifest.jsonl > $O/m_probe.jsonl
python $T/eval_vlm.py --manifest $O/m_probe.jsonl --out $O/vlm_probe.jsonl --base-url http://127.0.0.1:8004/v1 \
  --model qwen3-vl-8b --concurrency 1 2>$O/vlm_probe.err
if [ $? -ne 0 ] || grep -q '"error"' $O/vlm_probe.jsonl; then
  log "zkušební request SELHAL:"; grep -m1 '^first error' $O/vlm_probe.err | cut -c1-2000 | tee -a $O/run.log
  bash $R/vllm_vl8b.sh logs >/dev/null; exit 4
fi
log "zkušební request OK: $(head -c 400 $O/vlm_probe.jsonl)"

# fáze B (starý vlm.jsonl se samými chybami odložit)
[ -f $O/vlm.jsonl ] && mv $O/vlm.jsonl $O/vlm.failed-$(date +%Y%m%dT%H%M).jsonl
PHASES=B bash $R/gpu_batch.sh; log "fáze B exit $?"
bash $R/vllm_vl8b.sh down >/dev/null 2>&1; started=0; log "vllm-vl8b down (docker logs → $O/vllm-vl8b.log)"
ok=$(grep -vc '"error"' $O/vlm.jsonl 2>/dev/null); ok=${ok:-0}
[ "$ok" -gt 0 ] || { log "fáze B nevrátila ani jednu odpověď — fáze C nespouštím"; exit 5; }
log "VLM odpovědi: $ok z $(wc -l < $O/vlm.jsonl)"

PHASES=C NOTE="${NOTE:-fáze A+B na L40S, C na CPU; VLM Qwen3-VL-8B guided JSON}" bash $R/gpu_batch.sh
rc=$?; log "fáze C exit $rc"; exit $rc
