#!/usr/bin/env bash
# UC25 plný běh na L40S — QUEUE-READY (spouští Director podle okna GPU).
#
#   fáze A  (bez vLLM, ~6 GB VRAM)   WD14 + 2× NSFW + SigLIP 2 přes všechny sandbox obrázky (+ kalibrační sadu)
#   fáze B  (vLLM Qwen3-VL-8B :8004) strukturovaná extrakce na stratifikovaném vzorku (VLM_N na zdroj)
#   fáze C  (CPU, minuty)            konsenzus, multi-head hlava + ONNX, F1 hlava vs VLM, conformal, guard, souhrn
#
#   PHASES=A ./gpu_batch.sh          # jen evaluátory (GPU volná ≥ 8 GB)
#   PHASES=B ./gpu_batch.sh          # po ./vllm_vl8b.sh up
#   PHASES=C ./gpu_batch.sh
#   PHASES=ABC ./gpu_batch.sh        # vše (B přeskočí, když :8004 neodpovídá)
#
# Odhad (CPU měření × typický poměr GPU/CPU, ověří se během): A ≈ 10 757 obr. → 20–40 min,
# B ≈ 3 000 obr. při ~0,1–0,3 s/obr. (64 souběžně) → 10–20 min, C ≈ 2–3 min na CPU (změřeno 28. 10., 4 vlákna).
set -uo pipefail
W=/srv/work/w2/uc25; T=$W/ftg/taxo; O=$W/out/gpu; R=~/tmobile-sandbox/uc25
export HF_HOME=/srv/hf HF_HUB_OFFLINE=1
. /cold/venvs/uc25/bin/activate
mkdir -p $O; cd $W
exec 3>&2   # log jde na fd 3 (= původní stderr), takže přesměrování stdout/stderr kroků ho nezachytí
log() { echo "$(date -Is) $*" | tee -a $O/run.log >&3; }
PHASES=${PHASES:-ABC}
M=$O/manifest.jsonl
[ -f $M ] || { python3 $R/collect_images.py --out $O/m_sandbox.jsonl >/dev/null
               cat $O/m_sandbox.jsonl $( [ -f /cold/w2/uc25/calib/manifest.jsonl ] && echo /cold/w2/uc25/calib/manifest.jsonl ) > $M; }
log "manifest $(wc -l < $M) obrázků"
gpu() { nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader | tee -a $O/gpu.log; }

if [[ $PHASES == *A* ]]; then
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "$free" -lt 8000 ] && { log "fáze A: volná VRAM $free MiB < 8000 — končím"; exit 2; }
  gpu
  ( while sleep 15; do gpu >/dev/null; done ) & MON=$!
  python $T/eval_wd14.py   --manifest $M --out $O/wd14.jsonl   --device cuda --batch 32 2>$O/wd14.err;  log "wd14 $? $(grep img/h $O/wd14.err)"
  python $T/eval_nsfw.py   --manifest $M --out $O/nsfw.jsonl   --device cuda --batch 64 2>$O/nsfw.err;  log "nsfw $? $(grep img/h $O/nsfw.err | tr '\n' ' ')"
  python $T/eval_siglip.py --manifest $M --out $O/siglip.jsonl --emb-out /cold/w2/uc25/emb_gpu.npz --device cuda --batch 32 2>$O/siglip.err
  log "siglip $? $(grep img_per_h $O/siglip.err)"
  kill $MON
fi

if [[ $PHASES == *B* ]]; then
  if curl -sf localhost:8004/health >/dev/null; then
    python3 $R/collect_images.py --out $O/m_vlm.jsonl --per-source ${VLM_N:-250} --seed 2 >/dev/null
    [ -f /cold/w2/uc25/calib/manifest.jsonl ] && cat /cold/w2/uc25/calib/manifest.jsonl >> $O/m_vlm.jsonl
    gpu
    python $T/eval_vlm.py --manifest $O/m_vlm.jsonl --out $O/vlm.jsonl --base-url http://127.0.0.1:8004/v1 \
      --model qwen3-vl-8b --concurrency ${VLM_CONC:-64} 2>$O/vlm.err
    rc=$?; log "vlm $rc $(grep '"vlm_model"' $O/vlm.err | tail -1)"; grep -m1 '^first error' $O/vlm.err | cut -c1-600 | tee -a $O/run.log; gpu
  else
    log "fáze B: :8004 neodpovídá (./vllm_vl8b.sh up) — přeskočeno"
  fi
fi

if [[ $PHASES == *C* ]]; then
  # CPU fáze. Vlákna natvrdo ≤ 4: head.py train s výchozími 8 vlákny na 8 vCPU (steal ~1,2 jádra)
  # běžel 28. 10. celou hodinu do timeoutu; se 4 vlákny 200 epoch ≈ 100 s (změřeno 28. 10.).
  export OMP_NUM_THREADS=${C_THREADS:-4} MKL_NUM_THREADS=${C_THREADS:-4} OPENBLAS_NUM_THREADS=${C_THREADS:-4}
  # krok s měřením času a vlastním limitem; výpadek kroku nezastaví zbytek fáze
  step() { local n=$1 lim=$2; shift 2; local t0=$SECONDS
           timeout "$lim" nice -n ${C_NICE:-10} "$@"; local rc=$?; log "C $n rc=$rc $((SECONDS - t0)) s"; return $rc; }
  V=""
  if [ -s $O/vlm.jsonl ]; then
    ok=$(grep -vc '"error"' $O/vlm.jsonl || true)
    if [ "${ok:-0}" -gt 0 ]; then V="--vlm $O/vlm.jsonl"; log "C: VLM $ok OK záznamů"
    else log "C: vlm.jsonl má jen chyby — konsenzus bez VLM"; fi
  fi
  step combine 10m python $T/combine.py --wd14 $O/wd14.jsonl --nsfw $O/nsfw.jsonl --siglip $O/siglip.jsonl $V \
    --out $O/combined.jsonl --stats $O/stats.json 2>>$O/run.log
  step combine_2eval 10m python $T/combine.py --wd14 $O/wd14.jsonl --nsfw $O/nsfw.jsonl --siglip $O/siglip.jsonl \
    --out $O/combined_2eval.jsonl --stats $O/stats_2eval.json 2>>$O/run.log
  step head_train 20m python $T/head.py train --emb /cold/w2/uc25/emb_gpu.npz --labels $O/combined.jsonl --out $O/head \
    --threads ${C_THREADS:-4} 2>>$O/run.log
  step head_predict 5m python $T/head.py predict --emb /cold/w2/uc25/emb_gpu.npz --model $O/head --out $O/head.jsonl
  step head_bench 5m python $T/head.py bench --model $O/head > $O/head_bench.json
  step conformal_fit 10m python $T/conformal.py fit --scores $O/combined.jsonl --pseudo wd14-argmax --wd14 $O/wd14.jsonl --out $O/bands_pseudo.json
  step conformal_synth 5m python $T/conformal.py synth --n 3000 > $O/bands_synth.json
  [ -f /cold/w2/uc25/calib/manifest.jsonl ] && step calib_guard 5m python3 $R/calib_guard.py --combined $O/combined.jsonl \
    --manifest /cold/w2/uc25/calib/manifest.jsonl --out $O/calib_guard.json
  step summarize 5m python3 $R/summarize.py --dir $O --tag ${TAG:-gpu} --out ~/tmobile-sandbox/bench/results ${NOTE:+--note "$NOTE"}
  log "fáze C hotová"
fi
log HOTOVO
