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
# B ≈ 3 000 obr. při ~0,1–0,3 s/obr. (64 souběžně) → 10–20 min, C < 10 min.
set -uo pipefail
W=/srv/work/w2/uc25; T=$W/ftg/taxo; O=$W/out/gpu; R=~/tmobile-sandbox/uc25
export HF_HOME=/srv/hf HF_HUB_OFFLINE=1
. /cold/venvs/uc25/bin/activate
mkdir -p $O; cd $W
log() { echo "$(date -Is) $*" | tee -a $O/run.log; }
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
    log "vlm $? $(tail -1 $O/vlm.err)"; gpu
  else
    log "fáze B: :8004 neodpovídá (./vllm_vl8b.sh up) — přeskočeno"
  fi
fi

if [[ $PHASES == *C* ]]; then
  V=""; [ -s $O/vlm.jsonl ] && V="--vlm $O/vlm.jsonl"
  python $T/combine.py --wd14 $O/wd14.jsonl --nsfw $O/nsfw.jsonl --siglip $O/siglip.jsonl $V \
    --out $O/combined.jsonl --stats $O/stats.json 2>>$O/run.log
  python $T/combine.py --wd14 $O/wd14.jsonl --nsfw $O/nsfw.jsonl --siglip $O/siglip.jsonl \
    --out $O/combined_2eval.jsonl --stats $O/stats_2eval.json 2>>$O/run.log
  python $T/head.py train --emb /cold/w2/uc25/emb_gpu.npz --labels $O/combined.jsonl --out $O/head 2>>$O/run.log
  python $T/head.py predict --emb /cold/w2/uc25/emb_gpu.npz --model $O/head --out $O/head.jsonl
  python $T/head.py bench --model $O/head > $O/head_bench.json
  python $T/conformal.py fit --scores $O/combined.jsonl --pseudo wd14-argmax --wd14 $O/wd14.jsonl --out $O/bands_pseudo.json
  python $T/conformal.py synth --n 3000 > $O/bands_synth.json
  [ -f /cold/w2/uc25/calib/manifest.jsonl ] && python3 $R/calib_guard.py --combined $O/combined.jsonl \
    --manifest /cold/w2/uc25/calib/manifest.jsonl --out $O/calib_guard.json
  python3 $R/summarize.py --dir $O --tag gpu --out ~/tmobile-sandbox/bench/results
  log "fáze C hotová"
fi
log HOTOVO
