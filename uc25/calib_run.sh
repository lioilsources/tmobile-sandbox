#!/usr/bin/env bash
# UC25 kalibrační sada — QUEUE-READY. Potřebuje profil D (ComfyUI na :8188) = přepíná Director.
# ~882 obrázků (6 checkpointů × 147), SDXL 28–30 kroků ≈ 4–6 s, FLUX.1-dev 20 kroků ≈ 10–15 s
# → odhad 1,5–2 h; VRAM ComfyUI ~20–35 GB + evaluátory guardu ~4 GB.
# Hned po generování: WD14 + NSFW (GPU) na kalibrační sadě → calib_guard.py smaže každý obrázek
# se signálem nezletilosti mimo jisté safe. Výstupy jen /cold/w2/uc25/calib (safe/, private/).
set -uo pipefail
W=/srv/work/w2/uc25; T=$W/ftg/taxo; R=~/tmobile-sandbox/uc25; C=/cold/w2/uc25/calib
export HF_HOME=/srv/hf HF_HUB_OFFLINE=1
. /cold/venvs/uc25/bin/activate
mkdir -p $C/eval
log() { echo "$(date -Is) $*" | tee -a $C/run.log; }
until curl -sf localhost:8188/system_stats >/dev/null; do sleep 5; done
log "ComfyUI ok, start"
python3 $R/calib_gen.py --out $C --per-ckpt ${PER_CKPT:-120} --tiers ${TIERS:-8} >> $C/run.log 2>&1; log "gen $?"
python $T/eval_wd14.py --manifest $C/manifest.jsonl --out $C/eval/wd14.jsonl --device cuda --batch 16 2>$C/eval/wd14.err
python $T/eval_nsfw.py --manifest $C/manifest.jsonl --out $C/eval/nsfw.jsonl --device cuda 2>$C/eval/nsfw.err
: > $C/eval/siglip_empty.jsonl
python $T/combine.py --wd14 $C/eval/wd14.jsonl --nsfw $C/eval/nsfw.jsonl --siglip $C/eval/siglip_empty.jsonl \
  --out $C/eval/combined_guard.jsonl --stats $C/eval/stats_guard.json 2>>$C/run.log
python3 $R/calib_guard.py --combined $C/eval/combined_guard.jsonl --manifest $C/manifest.jsonl --out $C/eval/guard1.json | tee -a $C/run.log
log "HOTOVO (calib) — druhý guard proběhne v gpu_batch.sh fáze C se všemi třemi hodnotiteli"
