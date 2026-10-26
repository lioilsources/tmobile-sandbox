#!/usr/bin/env bash
# UC1: QLoRA fine-tune (train_qlora.py) → merge → benchmark přes vLLM profil O.
#   ft_chain.sh <label> <hf_base> [epochs=3] [rank=16]
set -euo pipefail
cd "$(dirname "$0")"; L=$1; B=$2; E=${3:-3}; R=${4:-16}
OUT=/srv/hf/ft/$L
[ "$(cat /srv/work/.gpu_profile 2>/dev/null)" = off ] || ../50_vllm/switch.sh off >/dev/null
if [ ! -f $OUT-merged/config.json ]; then
  read -r PID CSV < <(../bench/gpu_log.sh uc1_train_$L)
  HF_HOME=/srv/hf /srv/work/uc1/ft-venv/bin/python train_qlora.py --base $B --epochs $E --rank $R --output $OUT --merge
  kill $PID
  ../bench/summarize.py "$CSV" --label UC1_train_$L --merge $OUT/train_meta.json >/dev/null
  cp $OUT/train_meta.json /srv/work/bench/results/UC1_train_${L}_meta.json
fi
./bench_engine.sh $L /root/.cache/huggingface/ft/$L-merged 16 "" fix
