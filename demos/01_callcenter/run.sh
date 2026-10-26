#!/usr/bin/env bash
# Demo 1 end-to-end (profil B + TTS): scénáře → audio → přepis + analýza → vyhodnocení.
#   ./run.sh [souběh=8] [počet hovorů=40]
set -euo pipefail
cd "$(dirname "$0")"; D=/srv/work/demo1; PY=$D/venv/bin/python
[ -d $D/venv ] || ./setup.sh
curl -sf localhost:8102/health >/dev/null || ../../55_tts/up.sh
[ "$(cat /srv/work/.gpu_profile 2>/dev/null)" = B ] || ../../50_vllm/switch.sh B
[ -d $D/calls ] || $PY scenarios.py --n ${2:-40} --out $D/calls   # existující (validovanou) sadu nedoplňovat
[ "$(ls $D/audio/*.wav 2>/dev/null | wc -l)" -ge "$(ls $D/calls/*.json | wc -l)" ] || $PY synth.py --calls $D/calls --out $D/audio
mkdir -p $D/runs; RUN=$D/runs/$(date +%Y%m%d-%H%M%S)-c${1:-8}
$PY process.py --audio $D/audio --out $RUN --concurrency ${1:-8} > $RUN.log
$PY evaluate.py --calls $D/calls --run $RUN ${KC_GPU_H:+--kc-per-gpu-hour $KC_GPU_H}
echo "run: $RUN"
