#!/usr/bin/env bash
# Sada v2: scénáře Qwen3-32B (A) → validace rozhodčím (A) → audio → zpracování v produkčním profilu B.
set -euo pipefail
cd "$(dirname "$0")"; D=/srv/work/demo1; PY=$D/venv/bin/python
# předchozí sadu odložit do $D/<VER_OLD>/ (v1 = generátor 30B-A3B, v2 = 32B bez definic štítků)
for old in ${ARCHIVE:-}; do [ -d $D/calls ] && { mkdir -p $D/$old; mv $D/calls $D/calls_rejected $D/audio $D/validace.json $D/$old/ 2>/dev/null || true; }; done
[ "$(cat /srv/work/.gpu_profile)" = A ] || ../../50_vllm/switch.sh A
$PY scenarios.py --n ${N:-150} --out $D/calls          # navazuje
$PY validate.py --calls $D/calls --keep 50
../../50_vllm/switch.sh B
$PY synth.py --calls $D/calls --out $D/audio
./run.sh 8
