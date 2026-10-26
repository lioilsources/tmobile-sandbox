#!/usr/bin/env bash
# Demo 3 end-to-end: (generovat) → přepnout na profil C → extrakce → vyhodnocení.
#   ./run.sh [souběh=8] [Kč/GPU-hod]     (DPI=150 PROMPT=v2 v prostředí)
set -euo pipefail
cd "$(dirname "$0")"; D=/srv/work/demo3; PY=$D/venv/bin/python
[ -d $D/venv ] || ./setup.sh
[ -d $D/docs ] || $PY generate.py --out $D/docs
[ "$(cat /srv/work/.gpu_profile 2>/dev/null)" = C ] && curl -sf localhost:8000/health >/dev/null || ../../50_vllm/switch.sh C
mkdir -p $D/runs; RUN=$D/runs/$(date +%Y%m%d-%H%M%S)-c${1:-8}-d${DPI:-150}-${PROMPT:-v2}
$PY extract.py --docs $D/docs --out $RUN --concurrency ${1:-8} --dpi ${DPI:-150} --prompt ${PROMPT:-v2} > $RUN.log
$PY evaluate.py --docs $D/docs --run $RUN ${2:+--kc-per-gpu-hour $2}
echo "report: $RUN/report.md"
