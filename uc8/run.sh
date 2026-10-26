#!/usr/bin/env bash
# UC8: 200 UGC předmětů obrázek → 3D (FLUX-schnell + Hunyuan3D 2.1 v ComfyUI, profil D) s GPU logem.
set -euo pipefail
cd "$(dirname "$0")"; W=/srv/work/uc8; mkdir -p $W
[ -x $W/venv/bin/python ] || { python3 -m venv $W/venv && $W/venv/bin/pip -q install trimesh; }
[ -f /srv/comfy/ComfyUI/models/checkpoints/hunyuan_3d_v2.1.safetensors ] || { echo "chybí Hunyuan3D checkpoint"; exit 1; }
[ "$(cat /srv/work/.gpu_profile 2>/dev/null)" = D ] || ../50_vllm/switch.sh D >/dev/null
until curl -sf localhost:8188/system_stats >/dev/null; do sleep 5; done
read -r PID CSV < <(../bench/gpu_log.sh uc8_3d)
if [ -s $W/run.log ]; then cat $W/run.log >> $W/run.prev.log; fi
$W/venv/bin/python gen3d.py --out $W --n ${1:-200} --resume $W/run.prev.log | tee $W/run.log | tail -1
kill $PID
../bench/summarize.py "$CSV" --label UC8_hunyuan3d --extra "{\"predmetu\": $(jq .predmetu $W/report.json)}" --merge $W/report.json
