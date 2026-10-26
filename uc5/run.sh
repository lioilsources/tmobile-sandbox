#!/usr/bin/env bash
# UC5(a): benchmark ComfyUI workflow (profil D), každý s vlastním GPU logem → /srv/work/bench/results/UC5_<wf>.json
set -euo pipefail
cd "$(dirname "$0")"; W=/srv/work/uc5; mkdir -p $W
[ "$(cat /srv/work/.gpu_profile 2>/dev/null)" = D ] || ../50_vllm/switch.sh D >/dev/null
until curl -sf localhost:8188/system_stats >/dev/null; do sleep 5; done
IMG=$( (ls /srv/work/uc10/styles/img/*.watercolor.0.jpg /srv/work/comfy-out/uc8/img_*.png 2>/dev/null || true) | head -1)
[ -z "$IMG" ] || cp "$IMG" /srv/comfy/ComfyUI/input/uc5_start.jpg
for spec in flux_schnell:10 sdxl:10 qwen_image:10 ace_step:5 wan_t2v:5 wan_i2v:5; do
  WF=${spec%%:*}; N=${spec##*:}
  [ $WF = wan_i2v ] && [ -z "$IMG" ] && continue
  read -r PID CSV < <(../bench/gpu_log.sh uc5_$WF)
  /srv/work/uc1/venv/bin/python comfy_bench.py --only $WF --reps $N --image uc5_start.jpg --out $W/$WF.json || echo "$WF selhal"
  kill $PID
  [ -f $W/$WF.json ] && ../bench/summarize.py "$CSV" --label UC5_$WF --merge $W/$WF.json | jq -c '{label, median_s: .metriky.median_s, vram_peak_mib, power_avg_w, gpu_util_avg}'
done
