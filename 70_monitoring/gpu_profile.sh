#!/usr/bin/env bash
# Aktivní GPU profil (/srv/work/.gpu_profile) → textfile metriky pro node_exporter. Volá ho 50_vllm/switch.sh.
#   l40s_gpu_profile{profile="D"} 1
#   l40s_service_expected{service="comfyui|vllm"} 0/1   — witness alertuje ServiceDown jen u služby, která má běžet
set -euo pipefail
D=/srv/work/metrics; mkdir -p "$D"
p=$(awk '{print $1}' /srv/work/.gpu_profile 2>/dev/null | tr -dc 'A-Za-z0-9_' || true); p=${p:-off}
comfy=0; vllm=0
case "$p" in
  D) comfy=1 ;;
  off) ;;
  *) vllm=1 ;;     # A B C L O S W …: vLLM na :8000
esac
{
  echo "# HELP l40s_gpu_profile Active GPU profile (1 = active)."
  echo "# TYPE l40s_gpu_profile gauge"
  echo "l40s_gpu_profile{profile=\"$p\"} 1"
  echo "# HELP l40s_service_expected Service that the active GPU profile should be running."
  echo "# TYPE l40s_service_expected gauge"
  echo "l40s_service_expected{service=\"comfyui\"} $comfy"
  echo "l40s_service_expected{service=\"vllm\"} $vllm"
} > "$D/gpu_profile.prom.tmp"
mv "$D/gpu_profile.prom.tmp" "$D/gpu_profile.prom"
