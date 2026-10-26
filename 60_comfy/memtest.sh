#!/usr/bin/env bash
# Porovná paměťové nastavení ComfyUI (RAM/swap/VRAM špička + časy) na FLUX a Wan smoke testu.
#   memtest.sh "<COMFY_ARGS varianta 1>" "<varianta 2>" ...   (prázdný řetězec = výchozí)
# Vyžaduje profil D. Na konci obnoví ~/.config/comfyui.env na původní hodnotu.
set -uo pipefail
cd "$(dirname "$0")"; ENV=~/.config/comfyui.env; ORIG=$(cat $ENV 2>/dev/null || true)
OUT=/srv/work/comfy-memtest/$(date +%Y%m%d-%H%M); mkdir -p $OUT; PY=/srv/comfy/ComfyUI/venv/bin/python
for ARGS in "$@"; do
  echo "COMFY_ARGS=$ARGS" > $ENV
  sudo swapoff -a && sudo swapon -a          # vynulovat swap, ať je špička férová
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  systemctl --user restart comfyui
  for i in $(seq 60); do curl -sf localhost:8188/system_stats >/dev/null && break; sleep 2; done
  tag=${ARGS:-default}; tag=${tag//[^a-z-]/_}
  ( while sleep 2; do PID=$(systemctl --user show -p MainPID --value comfyui)
      echo "$(free -m | awk '/Mem/{print $3}') $(free -m | awk '/Swap/{print $3}') $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) $(ps -o rss= -p $PID 2>/dev/null | awk '{print int($1/1024)}')"
    done > $OUT/$tag.mem ) & MP=$!
  { echo "== $tag"; $PY smoke_flux.py; $PY smoke_wan.py; $PY smoke_flux.py | sed 's/^/po Wan /'; } 2>&1 | tee $OUT/$tag.log
  kill $MP
  for c in 1 2 3 4; do printf "%s " "$(sort -n -k$c $OUT/$tag.mem | tail -1 | cut -d' ' -f$c)"; done | \
    awk -v t="$tag" '{printf "%-40s RAM %6d MiB  swap %5d MiB  VRAM %6d MiB  comfy RSS %6d MiB\n", t, $1, $2, $3, $4}' | tee -a $OUT/summary.txt
done
[ -n "$ORIG" ] && echo "$ORIG" > $ENV || rm -f $ENV
systemctl --user restart comfyui
cat $OUT/summary.txt
