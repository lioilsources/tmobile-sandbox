#!/usr/bin/env bash
# Druhý disk (vdb, LABEL=data, 650 GB) → /data. Přesune /srv (HF cache, Comfy, docker data-root, work)
# a /var/lib/containerd (docker image) na /data a bind-mountne zpět → ostatní skripty beze změny. Idempotentní.
set -euo pipefail
findmnt -n /data >/dev/null || { echo "/data není připojený"; exit 1; }
REPO=$(cd "$(dirname "$0")" && pwd)
OPT="none bind,nofail,x-systemd.requires-mounts-for=/data 0 0"

move() {  # move <zdroj> <cíl na /data>
  local src=$1 dst=$2
  if [ "$(findmnt -n -o SOURCE --target "$src" | grep -c vdb)" = 1 ]; then echo "$src už je na /data"; return; fi
  echo "==> $src → $dst"
  sudo mkdir -p "$dst"
  sudo rsync -aHAX --info=progress2 "$src/" "$dst/"
  sudo mv "$src" "$src.old" && sudo mkdir -p "$src"
  grep -q " $src none bind" /etc/fstab || echo "$dst $src $OPT" | sudo tee -a /etc/fstab >/dev/null
  sudo mount "$src"
}

"$REPO/50_vllm/switch.sh" off || true
(cd "$REPO/70_monitoring" && docker compose stop) || true
sudo systemctl stop docker.socket docker containerd

move /srv /data/srv
move /var/lib/containerd /data/containerd
sudo systemctl daemon-reload
sudo systemctl start containerd docker

docker images --format '{{.Repository}}:{{.Tag}}' | wc -l
(cd "$REPO/70_monitoring" && docker compose up -d)
sudo rm -rf /srv.old /var/lib/containerd.old
df -h / /data
