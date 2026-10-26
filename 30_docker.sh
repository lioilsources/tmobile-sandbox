#!/usr/bin/env bash
# Docker + NVIDIA Container Toolkit, data-root na /srv/docker. Idempotentní.
set -euo pipefail
command -v docker >/dev/null || curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"

KR=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
[ -f $KR ] || curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o $KR
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed "s#deb https://#deb [signed-by=$KR] https://#g" \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

sudo mkdir -p /etc/docker
[ -f /etc/docker/daemon.json ] || echo '{}' | sudo tee /etc/docker/daemon.json >/dev/null
sudo nvidia-ctk runtime configure --runtime=docker
tmp=$(mktemp)
sudo jq '. + {"data-root":"/srv/docker","log-driver":"json-file","log-opts":{"max-size":"50m","max-file":"3"}}' \
  /etc/docker/daemon.json > "$tmp" && sudo mv "$tmp" /etc/docker/daemon.json
sudo systemctl restart docker
sudo docker info --format 'root={{.DockerRootDir}} runtimes={{json .Runtimes}}'
sudo docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi -L
