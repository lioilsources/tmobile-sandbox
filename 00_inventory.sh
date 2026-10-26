#!/usr/bin/env bash
# Inventura VM — bez sudo, bezpečné spouštět kdykoli.
set -u
echo "== OS";      grep PRETTY /etc/os-release; uname -r; systemd-detect-virt
echo "== GPU";     lspci -nnk | grep -A3 -i nvidia; command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "== CPU/RAM"; nproc; free -h
echo "== Disk";    lsblk; df -h / /srv 2>/dev/null
echo "== Síť";     ip -br a; grep nameserver /etc/resolv.conf
echo "== Egress"
for u in https://huggingface.co https://github.com https://pypi.org https://download.docker.com https://registry-1.docker.io/v2/ https://nvcr.io; do
  printf '%-40s %s\n' "$u" "$(curl -sI -m 10 "$u" | head -1 | tr -d '\r')"
done
echo "== Služby";  systemctl is-active docker qemu-guest-agent 2>/dev/null
command -v docker >/dev/null && docker ps --format '{{.Names}}\t{{.Status}}'
