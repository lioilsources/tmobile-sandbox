#!/usr/bin/env bash
# NVIDIA driver (PCI passthrough). Přeskočí, pokud už nvidia-smi funguje.
set -euo pipefail
if nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  exit 0
fi
mokutil --sb-state 2>/dev/null || true
sudo apt-get install -y nvidia-driver-570-server nvidia-utils-570-server
echo "Driver nainstalován → sudo reboot, pak nvidia-smi"
