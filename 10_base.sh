#!/usr/bin/env bash
# Základní balíky, timezone, /srv layout, firewall. Idempotentní.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y build-essential git tmux htop nvtop curl jq rclone unzip \
  python3-venv python3-pip pipx linux-headers-$(uname -r) ufw ffmpeg espeak-ng
sudo timedatectl set-timezone Europe/Prague
sudo loginctl enable-linger "$USER"   # systemd --user služby (ComfyUI) běží i bez přihlášení

# QEMU guest agent — CloudSigma nemusí mít virtio kanál; instalace nevadí
sudo apt-get install -y qemu-guest-agent
[ -e /dev/virtio-ports/org.qemu.guest_agent.0 ] && sudo systemctl enable --now qemu-guest-agent \
  || echo "WARN: virtio kanál guest agenta chybí (hypervizor ho nenabízí)"

# /srv layout (disk je jeden, adresáře jen logicky)
sudo mkdir -p /srv/{hf,comfy,docker,work,backup}
sudo chown "$USER:$USER" /srv/{hf,comfy,work,backup}
grep -q 'HF_HOME=/srv/hf' ~/.bashrc || echo 'export HF_HOME=/srv/hf' >> ~/.bashrc
grep -q '.local/bin' ~/.bashrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Firewall: ven jen SSH; vše ostatní přes SSH tunel.
# Docker -p obchází ufw → všechny služby publikujeme na 127.0.0.1.
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp
sudo ufw --force enable
sudo ufw status verbose
