#!/usr/bin/env bash
# Witness (Debian 12, klon disku bez cloud-init): jednorázové nastavení, idempotentní.
# Šablona: ${...} doplní render.py z witness.env → out/bootstrap.sh. Spouštět jako root na witness:
#   sudo bash bootstrap.sh
# Witness nemá veřejnou IP: jen VLAN ${WITNESS_VLAN_IP}/${VLAN_PREFIX}, brána = router ${VLAN_GW},
# přístup z VPN (${VPN_NET}) přes router. Bastion dělá router → witness je jen monitoring.
set -euo pipefail

# --- balíky (odchozí provoz z LAN povolený routerem)
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get -y upgrade
apt-get -y install ufw fail2ban unattended-upgrades jq curl ca-certificates gnupg
if ! command -v docker >/dev/null; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update
  apt-get -y install docker-ce docker-ce-cli containerd.io docker-compose-plugin
fi

# --- uživatel jen s klíčem
id ol1n >/dev/null 2>&1 || useradd -m -s /bin/bash -G sudo,docker ol1n
usermod -aG sudo,docker ol1n
install -d -m 700 -o ol1n -g ol1n /home/ol1n/.ssh
grep -qF "${SSH_PUBKEY}" /home/ol1n/.ssh/authorized_keys 2>/dev/null || echo "${SSH_PUBKEY}" >> /home/ol1n/.ssh/authorized_keys
chown ol1n:ol1n /home/ol1n/.ssh/authorized_keys; chmod 600 /home/ol1n/.ssh/authorized_keys
echo 'ol1n ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/90-ol1n; chmod 440 /etc/sudoers.d/90-ol1n
passwd -l ol1n >/dev/null

cat > /etc/ssh/sshd_config.d/10-witness.conf <<EOF
PermitRootLogin no
PasswordAuthentication no
KbdInteractiveAuthentication no
AllowUsers ol1n
AllowTcpForwarding yes
X11Forwarding no
AllowAgentForwarding no
EOF
cat > /etc/fail2ban/jail.d/sshd.local <<EOF
[sshd]
enabled = true
backend = systemd
maxretry = 5
bantime = 1h
EOF
cat > /etc/apt/apt.conf.d/20auto-upgrades <<EOF
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
EOF

# --- firewall: SSH a Grafana jen z VLAN a VPN. Docker publikuje Grafanu jen na VLAN IP (compose),
# takže ufw ji neobchází z jiné sítě. Pozor: pravidla dřív, než se ufw zapne, ať se nezamkneme.
ufw default deny incoming
ufw default allow outgoing
for net in "${VLAN_NET}" "${VPN_NET}"; do
  ufw allow from "$net" to any port 22 proto tcp
  ufw allow from "$net" to any port 3000 proto tcp
done
ufw --force enable

systemctl restart ssh
systemctl enable --now docker fail2ban unattended-upgrades
echo "witness bootstrap hotový: $(hostname)"

# --- síť až na konec: přepnutí z DHCP na statickou adresu přeruší SSH, které skript spustilo.
# Rozhraní = první ne-lo (ifupdown, Debian default).
IF=$(ip -o link show | awk -F': ' '$2!="lo"{print $2; exit}')
if ! grep -q "${WITNESS_VLAN_IP}" /etc/network/interfaces 2>/dev/null; then
  cp -a /etc/network/interfaces /etc/network/interfaces.bak-witness 2>/dev/null || true
  cat > /etc/network/interfaces <<EOF
auto lo
iface lo inet loopback

auto $IF
iface $IF inet static
    address ${WITNESS_VLAN_IP}/${VLAN_PREFIX}
    gateway ${VLAN_GW}
EOF
  printf 'nameserver %s\nnameserver 1.1.1.1\n' "${DNS}" > /etc/resolv.conf
  echo "síť: $IF → ${WITNESS_VLAN_IP} (spojení z DHCP adresy se teď přeruší, připoj se na ${WITNESS_VLAN_IP})"
  systemctl restart networking || { ifdown "$IF"; ifup "$IF"; }
fi

