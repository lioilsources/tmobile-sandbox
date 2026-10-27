# Witness: monitoring a bastion vedle L40S

Malá VM (2 vCPU / 4 GB / 50 GB Standard, Ubuntu 24.04) podle `Prompts/02-WITNESS_PLAN.md`. Hlídá L40S, posílá alerty
a slouží jako jediný SSH vstup. Veřejné IP a tokeny patří jen do `witness.env` (gitignored).

| Soubor | Kde běží | Co dělá |
|---|---|---|
| `cloud-init.yaml` | witness | uživatel jen s klíčem, bez roota a hesel, ufw, fail2ban, unattended-upgrades, docker |
| `compose.yaml` | witness | VictoriaMetrics (90 dní), vmalert, Alertmanager, blackbox, Grafana; vše na 127.0.0.1, ~1,7 GB limitů |
| `scrape.yml`, `alerts.yml`, `alertmanager.yml` | witness | sběr z L40S přes VLAN, alerty z plánu §5, Telegram a heartbeat |
| `render.py` | Mac | doplní šablony z `witness.env` → `out/` |
| `../91_witness_proxy/` | L40S | haproxy jen na VLAN IP → exportéry a služby, které jinak poslouchají na 127.0.0.1 |

## Postup

1. **Mac:** `cp witness.example.env witness.env`, vyplnit. Klíč: `~/.ssh/id_ed25519_witness`. Pak `./render.py`.
2. **Portál (uživatel):** VM `witness` podle plánu §3, user-data = `out/cloud-init.yaml`, NIC1 veřejná IP,
   NIC2 private VLAN. Zkontrolovat Usage & Purchasing (CPU/RAM bez burstu).
3. **Witness:** `ssh witness`, `ip a` ukazuje obě rozhraní. VLAN adresa staticky přes netplan, pokud ji nepřidělí portál.
   `rsync -a out/ witness:~/witness/ && ssh witness 'cd witness && docker compose up -d'`.
4. **L40S, v okně mezi běhy (shutdown):** přidat NIC ve stejné VLAN, netplan `10.10.0.10/24`, ping na witness.
   Pak `cd 70_monitoring && docker compose up -d` (exportéry na 127.0.0.1) a
   `cd 91_witness_proxy && VLAN_IP=10.10.0.10 docker compose up -d`.
5. **Mac `~/.ssh/config`:** `Host witness` (veřejná IP, klíč witness) a `Host l40s` s `ProxyJump witness` + `HostName 10.10.0.10`.
   Tunel `tcloud-tunnel` (launchd) přepnout na `ProxyJump witness`, Grafana pak z witness (`LocalForward 3002 localhost:3001`).
6. **Teprve po ověření ProxyJump:** na L40S `ufw allow in on <vlan-if> to any port 22`, `ufw deny 22`, `ufw enable`.
   Do té doby nechat přímý SSH, ať se nezamkneme.
7. **Test alertu:** `docker stop vllm` na L40S → zpráva v Telegramu do ~4 min (3 min `for` + 30 s group_wait).

Lokální Prometheus a Grafana na L40S (`70_monitoring`) zatím zůstávají jako záloha. Vypnout je jde až potom, co witness poběží týden.
