# Witness: monitoring L40S zevnitř VLAN

Malá VM (2 vCPU / 4 GB / 40 GB Standard, Debian 12) ve stejné VLAN jako L40S, **bez veřejné IP**.
Hlídá L40S i když L40S sama nejede, posílá alerty do Telegramu a drží 90 dní metrik.
Bastion i VPN dělá router (`vr-l40s`: NAT, firewall, OpenVPN), witness je jen monitoring.
Adresy, klíče a tokeny patří jen do `witness.env` (gitignored).

```
internet ── router (WAN: veřejná IP; FW: jen OpenVPN + SSH z domova)
              │ LAN 10.10.0.1/24 (VLAN)          VPN klienti 10.20.0.0/24
              ├── L40S    10.10.0.10  (ens4; ens3 má dál veřejnou IP jako zálohu)
              └── witness 10.10.0.2   (Grafana :3000, SSH :22 — jen z VLAN a VPN)
```

| Soubor | Kde běží | Co dělá |
|---|---|---|
| `bootstrap.sh` | witness | jednorázově: uživatel jen s klíčem, sshd, fail2ban, unattended-upgrades, Docker, ufw (22 a 3000 jen z VLAN + VPN), nakonec statická VLAN adresa |
| `compose.yaml` | witness | VictoriaMetrics (90 dní), vmalert, Alertmanager, blackbox, Grafana; vše na 127.0.0.1, Grafana na VLAN IP; limity ~1,7 GB |
| `scrape.yml`, `alerts.yml`, `alertmanager.yml` | witness | sběr z L40S přes VLAN, alerty, Telegram a heartbeat |
| `render.py` | Mac | doplní šablony z `witness.env` → `out/` |
| `../91_witness_proxy/` | L40S | haproxy jen na VLAN IP → exportéry a služby, které jinak poslouchají na 127.0.0.1 |

## Postup

1. **Telegram (uživatel):** bot přes @BotFather → token; chat ID z `getUpdates` po první zprávě botovi.
2. **Mac:** `cp witness.example.env witness.env`, vyplnit (klíč `~/.ssh/id_ed25519_witness.pub`, Telegram, heslo Grafany), `./render.py`.
3. **Portál (uživatel):** VM ze stávajícího klonu Debianu 12, disk zvětšit na 40 GB, jen NIC ve VLAN, SSH klíč `olin-m2`.
   Witness po startu dostane z DHCP routeru adresu z 10.10.0.100–200 (viz portál nebo DHCP leases routeru).
4. **Přes VPN:** `scp -r out/ ol1n@<dhcp-adresa>:witness/` a `ssh ol1n@<dhcp-adresa> 'sudo bash witness/bootstrap.sh'`.
   Na konci se přepne na 10.10.0.2 a spojení spadne — připojit se znovu na 10.10.0.2.
   Disk zvětšený v portálu: `sudo growpart /dev/vda 1 && sudo resize2fs /dev/vda1` (ověřit `lsblk`, ne naslepo).
5. **Witness:** `cd witness && docker compose up -d`; Grafana na `http://10.10.0.2:3000` z VPN.
6. **L40S:** exportéry z `70_monitoring` už běží na 127.0.0.1; na VLAN je zpřístupní
   `cd 91_witness_proxy && VLAN_IP=10.10.0.10 docker compose up -d`. Pokud má L40S zapnutý ufw:
   `ufw allow in from 10.10.0.2 to 10.10.0.10 port 9100,9400,8000,8080,8188 proto tcp`.
7. **Test alertu:** `docker stop vllm` na L40S (mimo běžící úlohu) → zpráva v Telegramu do ~4 min.

Lokální Prometheus a Grafana na L40S (`70_monitoring`) zatím zůstávají jako záloha. Vypnout je jde až po týdnu provozu witness.
