#!/usr/bin/env python3
"""Doplní šablony witness z witness.env → out/ (bootstrap.sh, scrape.yml, alertmanager.yml, .env pro compose)
a připraví Grafanu (datasource VictoriaMetrics, dashboard L40S Sandbox z 70_monitoring).

  render.py [witness.env]
Bez Telegramu / heartbeatu se příslušný receiver nahradí prázdným (alerty jsou vidět jen v UI Alertmanageru).
"""
import pathlib, re, shutil, string, sys

H = pathlib.Path(__file__).parent
envf = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else H / "witness.env")
env = {k.strip(): v.split("#")[0].strip() for k, v in
       (l.split("=", 1) for l in envf.read_text().splitlines() if "=" in l and not l.lstrip().startswith("#"))}
for k in ("WITNESS_VLAN_IP", "L40S_VLAN_IP", "VLAN_PREFIX", "VLAN_NET", "VLAN_GW", "VPN_NET", "DNS", "SSH_PUBKEY", "GRAFANA_PASSWORD"):
    if not env.get(k): sys.exit(f"{envf}: chybí {k}")

out = H / "out"; shutil.rmtree(out, ignore_errors=True); out.mkdir()
for n in ("bootstrap.sh", "scrape.yml", "alertmanager.yml"):
    (out / n).write_text(string.Template((H / n).read_text()).safe_substitute(env))

am = (out / "alertmanager.yml").read_text()
if not env.get("TELEGRAM_BOT_TOKEN"):
    am = re.sub(r"  - name: telegram\n(    .*\n)+", "  - name: telegram\n", am)
if not env.get("HEARTBEAT_URL"):
    am = re.sub(r"  - name: heartbeat\n(    .*\n)+", "  - name: heartbeat\n", am)
(out / "alertmanager.yml").write_text(am)

for n in ("compose.yaml", "alerts.yml", "blackbox.yml"):
    shutil.copy(H / n, out / n)
(out / ".env").write_text(f"GRAFANA_PASSWORD={env['GRAFANA_PASSWORD']}\nWITNESS_VLAN_IP={env['WITNESS_VLAN_IP']}\n")
g = out / "grafana"
(g / "provisioning/datasources").mkdir(parents=True); (g / "provisioning/dashboards").mkdir(parents=True); (g / "dashboards").mkdir()
(g / "provisioning/datasources/vm.yaml").write_text(
    "apiVersion: 1\ndatasources:\n  - { name: VictoriaMetrics, uid: prom, type: prometheus, url: http://vm:8428, isDefault: true }\n")
(g / "provisioning/dashboards/files.yaml").write_text(
    "apiVersion: 1\nproviders:\n  - { name: files, type: file, options: { path: /var/lib/grafana/dashboards } }\n")
# dashboard L40S: node_* metriky omezit na L40S (witness má vlastní node_exporter a dashboard)
sb = (H.parent / "70_monitoring/grafana/dashboards/sandbox.json").read_text()
sb = re.sub(r'(node_[A-Za-z0-9_]+)\{', r'\1{host=\\"ubuntu-l40s\\",', sb)
sb = re.sub(r'(node_[A-Za-z0-9_]+)(?![A-Za-z0-9_{])', r'\1{host=\\"ubuntu-l40s\\"}', sb)
(g / "dashboards/sandbox.json").write_text(sb)
shutil.copy(H / "witness_dashboard.json", g / "dashboards/witness.json")
print(f"hotovo → {out}")
