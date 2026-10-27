#!/usr/bin/env python3
"""Doplní šablony witness z witness.env → out/ (cloud-init.yaml, scrape.yml, alertmanager.yml, .env pro compose)
a připraví Grafanu (datasource VictoriaMetrics, dashboard L40S Sandbox z 70_monitoring).

  render.py [witness.env]
Bez Telegramu / heartbeatu se příslušný receiver nahradí prázdným (alerty jsou vidět jen v UI Alertmanageru).
"""
import pathlib, re, shutil, string, sys

H = pathlib.Path(__file__).parent
envf = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else H / "witness.env")
env = {k.strip(): v.split("#")[0].strip() for k, v in
       (l.split("=", 1) for l in envf.read_text().splitlines() if "=" in l and not l.lstrip().startswith("#"))}
for k in ("WITNESS_VLAN_IP", "L40S_VLAN_IP", "SSH_PUBKEY", "GRAFANA_PASSWORD"):
    if not env.get(k): sys.exit(f"{envf}: chybí {k}")
env.setdefault("SSH_PORT", "22")

out = H / "out"; shutil.rmtree(out, ignore_errors=True); out.mkdir()
for n in ("cloud-init.yaml", "scrape.yml", "alertmanager.yml"):
    (out / n).write_text(string.Template((H / n).read_text()).safe_substitute(env))

am = (out / "alertmanager.yml").read_text()
if not env.get("TELEGRAM_BOT_TOKEN"):
    am = re.sub(r"  - name: telegram\n(    .*\n)+", "  - name: telegram\n", am)
if not env.get("HEARTBEAT_URL"):
    am = re.sub(r"  - name: heartbeat\n(    .*\n)+", "  - name: heartbeat\n", am)
(out / "alertmanager.yml").write_text(am)

for n in ("compose.yaml", "alerts.yml", "blackbox.yml"):
    shutil.copy(H / n, out / n)
(out / ".env").write_text(f"GRAFANA_PASSWORD={env['GRAFANA_PASSWORD']}\n")
g = out / "grafana"
(g / "provisioning/datasources").mkdir(parents=True); (g / "provisioning/dashboards").mkdir(parents=True); (g / "dashboards").mkdir()
(g / "provisioning/datasources/vm.yaml").write_text(
    "apiVersion: 1\ndatasources:\n  - { name: VictoriaMetrics, uid: prom, type: prometheus, url: http://vm:8428, isDefault: true }\n")
(g / "provisioning/dashboards/files.yaml").write_text(
    "apiVersion: 1\nproviders:\n  - { name: files, type: file, options: { path: /var/lib/grafana/dashboards } }\n")
shutil.copy(H.parent / "70_monitoring/grafana/dashboards/sandbox.json", g / "dashboards/sandbox.json")
print(f"hotovo → {out}")
