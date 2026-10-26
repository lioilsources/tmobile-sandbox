#!/usr/bin/env bash
# Spustí monitoring; stáhne Grafana dashboardy (12239 DCGM, 1860 node, vLLM z repa vllm).
set -euo pipefail
cd "$(dirname "$0")"
docker network inspect tms >/dev/null 2>&1 || docker network create tms >/dev/null
[ -f .env ] || echo "GRAFANA_PASSWORD=$(openssl rand -hex 12)" > .env
mkdir -p grafana/dashboards
fix() { jq 'walk(if type=="object" and has("datasource") and (.datasource|type)=="string" then .datasource={"type":"prometheus","uid":"prom"} else . end)
         | walk(if type=="string" then gsub("\\$\\{DS_PROMETHEUS\\}";"prom") else . end) | del(.__inputs) | .id=null'; }
for id in 12239 1860; do
  [ -s grafana/dashboards/$id.json ] || curl -sfL "https://grafana.com/api/dashboards/$id/revisions/latest/download" | fix > grafana/dashboards/$id.json
done
[ -s grafana/dashboards/vllm.json ] || curl -sfL https://raw.githubusercontent.com/vllm-project/vllm/main/examples/observability/prometheus_grafana/grafana.json | fix > grafana/dashboards/vllm.json || true
docker compose up -d
docker compose ps --format '{{.Service}}\t{{.Status}}'
echo "Grafana admin heslo: $(cut -d= -f2 .env)"
