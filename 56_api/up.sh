#!/usr/bin/env bash
# Build + start gateway a LiteLLM; při prvním běhu vygeneruje API klíče tenantů. Idempotentní.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p /srv/work/demo2
if [ ! -f tenants.json ]; then
  jq --arg a "sk-a-$(openssl rand -hex 16)" --arg b "sk-b-$(openssl rand -hex 16)" --arg c "sk-c-$(openssl rand -hex 16)" \
    'del(._poznamka) | .tenants[0].keys=[$a] | .tenants[1].keys=[$b] | .tenants[2].keys=[$c]' tenants.example.json > tenants.json
  chmod 600 tenants.json
fi
[ -f .env ] || { echo "LITELLM_MASTER_KEY=sk-$(openssl rand -hex 32)" > .env; chmod 600 .env; }
docker network inspect tms >/dev/null 2>&1 || docker network create tms >/dev/null
docker compose up -d --build --wait
grep -q ai-gateway ../70_monitoring/prometheus.yml || echo '  - { job_name: ai-gateway, static_configs: [{ targets: ["ai-gateway:8080"] }] }' >> ../70_monitoring/prometheus.yml
docker compose -f ../70_monitoring/compose.yaml restart prometheus >/dev/null
curl -s localhost:8080/health; echo
jq -r '.tenants[] | "\(.name)\t\(.keys[0])"' tenants.json
