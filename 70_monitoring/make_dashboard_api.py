#!/usr/bin/env python3
"""Grafana dashboard „AI API – tenanti“ (metriky gateway aigw_*) → grafana/dashboards/api.json."""
import json, pathlib
DS = {"type": "prometheus", "uid": "prom"}
panels = []

def p(title, exprs, unit, x, y, w, h, kind="timeseries", stack=False, decimals=None):
    fc = {"unit": unit, "min": 0, **({"decimals": decimals} if decimals is not None else {})}
    if kind == "timeseries":
        fc["custom"] = {"fillOpacity": 25 if stack else 10, "lineWidth": 2, "showPoints": "never", **({"stacking": {"mode": "normal"}} if stack else {})}
    panels.append({"type": kind, "title": title, "datasource": DS, "gridPos": {"x": x, "y": y, "w": w, "h": h},
                   "fieldConfig": {"defaults": fc, "overrides": []},
                   "options": {"legend": {"displayMode": "table", "placement": "right", "calcs": ["lastNotNull", "max"]} if kind == "timeseries" else {},
                               "reduceOptions": {"calcs": ["lastNotNull"]}, "textMode": "value_and_name"},
                   "targets": [{"expr": e, "legendFormat": l, "refId": chr(65 + i), "datasource": DS} for i, (e, l) in enumerate(exprs)]})

R = "$__rate_interval"
# řádek 1: souhrn za zvolené období
p("Požadavky (období)", [("sum by (tenant) (increase(aigw_requests_total[$__range]))", "{{tenant}}")], "short", 0, 0, 6, 5, "stat", decimals=0)
p("Tokeny (období)", [("sum by (tenant) (increase(aigw_tokens_total[$__range]))", "{{tenant}}")], "short", 6, 0, 6, 5, "stat", decimals=0)
p("Účtováno Kč (období, ilustrativní ceník)", [("sum by (tenant) (increase(aigw_cost_czk_total[$__range]))", "{{tenant}}")], "currencyCZK", 12, 0, 6, 5, "stat", decimals=2)
p("Odmítnuto (období)", [("sum by (tenant, reason) (increase(aigw_rejected_total[$__range]))", "{{tenant}} · {{reason}}")], "short", 18, 0, 6, 5, "stat", decimals=0)
# řádek 2: provoz
p("Tokeny/s podle tenanta", [(f"sum by (tenant) (rate(aigw_tokens_total[{R}]))", "{{tenant}}")], "short", 0, 5, 12, 8, stack=True)
p("Požadavky/s podle stavu", [(f'sum by (tenant, status) (rate(aigw_requests_total[{R}]))', "{{tenant}} · {{status}}"),
                              (f"sum by (tenant) (rate(aigw_rejected_total[{R}]))", "{{tenant}} · odmítnuto")], "reqps", 12, 5, 12, 8)
# řádek 3: kvalita služby
p("TTFT p95 (stream)", [(f"histogram_quantile(0.95, sum by (le, tenant) (rate(aigw_ttft_seconds_bucket[{R}])))", "{{tenant}}")], "s", 0, 13, 8, 8)
p("Doba požadavku p95", [(f"histogram_quantile(0.95, sum by (le, tenant) (rate(aigw_request_duration_seconds_bucket[{R}])))", "{{tenant}}")], "s", 8, 13, 8, 8)
p("Souběžné požadavky", [("sum by (tenant) (aigw_inflight)", "{{tenant}}")], "short", 16, 13, 8, 8)
# řádek 4: náklady a GPU
p("Kč za hodinu (tempo)", [(f"sum by (tenant) (rate(aigw_cost_czk_total[{R}])) * 3600", "{{tenant}}")], "currencyCZK", 0, 21, 12, 8, stack=True)
p("GPU využití a KV cache", [("DCGM_FI_DEV_GPU_UTIL", "GPU %"), ("vllm:kv_cache_usage_perc{model_name=\"qwen3-30b-a3b\"} * 100", "KV cache %")], "percent", 12, 21, 12, 8)

dash = {"uid": "ai-api", "title": "AI API – tenanti", "tags": ["tmobile"], "timezone": "browser", "refresh": "10s",
        "time": {"from": "now-30m", "to": "now"}, "schemaVersion": 39, "panels": panels}
out = pathlib.Path(__file__).parent / "grafana/dashboards/api.json"
out.write_text(json.dumps(dash, ensure_ascii=False, indent=1)); print(out, len(panels), "panelů")
