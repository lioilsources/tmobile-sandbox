#!/usr/bin/env python3
"""Generuje přehledový Grafana dashboard „L40S Sandbox“ (GPU, RAM, CPU, disk, vLLM) → grafana/dashboards/sandbox.json."""
import json, pathlib
DS = {"type": "prometheus", "uid": "prom"}
panels, y = [], 0

def p(title, exprs, unit, w=8, h=7, kind="timeseries", mx=None, legend=None, thresholds=None):
    global y
    x = sum(q["gridPos"]["w"] for q in panels if q["gridPos"]["y"] == y)
    if x + w > 24: y += h; x = 0
    fc = {"unit": unit, "min": 0, **({"max": mx} if mx is not None else {})}
    if kind == "timeseries": fc["custom"] = {"fillOpacity": 15, "lineWidth": 2, "showPoints": "never"}
    if thresholds: fc["thresholds"] = {"mode": "absolute", "steps": [{"color": c, "value": v} for v, c in thresholds]}
    panels.append({"type": kind, "title": title, "datasource": DS, "gridPos": {"x": x, "y": y, "w": w, "h": h},
                   "fieldConfig": {"defaults": fc, "overrides": []},
                   "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "reduceOptions": {"calcs": ["lastNotNull"]}},
                   "targets": [{"expr": e, "legendFormat": l, "refId": chr(65 + i), "datasource": DS}
                               for i, (e, l) in enumerate(exprs)]})

RAMU = "(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100"
TRAFFIC = [(None, "green"), (70, "yellow"), (90, "red")]
# řádek 1: stav teď
p("GPU využití", [("DCGM_FI_DEV_GPU_UTIL", "")], "percent", 4, 5, "gauge", 100, thresholds=TRAFFIC)
p("VRAM", [("DCGM_FI_DEV_FB_USED / (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE) * 100", "")], "percent", 4, 5, "gauge", 100, thresholds=TRAFFIC)
p("RAM", [(RAMU, "")], "percent", 4, 5, "gauge", 100, thresholds=TRAFFIC)
p("Swap použitý", [("node_memory_SwapTotal_bytes - node_memory_SwapFree_bytes", "")], "bytes", 4, 5, "stat")
p("GPU příkon", [("DCGM_FI_DEV_POWER_USAGE", "")], "watt", 4, 5, "stat")
p("GPU teplota", [("DCGM_FI_DEV_GPU_TEMP", "")], "celsius", 4, 5, "stat")
y += 5
# řádek 2: GPU v čase
p("GPU využití", [("DCGM_FI_DEV_GPU_UTIL", "compute"), ("DCGM_FI_DEV_MEM_COPY_UTIL", "paměťová sběrnice")], "percent", 12, 8, mx=100)
p("VRAM", [("DCGM_FI_DEV_FB_USED * 1048576", "použito"), ("(DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE) * 1048576", "celkem")], "bytes", 12, 8)
y += 8
# řádek 3: stroj
p("RAM + swap", [("node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes", "RAM použito"), ("node_memory_MemTotal_bytes", "RAM celkem"),
                 ("node_memory_SwapTotal_bytes - node_memory_SwapFree_bytes", "swap")], "bytes", 8, 8)
p("CPU", [('100 * (1 - avg(rate(node_cpu_seconds_total{mode="idle"}[1m])))', "CPU %"), ("node_load1 / 8 * 100", "load1 / 8 vCPU")], "percent", 8, 8)
p("Disk volno", [('node_filesystem_avail_bytes{mountpoint="/"}', "/ (systém)"), ('node_filesystem_avail_bytes{mountpoint="/data"}', "/data")], "bytes", 8, 8)
y += 8
# řádek 4: vLLM
p("vLLM tokeny/s", [('sum by (model_name) (rate(vllm:generation_tokens_total[1m]))', "výstup {{model_name}}"),
                    ('sum by (model_name) (rate(vllm:prompt_tokens_total[1m]))', "vstup {{model_name}}")], "short", 8, 8)
p("vLLM požadavky", [("sum by (model_name) (vllm:num_requests_running)", "běží {{model_name}}"),
                     ("sum by (model_name) (vllm:num_requests_waiting)", "čeká {{model_name}}")], "short", 8, 8)
p("vLLM latence p50 / p95", [('histogram_quantile(0.5, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))', "TTFT p50"),
                             ('histogram_quantile(0.95, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))', "TTFT p95"),
                             ('histogram_quantile(0.95, sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket[5m])))', "celý požadavek p95")], "s", 8, 8)
y += 8
p("vLLM KV cache", [("vllm:kv_cache_usage_perc * 100", "{{model_name}}")], "percent", 24, 6, mx=100)

dash = {"uid": "sandbox", "title": "L40S Sandbox", "tags": ["tmobile"], "timezone": "browser", "refresh": "10s",
        "time": {"from": "now-1h", "to": "now"}, "schemaVersion": 39, "panels": panels}
out = pathlib.Path(__file__).parent / "grafana/dashboards/sandbox.json"
out.write_text(json.dumps(dash, ensure_ascii=False, indent=1)); print(out, len(panels), "panelů")
