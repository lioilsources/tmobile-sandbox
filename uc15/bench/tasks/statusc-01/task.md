# Host temps: chart the GPU temperature from node_exporter's textfile collector

In `internal/host/host.go`, remote hosts are sampled through node_exporter (`NodeExporter.fill`) and
temperatures are reduced per sensor kind by `hottest(...)` (`TempSensors`, `TempZone`, `TempNVMe`).

Add a GPU sensor kind:
- constant `TempGPU = "gpu"`, listed in `TempSensors` between zone and nvme;
- `hottest` takes GPU readings as well (zone, gpu, nvme); the local sampler has no GPU readings;
- the node_exporter sampler reads metric `node_gpu_temp_celsius` (written by a textfile collector
  file `.../gpu.prom`). If `node_textfile_mtime_seconds{file=".../gpu.prom"}` is more than 5 minutes
  older than `node_time_seconds`, the GPU reading is dropped (a dead timer shows as a gap).

Verify: `go test ./internal/host` (TestNodeExporterGPUTemp + existing tests).
