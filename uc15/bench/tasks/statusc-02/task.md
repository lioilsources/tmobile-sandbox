# Host disks: report each filesystem once, report missing paths, skip tiny remote filesystems

In `internal/host`:
- `Local.Sample` must report each filesystem once: if two configured disk paths are on the same
  device (same `st_dev`), keep only the first path that named it.
- Add `func (l *Local) MissingDisks() []string` returning configured paths that cannot be stat'd
  (the caller in `cmd/status-collector` logs them once at startup).
- Remote (node_exporter) filesystems smaller than 1 GiB (e.g. /boot/efi) are skipped.
- Optional: allow the disk list to come from a `HOST_DISKS` env var in the systemd unit.

Verify: `go test ./internal/host` (TestLocalReportsEachFilesystemOnce, TestMissingDisksAreReported).
