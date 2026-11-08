#!/usr/bin/env bash
# Pitch (Prompts/08 P1.1–P1.3): re-run one product demo with GPU logging, copy its summary to /cold/pitch/<label>/.
#   bench/pitch_demo.sh demo3_110dpi -- env DPI=110 demos/03_documents/run.sh 32 55
#   bench/pitch_demo.sh demo1 -- demos/01_callcenter/run.sh 16
#   bench/pitch_demo.sh demo2 -- demos/02_api/run.sh 120
# Meant as a qctl `cmd` job with the demo's profile (qctl switches the profile before the job starts).
set -uo pipefail
L=${1:?label}; shift; [ "${1:-}" = -- ] && shift
cd "$(dirname "$0")/.."
OUT=/cold/pitch/$L; mkdir -p "$OUT"
read -r GPID GCSV < <(bench/gpu_log.sh "pitch_$L")
t0=$(date +%s)
"$@" 2>&1 | tee "$OUT/run.log"; rc=${PIPESTATUS[0]}
kill "$GPID" 2>/dev/null
bench/summarize.py "$GCSV" --label "pitch_$L" --extra "{\"wall_s\": $(( $(date +%s) - t0 )), \"exit\": $rc}" > "$OUT/gpu.json" || true
cp "$GCSV" "$OUT/"
run=$(grep -oE '(run|report): \S+' "$OUT/run.log" | tail -1 | awk '{print $2}')
[ -n "$run" ] && { d=${run%/report.md}; cp -r "$d" "$OUT/run" 2>/dev/null; cp "$d".log "$OUT/" 2>/dev/null; }
echo "pitch $L exit $rc → $OUT"
exit "$rc"
