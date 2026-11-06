#!/usr/bin/env bash
# UC26: PLACEHOLDER batch entry point for the T-Mobile L40S VM. Not wired into queue/ yet.
#
#   uc26/run_l40s.sh /cold/uc26/db/bestbet.sqlite [--bet-book NAME] [--close-book NAME]
#
# The VM never ingests or scrapes anything: the finished SQLite file is copied there from the Mac.
# This script only runs compute over it and writes everything under $OUT (default /cold/uc26/<stamp>/):
#   all/backtest_<date>.md            walk-forward over all leagues
#   league_<L>/backtest_<date>.md     the same per league (an edge living in one league is noise)
#   summary.jsonl                     one JSON line per run
# TODO (phase 2+, only after logreg shows a signal): GBM model + bootstrap sweep across
#   leagues x seasons x bet_offset (T-2h / T-1h) x slippage, parallel over CPUs; GPU only if GBM needs it.
# Guard: refuses to run unless UC26_RUN=1 (it is a placeholder; do not run now).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DB=${1:?usage: run_l40s.sh <db.sqlite> [--bet-book NAME] [--close-book NAME]}
shift
BET_BOOK=softbook
CLOSE_BOOK=sharpbook
while [ $# -gt 0 ]; do
  case "$1" in
    --bet-book) BET_BOOK=$2; shift 2 ;;
    --close-book) CLOSE_BOOK=$2; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
[ "${UC26_RUN:-0}" = 1 ] || { echo "UC26 run_l40s.sh: placeholder, set UC26_RUN=1 to run" >&2; exit 3; }
PY=${PY:-/cold/venvs/uc26/bin/python}
OUT=${OUT:-/cold/uc26/$(date +%Y%m%d-%H%M%S)}
[ -f "$DB" ] || { echo "UC26: missing DB $DB" >&2; exit 2; }
mkdir -p "$OUT"
cd "$HERE"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
LEAGUES=$("$PY" -c "import sqlite3,sys; print(' '.join(r[0] for r in sqlite3.connect(sys.argv[1]).execute('SELECT DISTINCT league FROM games ORDER BY 1')))" "$DB")
run() {  # $1 = subdir, rest = extra args
  local sub=$1; shift
  nice -n 10 "$PY" -m backtest.run --db "$DB" --bet-book "$BET_BOOK" --close-book "$CLOSE_BOOK" \
    --out "$OUT/$sub" "$@" | "$PY" -c "import json,sys; d=json.load(sys.stdin); d['run']='$sub'; print(json.dumps(d))" \
    >> "$OUT/summary.jsonl"
}
run all 2>&1 | tee -a "$OUT/run.log"
for L in $LEAGUES; do
  run "league_$L" --league "$L" 2>&1 | tee -a "$OUT/run.log" || echo "league $L failed (e.g. < 3 seasons)" | tee -a "$OUT/run.log"
done
echo "UC26 done -> $OUT"
