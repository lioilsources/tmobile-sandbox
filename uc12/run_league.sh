#!/usr/bin/env bash
# UC12: Elo liga AZ checkpointů proti Dart SearchAi Easy/Medium/Hard — jen CPU, smí běžet vedle tréninku.
#
#   uc12/run_league.sh                                   # 6 checkpointů × 3 úrovně × 100 partií + kalibrace
#   uc12/run_league.sh --games 200 --sims 64 --max-ckpts 10
#   uc12/run_league.sh --elo-only                        # jen přepočítat Elo z $OUT/games.jsonl
#
# Opakované spuštění dohraje jen chybějící partie (výsledky v $OUT/games.jsonl), Elo → $OUT/elo.json.
# Proměnné: PY, CLI (/srv/work/w2/uc12/bin/engine_cli), CKPT_DIR (/cold/w2/uc12/ckpt), OUT (/cold/w2/uc12/league),
#           NICE (10), CPUS (např. "2-7" → taskset; výchozí bez omezení)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH=$(cd "$HERE/../bench" && pwd)
PY=${PY:-/cold/venvs/uc12/bin/python}
CLI=${CLI:-/srv/work/w2/uc12/bin/engine_cli}
CK=${CKPT_DIR:-/cold/w2/uc12/ckpt}
OUT=${OUT:-/cold/w2/uc12/league}
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
[ -x "$CLI" ] || { echo "UC12 league: chybí engine_cli $CLI (dart compile exe --target-os linux tools/engine_cli.dart)" >&2; exit 2; }
mkdir -p "$OUT"
"$BENCH/memguard.sh" "$HERE/league.py" 1500 &
MG=$!
trap 'kill $MG 2>/dev/null' EXIT
PRE=(nice -n "${NICE:-10}")
[ -n "${CPUS:-}" ] && PRE+=(taskset -c "$CPUS")
"${PRE[@]}" "$PY" "$HERE/league.py" --cli "$CLI" --ckpts "$CK/nets/net_*.msgpack" --out "$OUT" "$@" 2>&1 \
  | tee -a "$OUT/league_$(date +%Y%m%d-%H%M%S).log"
