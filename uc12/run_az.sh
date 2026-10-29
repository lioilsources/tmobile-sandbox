#!/usr/bin/env bash
# UC12: dlouhý AlphaZero běh na L40S. Potřebuje GPU SAMOTNOU (profil off) — jinak skončí hned.
#
#   uc12/run_az.sh <hodin> [další argumenty az.py…]
#   uc12/run_az.sh 24
#   uc12/run_az.sh 6 --sims 64 --batch 1024          # cokoli z `az.py --help`
#
# Co dělá:
#  1. preflight: volná VRAM ≥ MIN_FREE_MIB (výchozí 40 000), /cold ≥ 20 GB volno, venv s CUDA JAX
#  2. GPU log (bench/gpu_log.sh), pojistka RAM (bench/memguard.sh, práh 1 500 MB)
#  3. az.py ve vlastní process group; resume z $CKPT_DIR/state.msgpack, pokud existuje
#  4. po HEALTH_MIN minutách (5) kontrola: trenér žije, za relaci dohrál > 0 partií a GPU util
#     (průměr posledních 60 s) > MIN_GPU_UTIL (50 %) — jinak HLASITĚ skončí (stderr, $CKPT_DIR/HEALTH_FAIL,
#     exit 3) a trenéra ukončí SIGTERM (ten ještě uloží checkpoint)
#  5. watchdog: žádný nový řádek metrik STALL_MIN minut (20) → stejně hlasitý konec
#  6. na konci souhrn GPU (bench/summarize.py) do $LOG_DIR
# SIGTERM/SIGINT na tento skript → předá trenérovi (dokončí iteraci, uloží checkpoint, exit 0).
#
# Proměnné: PY, CKPT_DIR (/cold/w2/uc12/ckpt), LOG_DIR (/srv/work/w2/uc12/logs), MEM_FRACTION (auto = min(0.9, (volná−1 GB)/celková)),
#           HEALTH_MIN (5), MIN_GPU_UTIL (50), MIN_FREE_MIB (40000), STALL_MIN (20)
#           CPU_SMOKE=1 jen pro test skriptu: JAX na CPU, bez kontroly VRAM a GPU util (ne pro frontu!)
set -uo pipefail
H=${1:?použití: run_az.sh <hodin> [argumenty az.py]}; shift
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH=$(cd "$HERE/../bench" && pwd)
PY=${PY:-/cold/venvs/uc12/bin/python}
CK=${CKPT_DIR:-/cold/w2/uc12/ckpt}
LOGD=${LOG_DIR:-/srv/work/w2/uc12/logs}
MEMFRAC=${MEM_FRACTION:-}
HEALTH_MIN=${HEALTH_MIN:-5}
MIN_GPU=${MIN_GPU_UTIL:-50}
MIN_FREE=${MIN_FREE_MIB:-40000}
STALL_MIN=${STALL_MIN:-20}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
mkdir -p "$CK" "$LOGD"
TS=$(date +%Y%m%d-%H%M%S)
LOG=$LOGD/az_$TS.log
METRICS=$CK/metrics.jsonl

die() { echo "UC12 run_az ABORT: $1" >&2; echo "$(date -Is) $1" >> "$CK/HEALTH_FAIL"; exit "${2:-2}"; }

# --- 1. preflight
if [ "${CPU_SMOKE:-0}" = 1 ]; then
  export JAX_PLATFORMS=cpu; MIN_FREE=0; MIN_GPU=-1
  echo "UC12 run_az: CPU_SMOKE=1 — test skriptu na CPU, GPU kontroly vypnuté" >&2
fi
read -r used total < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | tr -d ',' | head -1)
free=$((total - used))
[ "$free" -ge "$MIN_FREE" ] || die "GPU není volná: ${free} MiB volno < ${MIN_FREE} (obsazeno ${used} MiB). Nejdřív profil off — run_az.sh potřebuje GPU samotnou."
if [ -z "$MEMFRAC" ]; then  # GPU sama: 0.9, ale nikdy víc, než je volné (−1 GB rezerva)
  MEMFRAC=$(awk -v f="$free" -v t="$total" 'BEGIN {x = (f - 1024) / t; if (x > 0.9) x = 0.9; printf "%.2f", x}')
fi
echo "UC12 run_az: volná VRAM ${free} / ${total} MiB → XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC}" >&2
cold_free=$(df -BG --output=avail /cold | tail -1 | tr -dc 0-9)
[ "$cold_free" -ge 20 ] || die "/cold má jen ${cold_free} GB volno"
want=gpu; [ "${CPU_SMOKE:-0}" = 1 ] && want=cpu
XLA_PYTHON_CLIENT_PREALLOCATE=false "$PY" -c "import jax; assert jax.default_backend()=='$want', jax.devices(); print('JAX', jax.__version__, jax.devices())" \
  || die "JAX v $PY nevidí $want"
rm -f "$CK/HEALTH_FAIL"

# --- 2. hlídači
read -r GPID GCSV < <("$BENCH/gpu_log.sh" uc12_az)
"$BENCH/memguard.sh" "$HERE/az.py" 1500 &
MGPID=$!
start_lines=$( [ -f "$METRICS" ] && wc -l < "$METRICS" || echo 0)

# --- 3. trenér
echo "UC12 run_az: ${H} h, log $LOG, GPU csv $GCSV, checkpointy $CK" >&2
setsid "$PY" "$HERE/az.py" --hours "$H" --ckpt-dir "$CK" --metrics "$METRICS" --gpu-log "$GCSV" \
  --mem-fraction "$MEMFRAC" "$@" > "$LOG" 2>&1 &
TPID=$!

stop_trainer() {  # SIGTERM celé skupině → az.py uloží checkpoint
  kill -TERM -- "-$TPID" 2>/dev/null
  for _ in $(seq 120); do kill -0 "$TPID" 2>/dev/null || return 0; sleep 1; done
  echo "trenér po 120 s stále běží → SIGKILL" >&2; kill -KILL -- "-$TPID" 2>/dev/null
}
cleanup() {
  kill "$GPID" "$MGPID" 2>/dev/null
  "$BENCH/summarize.py" "$GCSV" --label "uc12_az_$TS" > "$LOGD/gpu_$TS.json" 2>/dev/null || true
}
trap 'echo "run_az: signál → zastavuji trenéra" >&2; stop_trainer; cleanup; exit 0' TERM INT

session_metrics() { tail -n +"$((start_lines + 1))" "$METRICS" 2>/dev/null; }
gpu_util_60s() { grep -v '^#' "$GCSV" | tail -60 | awk '{s+=$6; n++} END {if (n) printf "%d", s/n; else print 0}'; }

# --- 4. zdravotní kontrola po HEALTH_MIN minutách
t_end=$(( $(date +%s) + HEALTH_MIN * 60 ))
while [ "$(date +%s)" -lt "$t_end" ]; do
  kill -0 "$TPID" 2>/dev/null || { tail -30 "$LOG" >&2; cleanup; die "trenér skončil během prvních ${HEALTH_MIN} min (viz $LOG)" 3; }
  sleep 10
done
games=$(session_metrics | "$PY" -c "import sys,json; print(sum(json.loads(l).get('games',0) for l in sys.stdin if l.strip()))")
gpm=$(session_metrics | tail -1 | "$PY" -c "import sys,json; l=sys.stdin.read().strip(); print(json.loads(l)['games_per_min'] if l else 0)")
util=$(gpu_util_60s)
echo "UC12 health po ${HEALTH_MIN} min: partií za relaci ${games}, poslední games/min ${gpm}, GPU util 60 s ${util} %" >&2
if [ "${games:-0}" -le 0 ] || [ "${util:-0}" -le "$MIN_GPU" ]; then
  tail -5 "$LOG" >&2
  stop_trainer; cleanup
  die "HEALTH FAIL: partií ${games}, GPU util ${util} % (práh > ${MIN_GPU} %). Běh zastaven." 3
fi
echo "$(date -Is) health OK: games ${games}, games/min ${gpm}, GPU ${util} %" >> "$LOGD/health_$TS.log"

# --- 5. watchdog do konce
while kill -0 "$TPID" 2>/dev/null; do
  sleep 60
  age=$(( $(date +%s) - $(stat -c %Y "$METRICS" 2>/dev/null || echo 0) ))
  if [ "$age" -gt $((STALL_MIN * 60)) ] && kill -0 "$TPID" 2>/dev/null; then
    stop_trainer; cleanup
    die "STALL: žádné nové metriky ${age} s (> ${STALL_MIN} min). Běh zastaven." 3
  fi
done
wait "$TPID"; rc=$?
cleanup
tail -2 "$LOG" >&2
echo "UC12 run_az hotovo (exit $rc), metriky $METRICS, souhrn GPU $LOGD/gpu_$TS.json" >&2
exit $rc
