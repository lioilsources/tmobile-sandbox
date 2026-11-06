#!/usr/bin/env bash
# UC4 noční záznam trailerových klipů (spouští fronta po tréninku Kirian iterace 2).
# ~20 kampaní od sektoru 0 s agenty it2 (záloha ppo3), pak stříhání klipů. Vše do /cold/uc4/trailer/<datum>/.
# Celé skončí samo nejpozději za 3 h (timeout). GPU je v noci volná → NVENC pro mezizáznam i finální kódování.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/srv/work/uc1/ft-venv/bin/python
BUNDLE=/srv/work/uc4/kiran-rec/tyrian_mobile/build/linux/x64/release/bundle
DATE=$(date +%F)
OUT=/cold/uc4/trailer/$DATE
RUNS=${RUNS:-20}
mkdir -p "$OUT" /cold/uc4/trailer/tmp
LOG="$OUT/queue_rec.log"
exec >>"$LOG" 2>&1
echo "== queue_rec start $(date -Is)"

exec 9>/cold/uc4/trailer/tmp/queue_rec.lock
flock -n 9 || { echo "už běží jiný queue_rec, končím"; exit 0; }

PILOT=/srv/work/uc4/runs/it2/campaign/agent.pt
SHOP=/srv/work/uc4/runs/it2/campaign/shop.pt
if [[ ! -f $PILOT ]]; then
  echo "it2 pilot chybí, záloha ppo3"; PILOT=/srv/work/uc4/runs/ppo3/agent.pt; SHOP=/srv/work/uc4/runs/ppo3/shop.pt
fi
[[ -f $SHOP ]] || SHOP=greedy
[[ -x $BUNDLE/kirian ]] || { echo "chybí build $BUNDLE/kirian"; exit 1; }
echo "pilot=$PILOT shop=$SHOP runs=$RUNS out=$OUT"

# Záznam: nejvýš 2 h 35 min celkem, jeden běh nejvýš 15 min; vzorkovaná politika (jiný seed na běh), aby se běhy lišily.
# Většina běhů v nativním 16:9 (desktopová hra), PORTRAIT_RUNS v poli 600×300 přesně jako při tréninku
# (úzký pruh uprostřed, pozadí rozmazané) — agent tam hraje přesně v tréninkové geometrii.
PORTRAIT_RUNS=${PORTRAIT_RUNS:-0}  # hra je na šířku, 16:9 nativně stačí
NATIVE_RUNS=$(( RUNS - PORTRAIT_RUNS ))
END=$(( $(date +%s) + 9300 ))
timeout -k 60 7800 "$PY" "$HERE/record.py" --bundle "$BUNDLE" --pilot "$PILOT" --shop "$SHOP" --policy sample \
  --runs "$NATIVE_RUNS" --sector 0 --layout native --capture nvenc --encoder nvenc --run-timeout 900 \
  --out "$OUT" --name camp --seed 100
echo "record native rc=$? $(date -Is)"
LEFT=$(( END - $(date +%s) ))
if (( LEFT > 300 && PORTRAIT_RUNS > 0 )); then
  timeout -k 60 "$LEFT" "$PY" "$HERE/record.py" --bundle "$BUNDLE" --pilot "$PILOT" --shop "$SHOP" --policy sample \
    --runs "$PORTRAIT_RUNS" --sector 0 --layout portrait --capture nvenc --encoder nvenc --run-timeout 900 \
    --out "$OUT" --name campP --seed 500
  echo "record portrait rc=$? $(date -Is)"
fi

# Stříhání: nejvýš 20 min.
timeout -k 30 1200 "$PY" "$HERE/cut_clips.py" --out "$OUT" --per-run 8 --min-score 4 --encoder nvenc
echo "cut rc=$? $(date -Is)"

# Po sobě nic nenechat běžet (Xvfb/pulse/hra patří jen tomuto skriptu).
pkill -u "$(id -u)" -x kirian 2>/dev/null
pkill -u "$(id -u)" -f "Xvfb :77 " 2>/dev/null
pkill -u "$(id -u)" -f "socket=/cold/uc4/trailer/tmp/pulse_rt/pulse.sock" 2>/dev/null  # vlastní pulseaudio z record.py
rm -rf /cold/uc4/trailer/tmp/home_* 2>/dev/null
ls "$OUT/full" "$OUT/clips" 2>/dev/null | head -100
echo "== queue_rec end $(date -Is)"
