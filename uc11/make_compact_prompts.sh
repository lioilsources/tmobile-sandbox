#!/usr/bin/env bash
# UC11: z promptů repa udělá variantu pro grid, kde reasoning = 1 krátká věta.
#   make_compact_prompts.sh <repo/prompts> <out_dir>
# Proč: v JSON odpovědi jsou "action" a "amount" PŘED "reasoning" → rozhodnutí je vygenerované
# dřív, než model začne psát zdůvodnění; 2–3 věty reasoning jen prodlužují decode (~60 % tokenů).
# Kratší reasoning ovlivní nanejvýš "confidence" (je až za ním; engine ji používá jako tie-break).
set -euo pipefail
SRC=${1:?src}; OUT=${2:?out}
mkdir -p "$OUT"; cp "$SRC"/*.md "$OUT"/
sed -i.bak 's/"<2-3 věty>"/"<max 1 krátká věta>"/' "$OUT"/decision_*.md && rm -f "$OUT"/*.bak
for f in "$OUT"/decision_*.md; do
  grep -q "max 1 krátká věta" "$f" || { echo "nahrazení selhalo: $f" >&2; exit 1; }
done
echo "compact prompty → $OUT"
