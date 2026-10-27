#!/usr/bin/env bash
# UC11 krok 3 — model vs model: stejné specifikace (stejní kupující, stejné parametry) na jiném LLM.
# Spouštět na VM z /srv/work/w2/uc11 v tmux AŽ PO přepnutí GPU profilu Directorem.
#
#   code/model_vs_model.sh <served-model-name> [max_rep=5] [parallel]
#
#   qwen3-32b  (profil A, hybridní thinking → vypínáme přes chat_template_kwargs)
#   qwen3-8b   (zatím bez profilu; served name dle toho, jak ho Director pustí)
#   qwen3-30b-a3b se NEspouští znovu — jeho výsledky jsou podmnožina hlavního gridu (stejná run_id).
#
# Podmnožina = replikace 0..max_rep-1 všech 135 buněk (max_rep=5 → 675 běhů).
# Porovnání: code/compare_models.py runs/full_results.jsonl runs/mvm_<model>_results.jsonl …
set -euo pipefail
M=${1:?served model name}; R=${2:-5}; PAR=${3:-48}
cd "$(dirname "$0")/.."
SPEC=runs/mvm_specs_rep${R}.jsonl
[ -f "$SPEC" ] || python3 -c "
import json,sys
for l in open('runs/full_specs.jsonl'):
    if json.loads(l)['factors']['rep'] < $R: sys.stdout.write(l)
" > "$SPEC"
export LLM_DECISION_MODEL="$M"
case "$M" in
  qwen3-30b-a3b*) ;;                                     # Instruct-2507 nemá thinking
  *) export LLM_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}' ;;
esac
# hustý model (32B) má nižší propustnost a KV budget → víc souběžných sims nepomůže
exec code/run_grid.sh "mvm_${M}" "$SPEC" "runs/mvm_${M}_results.jsonl" "$PAR"
