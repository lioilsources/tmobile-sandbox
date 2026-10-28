#!/usr/bin/env bash
# UC24 — po doběhnutí CPU soudců: vyhodnocení soudců, učený gate s CLAP, oprava (krok 5). Jen CPU.
set -uo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES= HF_HOME=/srv/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
PY="nice -n 19 /cold/venvs/uc24/bin/python"; R=/cold/uc24/results; mkdir -p $R
until grep -q "HOTOVO (uc24 judges cpu)" /cold/uc24/judges_cpu.log 2>/dev/null; do sleep 60; done
$PY eval_judges.py --out $R/UC24_judges.json > /dev/null 2>$R/eval_judges.err
$PY learned.py --json $R/UC24_learned.json > /dev/null 2>$R/learned.err
# eval_repair běží zvlášť (uc24rep), souběžně se soudci
echo "HOTOVO (uc24 after_judges)" >> /cold/uc24/judges_cpu.log
