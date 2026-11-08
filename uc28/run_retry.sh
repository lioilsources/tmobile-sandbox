#!/usr/bin/env bash
# UC28 I1 retry (review_I1): per-icon retry_seeds from the manifest, aspergillus in 768x1344.
set -euo pipefail
D=/cold/biodefense/icons
IDS=nk-bunka,b-bunka,t-bunka,treg,pametova-bunka,cytotoxicka-t-bunka,aspergillus,opsonizace
PY=/srv/comfy/ComfyUI/venv/bin/python
cd "$(dirname "$0")"
python3 icons.py gen --manifest $D/manifest.json --ids "$IDS" --retry --out $D/drafts
HF_HOME=/srv/hf $PY icons.py cut --out $D/drafts | tee $D/cut_retry.log
$PY icons.py sheet --manifest $D/manifest.json --ids "$IDS" --retry --out $D/drafts --sheets $D/sheets
