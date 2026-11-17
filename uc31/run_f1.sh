#!/bin/bash
# UC31 F0 rerun (fx-granula, bg-menu with the 13. 11. prompt fixes; F0 drafts kept in f0/) + F1 (remaining 5 items).
set -euo pipefail
cd "$(dirname "$0")"
PY=/srv/comfy/ComfyUI/venv/bin/python; M=/cold/biodefense/fx/manifest.json; D=/cold/biodefense/fx
I=../uc28/icons.py
FX=fx-granula,fx-toxin,fx-atp,meta-vakcinace; BG=bg-menu,bg-vyhra,bg-prohra
$PY $I gen --manifest $M --base 5000 --ids $FX --out $D/fx
$PY $I gen --manifest $M --base 5000 --ids $BG --out $D/bg
HF_HOME=/srv/hf $PY $I cut --out $D/fx
$PY $I sheet --manifest $M --base 5000 --ids $FX --out $D/fx --sheets $D/sheets
$PY $I sheet --manifest $M --base 5000 --ids $BG --out $D/bg --sheets $D/sheets
