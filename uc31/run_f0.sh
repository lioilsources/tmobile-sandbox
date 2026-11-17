#!/bin/bash
# UC31 F0 for BioDefenseRogue art/fx: fx-granula (cutout), bg-menu, bg-arena (seam fix), 4 seeds each, then sheets.
set -euo pipefail
cd "$(dirname "$0")"
PY=/srv/comfy/ComfyUI/venv/bin/python; M=/cold/biodefense/fx/manifest.json; D=/cold/biodefense/fx
I=../uc28/icons.py
$PY $I gen --manifest $M --base 5000 --ids fx-granula --out $D/fx
$PY $I gen --manifest $M --base 5000 --ids bg-menu --out $D/bg
$PY $I gen --manifest $M --base 5000 --ids bg-arena --out $D/tile
$PY seam.py --manifest $M --id bg-arena --out $D/tile
HF_HOME=/srv/hf $PY $I cut --out $D/fx
$PY $I sheet --manifest $M --base 5000 --ids fx-granula --out $D/fx --sheets $D/sheets
$PY $I sheet --manifest $M --base 5000 --ids bg-menu --out $D/bg --sheets $D/sheets
$PY $I sheet --manifest $M --base 5000 --ids bg-arena --out $D/tile --sheets $D/sheets
