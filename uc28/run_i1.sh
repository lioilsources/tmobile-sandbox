#!/usr/bin/env bash
# UC28 I1: remaining 16 icons x seeds k=0..3, plus redo of kok/protilatka with new subjects (k=4..7).
set -euo pipefail
D=/cold/biodefense/icons
REST=neutrofil,nk-bunka,dendriticka-bunka,zirna-bunka,b-bunka,t-bunka,cytotoxicka-t-bunka,treg,pametova-bunka,eozinofil,stafylokok,stafylokok-faze2,tycinka-mikrobiom,aspergillus,cervena-krvinka,mitochondrie
REDO=kok,protilatka
PY=/srv/comfy/ComfyUI/venv/bin/python
cd "$(dirname "$0")"
python3 icons.py gen --manifest $D/manifest.json --ids "$REDO" --k 4,5,6,7 --out $D/drafts
python3 icons.py gen --manifest $D/manifest.json --ids "$REST" --out $D/drafts
# BiRefNet loads remote code from HF (trust_remote_code, cached revision approved 7. 11.)
HF_HOME=/srv/hf $PY icons.py cut --out $D/drafts | tee $D/cut_i1.log
$PY icons.py sheet --manifest $D/manifest.json --ids "$REDO" --k 4,5,6,7 --out $D/drafts --sheets $D/sheets
$PY icons.py sheet --manifest $D/manifest.json --ids "$REST" --out $D/drafts --sheets $D/sheets
