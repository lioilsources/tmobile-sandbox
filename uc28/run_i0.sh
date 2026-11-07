#!/usr/bin/env bash
# UC28 I0: style test, 5 icons x 4 seeds -> drafts + BiRefNet cutouts + contact sheets on /cold.
set -euo pipefail
D=/cold/biodefense/icons
IDS=${IDS:-makrofag,kok,biofilm,virus-chripka,protilatka}
PY=/srv/comfy/ComfyUI/venv/bin/python
cd "$(dirname "$0")"
python3 icons.py gen --manifest $D/manifest.json --ids "$IDS" --out $D/drafts
# BiRefNet loads remote code from HF (trust_remote_code): run only when CUT=1
[ "${CUT:-0}" = 1 ] && HF_HOME=${HF_HOME:-/srv/hf} $PY icons.py cut --out $D/drafts
$PY icons.py sheet --manifest $D/manifest.json --ids "$IDS" --out $D/drafts --sheets $D/sheets
