#!/usr/bin/env bash
# UC29 Stickers emoji: one batch (I0, I1, ...) -> drafts + BiRefNet cutouts (light grey background) + sheets on /cold.
#   run.sh <batch> [strip] [bycat]   jobs in /cold/stickers/emoji/jobs/<batch>.json (from Stickers make_jobs.py, unchanged)
set -euo pipefail
B=${1:?batch}; D=/cold/stickers/emoji; PY=/srv/comfy/ComfyUI/venv/bin/python
cd "$(dirname "$0")"
python3 emoji.py gen --jobs $D/jobs/$B.json --out $D/$B
# BiRefNet loads remote code from HF (trust_remote_code, cached revision approved 7. 11. for UC28)
HF_HOME=/srv/hf $PY ../uc28/icons.py cut --out $D/$B --fill-all --bg auto --clear-bg-holes ${HOLE_TOL:-14} | tee $D/$B/cut.log
$PY emoji.py sheet --jobs $D/jobs/$B.json --out $D/$B --sheets $D/sheets --batch $B ${2:+--strip} ${3:+--by-category}
