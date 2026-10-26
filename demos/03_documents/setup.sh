#!/usr/bin/env bash
# Prostředí pro demo 3: fonty (víc vzhledů dokumentů) + venv.
set -euo pipefail
sudo apt-get install -y -q fonts-liberation fonts-noto-core fonts-dejavu-extra >/dev/null
D=/srv/work/demo3; mkdir -p $D
[ -d $D/venv ] || python3 -m venv $D/venv
$D/venv/bin/pip install -q -U pillow faker pymupdf numpy httpx
echo ok
