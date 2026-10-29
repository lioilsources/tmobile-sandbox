#!/usr/bin/env bash
# OPTIONAL ImageReward for UC18/UC19 scoring — NOT run yet (needs the user's/Director's OK for the download).
# CPU + network + disk only, no GPU: can run any time.
#
#   bash ~/tmobile-sandbox/uc18/setup_imagereward.sh
#
# Downloads (~1.8 GB → /cold, nothing on /srv):
#   zai-org/ImageReward (Apache-2.0): ImageReward.pt 1.79 GB + med_config.json   → /cold/hf/imagereward
#   google-bert/bert-base-uncased tokenizer files (Apache-2.0, < 1 MB)          → HF cache /cold/hf
#   venv /cold/venvs/uc18-ir: image-reward 1.5 + transformers 4.44 + torch cu128 (~3 GB, uv cache hardlinks)
# Why a separate venv: image-reward's BLIP (med.py) imports helpers removed from transformers ≥ 4.45.
set -euo pipefail
V=/cold/venvs/uc18-ir; D=/cold/hf/imagereward
export UV_CACHE_DIR=${UV_CACHE_DIR:-/cold/pipcache/uv} UV_LINK_MODE=hardlink HF_HOME=/cold/hf
mkdir -p "$D"
command -v uv >/dev/null || { echo "uv missing"; exit 1; }
[ -x $V/bin/python ] || uv venv -p 3.12 $V
uv pip install -p $V/bin/python --index-url https://download.pytorch.org/whl/cu128 torch torchvision
uv pip install -p $V/bin/python "image-reward==1.5" "transformers==4.44.2" "timm==0.6.13" "fairscale==0.4.13" pillow numpy requests
$V/bin/python - <<'EOF'
from huggingface_hub import hf_hub_download
for f in ("ImageReward.pt", "med_config.json"):
    print(hf_hub_download("zai-org/ImageReward", f, local_dir="/cold/hf/imagereward"))
from transformers import BertTokenizer
BertTokenizer.from_pretrained("bert-base-uncased")
EOF
# CPU self-test on one synthetic image
$V/bin/python - <<'EOF'
import ImageReward as RM, PIL.Image, tempfile, os
p = os.path.join(tempfile.mkdtemp(), "x.png"); PIL.Image.new("RGB", (512, 512), (200, 80, 40)).save(p)
m = RM.load("/cold/hf/imagereward/ImageReward.pt", device="cpu", med_config="/cold/hf/imagereward/med_config.json")
print("ImageReward CPU self-test score:", m.score("an orange square", p))
EOF
echo "ImageReward ready: score.sh picks it up automatically (IR_PY=$V/bin/python)"
