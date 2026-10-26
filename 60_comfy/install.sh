#!/usr/bin/env bash
# ComfyUI ve venv + custom nodes + systemd --user služba (127.0.0.1:8188). Idempotentní.
set -euo pipefail
ROOT=/srv/comfy/ComfyUI
[ -d $ROOT/.git ] || git clone https://github.com/comfyanonymous/ComfyUI $ROOT
git -C $ROOT pull --ff-only
cd $ROOT
[ -d venv ] || python3 -m venv venv
. venv/bin/activate
pip install -q -U pip wheel
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -q -r requirements.txt
pip install -q insightface onnxruntime-gpu   # InstantID, PuLID
pip install -q facenet-pytorch --no-deps     # PuLID (deps by stáhly torch<2.3)

NODES=(
  ltdrdata/ComfyUI-Manager kijai/ComfyUI-KJNodes kijai/ComfyUI-WanVideoWrapper
  Kosinkadink/ComfyUI-VideoHelperSuite rgthree/rgthree-comfy ltdrdata/ComfyUI-Impact-Pack
  city96/ComfyUI-GGUF Fannovel16/comfyui_controlnet_aux cubiq/ComfyUI_IPAdapter_plus
  # nody pro workflow ze Sparku (video-stack, Tsumiki, UI workflow)
  Kosinkadink/ComfyUI-AnimateDiff-Evolved Fannovel16/ComfyUI-Frame-Interpolation ltdrdata/ComfyUI-Impact-Subpack
  lquesada/ComfyUI-Inpaint-CropAndStitch Acly/comfyui-inpaint-nodes cubiq/ComfyUI_InstantID
  kijai/ComfyUI-MMAudio lldacing/ComfyUI_PuLID_Flux_ll 1038lab/ComfyUI-RMBG
)  # ComfyUI-Tsumiki je jen ve forku ComfyUI na Sparku  # ACE-Step je v ComfyUI nativně (TextEncodeAceStepAudio / EmptyAceStepLatentAudio)
for n in "${NODES[@]}"; do
  d=custom_nodes/${n#*/}
  [ -d "$d" ] || git clone -q --depth 1 "https://github.com/$n" "$d"
  git -C "$d" pull -q --ff-only || true
  if [ -f "$d/requirements.txt" ]; then pip install -q -r "$d/requirements.txt"; fi
  if [ "${n#*/}" = ComfyUI-Impact-Pack ] || [ "${n#*/}" = ComfyUI-Frame-Interpolation ]; then (cd "$d" && python install.py >/dev/null); fi
done

mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/comfyui.service <<UNIT
[Unit]
Description=ComfyUI
[Service]
WorkingDirectory=$ROOT
EnvironmentFile=-%h/.config/comfyui.env
ExecStart=$ROOT/venv/bin/python main.py --listen 127.0.0.1 --port 8188 --output-directory /srv/work/comfy-out \$COMFY_ARGS
Restart=on-failure
[Install]
WantedBy=default.target
UNIT
mkdir -p /srv/work/comfy-out
# RAM (23 GB) < VRAM (46 GB): pinned kopie vah v RAM jen překáží. Měřeno 60_comfy/memtest.sh 25. 10. 2026:
# výchozí RAM 22 GB / pinned off 5.4 GB, stejná rychlost; --highvram padá na VRAM OOM (FLUX + Wan naráz).
[ -f ~/.config/comfyui.env ] || echo 'COMFY_ARGS=--disable-pinned-memory' > ~/.config/comfyui.env
systemctl --user daemon-reload
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
echo "ComfyUI připraven; start přes ../50_vllm/switch.sh D"
