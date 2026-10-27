#!/usr/bin/env bash
# Stažení modelů. Spouštět v tmux:  tmux new -s models './40_models.sh core'
#   core  = LLM pro vLLM + Whisper + ComfyUI (FLUX-schnell, Qwen-Image, Wan T2V, SDXL, ACE-Step)
#   extra = Wan2.2 I2V (+29 GB) — jen když je místo
#   uc1   = OCR enginy pro UC1 (~85 GB)
#   w2img = druhá vlna, SDXL rodiny + FLUX.1-dev FP8 pro UC18/19 (~52 GB; FLUX.1-dev má nekomerční licenci → jen interní měření)
#   w2    = druhá vlna: coder pro UC15, malé Qwen3 pro UC16/19, WD14 tagger (~43 GB)
# Disk VM je 300 GB (z toho ~50 GB docker image v /var/lib/containerd) (ne 1 TB z plánu) → LLM ve FP8, Comfy jen repackaged fp8 soubory, ne celá repa.
set -euo pipefail
TIER=${1:-core}
export HF_HOME=/srv/hf HF_XET_HIGH_PERFORMANCE=1
COMFY_DL=/srv/comfy/dl
command -v hf >/dev/null || pipx install 'huggingface_hub[hf_xet]'
export PATH="$HOME/.local/bin:$PATH"

need_gb() {  # need_gb <GB> — stop, pokud by na disku zbylo < 15 GB
  local free; free=$(df -BG --output=avail /srv | tail -1 | tr -dc 0-9)
  (( free - $1 >= 15 )) || { echo "STOP: volno ${free} GB, potřeba $1 GB + 15 rezerva"; exit 1; }
}

llm() {  # llm <repo> <GB>  → HF cache (vLLM / speaches čtou z /srv/hf)
  echo "==> $1"
  [ -d "$HF_HOME/hub/models--${1//\//--}/snapshots" ] || need_gb "$2"   # už stažené = jen ověří/doplní
  hf download "$1" --exclude 'original/*' >/dev/null
}

comfy() {  # comfy <repo> <cílová složka models/...> <soubor v repu> <GB>
  local dst=/srv/comfy/ComfyUI/models/$2
  echo "==> $1 :: $3"
  if [ ! -e "$COMFY_DL/$1/$3" ]; then need_gb "$4"; hf download "$1" "$3" --local-dir "$COMFY_DL/$1" >/dev/null; fi
  mkdir -p "$dst" && ln -sfn "$COMFY_DL/$1/$3" "$dst/$(basename "$3")"
}

if [ "$TIER" = core ]; then
  # --- vLLM (FP8 → vejdou se do 48 GB VRAM i na disk)
  llm Qwen/Qwen3-32B-FP8                    35
  llm Qwen/Qwen3-30B-A3B-Instruct-2507-FP8  32
  llm Qwen/Qwen3-VL-32B-Instruct-FP8        36
  llm Qwen/Qwen3-Embedding-0.6B              2
  llm deepdml/faster-whisper-large-v3-turbo-ct2 2
  # --- ComfyUI
  comfy Comfy-Org/flux1-schnell        checkpoints     flux1-schnell-fp8.safetensors 18
  comfy Comfy-Org/Qwen-Image_ComfyUI   diffusion_models split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors 21
  comfy Comfy-Org/Qwen-Image_ComfyUI   text_encoders   split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors 10
  comfy Comfy-Org/Qwen-Image_ComfyUI   vae             split_files/vae/qwen_image_vae.safetensors 1
  W=split_files
  comfy Comfy-Org/Wan_2.2_ComfyUI_Repackaged diffusion_models $W/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors 15
  comfy Comfy-Org/Wan_2.2_ComfyUI_Repackaged diffusion_models $W/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors 15
  comfy Comfy-Org/Wan_2.2_ComfyUI_Repackaged text_encoders   $W/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors 7
  comfy Comfy-Org/Wan_2.2_ComfyUI_Repackaged vae             $W/vae/wan_2.1_vae.safetensors 1
  comfy Comfy-Org/Wan_2.2_ComfyUI_Repackaged loras           $W/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors 2
  comfy Comfy-Org/Wan_2.2_ComfyUI_Repackaged loras           $W/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors 2
  comfy stabilityai/stable-diffusion-xl-base-1.0 checkpoints sd_xl_base_1.0.safetensors 7
  comfy madebyollin/sdxl-vae-fp16-fix  vae             sdxl_vae.safetensors 1
  comfy Comfy-Org/ACE-Step_ComfyUI_repackaged checkpoints all_in_one/ace_step_v1_3.5b.safetensors 8
elif [ "$TIER" = uc1 ]; then   # UC1 genealogie OCR — kandidátní enginy (Qwen3-VL-32B-FP8 už je v core)
  llm Qwen/Qwen3-VL-8B-Instruct              18
  llm Qwen/Qwen2.5-VL-7B-Instruct            17   # baseline z matrika-ocr/finetune
  llm Qwen/Qwen3-VL-30B-A3B-Instruct-FP8     33
  llm rednote-hilab/dots.ocr                  7
  llm allenai/olmOCR-2-7B-1025-FP8           11
elif [ "$TIER" = w2 ]; then   # Prompts/04 §0.1 — Devstral, GLM, SDXL checkpointy a audio modely až podle místa
  llm Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8  33
  llm Qwen/Qwen3-8B-FP8                      10
  llm SmilingWolf/wd-swinv2-tagger-v3         1
elif [ "$TIER" = w2img ]; then   # Prompts/04 §0.1 #23
  comfy LyliaEngine/Pony_Diffusion_V6_XL  checkpoints ponyDiffusionV6XL_v6StartWithThisOne.safetensors 7
  comfy OnomaAIResearch/Illustrious-XL-v1.0 checkpoints Illustrious-XL-v1.0.safetensors 7
  comfy Laxhar/noobai-XL-1.1              checkpoints NoobAI-XL-v1.1.safetensors 7
  comfy cagliostrolab/animagine-xl-4.0    checkpoints animagine-xl-4.0.safetensors 7
  comfy RunDiffusion/Juggernaut-XL-v9     checkpoints Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors 8
  comfy Comfy-Org/Flux-1-Dev              checkpoints flux1-dev-fp8.safetensors 18
elif [ "$TIER" = extra ]; then
  W=split_files
  comfy Comfy-Org/Wan_2.2_ComfyUI_Repackaged diffusion_models $W/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors 15
  comfy Comfy-Org/Wan_2.2_ComfyUI_Repackaged diffusion_models $W/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors 15
  comfy Comfy-Org/Wan_2.2_ComfyUI_Repackaged loras           $W/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors 2
  comfy Comfy-Org/Wan_2.2_ComfyUI_Repackaged loras           $W/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors 2
fi

"$HOME/.local/share/pipx/venvs/huggingface-hub/bin/python" "$(dirname "$0")/41_models_manifest.py" > /srv/work/models.lock.json
df -h /srv
echo "HOTOVO ($TIER)"
