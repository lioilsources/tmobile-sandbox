#!/usr/bin/env bash
# UC7: SDXL LoRA „RealEyes“ na L40S — trénink (kohya sd-scripts v0.8.7, obraz finetune-kohya z FineTuneGallery,
# stejný baseline config jako jeho backend: aspekt face, dim 32/alpha 16, 10 epoch, batch 2, AdamW8bit) s GPU logem,
# pak vyhodnocení: mřížka epocha × síla LoRA, FID/KID base vs LoRA proti odloženým reálným výřezům.
# Dataset: crop_eyes.py → select_caption.py (/srv/work/uc7/dataset/{train,ref}).
set -euo pipefail
cd "$(dirname "$0")"; W=/srv/work/uc7; R=$W/run1; D=$W/dataset; N_GEN=${N_GEN:-300}
CK=/srv/comfy/ComfyUI/models/checkpoints; IMG=finetune-kohya:latest; PY=/srv/work/uc1/ft-venv/bin/python
[ -f $D/train/$(ls $D/train | head -1) ] || { echo "chybí dataset $D/train"; exit 1; }
docker image inspect $IMG >/dev/null || { echo "chybí obraz $IMG (docker build FineTuneGallery/train)"; exit 1; }
[ "$(cat /srv/work/.gpu_profile 2>/dev/null)" = off ] || ../50_vllm/switch.sh off >/dev/null
mkdir -p $R/output $R/gen
N=$(ls $D/train/*.jpg | wc -l)

cat > $R/dataset.toml <<EOF
[general]
enable_bucket = true
min_bucket_reso = 640
max_bucket_reso = 1536
bucket_reso_steps = 64
caption_extension = ".txt"

[[datasets]]
resolution = 1024
batch_size = 2
keep_tokens = 1

  [[datasets.subsets]]
  image_dir = "/dataset/img"
  num_repeats = 1
  shuffle_caption = true
EOF
cat > $R/train_config.toml <<EOF
pretrained_model_name_or_path = "/ckpts/sd_xl_base_1.0.safetensors"
output_dir = "/dataset/output"
output_name = "realeyes_sdxl_v1"
save_model_as = "safetensors"
network_module = "networks.lora"
network_dim = 32
network_alpha = 16
learning_rate = 1e-4
unet_lr = 1e-4
text_encoder_lr = 5e-5
lr_scheduler = "cosine_with_restarts"
lr_warmup_steps = 100
optimizer_type = "AdamW8bit"
max_train_epochs = 10
save_every_n_epochs = 2
mixed_precision = "bf16"
save_precision = "bf16"
gradient_checkpointing = true
cache_latents = true
min_snr_gamma = 5
seed = 42
sdpa = true
EOF

kohya() {  # kohya() <skript> <args…> — kontejner s datasetem, checkpointy a cache
  docker run --rm --gpus all --shm-size 8g -v $R:/dataset -v $D/train:/dataset/img:ro -v $D/ref:/ref:ro -v $CK:/ckpts:ro \
    -v finetune-kohya-cache:/cache $IMG "$@"
}

# ── trénink
if [ ! -f $R/output/realeyes_sdxl_v1.safetensors ]; then
  read -r PID CSV < <(../bench/gpu_log.sh uc7_train)
  T0=$(date +%s)
  kohya sdxl_train_network.py --config_file /dataset/train_config.toml --dataset_config /dataset/dataset.toml > $R/train.log 2>&1 \
    || { kill $PID; echo "trénink selhal"; tail -30 $R/train.log; exit 1; }
  T1=$(date +%s); kill $PID
  STEPS=$(grep -oP "total optimization steps.*?: \K[0-9]+" $R/train.log | head -1)
  echo "{\"obrazku\": $N, \"epoch\": 10, \"kroku\": $STEPS, \"trenink_s\": $((T1 - T0)), \"s_na_epochu\": $(( (T1 - T0) / 10 ))}" > $R/train.json
  ../bench/summarize.py "$CSV" --label UC7_lora_train --extra "{\"kroku\": $STEPS}" --merge $R/train.json
fi

# ── vyhodnocení: popisky z trénink. sady (bez trigger slova pro base nemá smysl rozlišovat — base slovo nezná)
$PY - "$D/train" "$R" "$N_GEN" <<'EOF'
import pathlib, random, sys
src, out, n = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
caps = sorted(p.read_text().strip() for p in src.glob("*.txt")); random.Random(1).shuffle(caps)
neg = " --n blurry, cartoon, drawing, deformed iris, asymmetric eyes"
(out / "prompts_fid.txt").write_text("".join(f"{c}{neg} --d {1000 + i}\n" for i, c in enumerate(caps[:n])))
grid = ["rlyz, close-up photo of a pair of human eyes, green eyes, eyes without makeup, the eyes of a young adult, eyes in soft natural daylight",
        "rlyz, close-up photo of a pair of human eyes, dark brown eyes, eyes with eyeliner and mascara, the eyes of a middle-aged person, eyes in studio lighting",
        "rlyz, close-up photo of a pair of human eyes, blue eyes, eyes without makeup, the wrinkled eyes of an elderly person, eyes in harsh bright sunlight",
        "rlyz, close-up photo of a pair of human eyes, hazel eyes, eyes with subtle mascara, the eyes of a young adult, eyes in dim warm indoor light"]
(out / "prompts_grid.txt").write_text("".join(f"{p}{neg} --am {m} --d 7\n" for p in grid for m in (0.0, 0.6, 1.0)))
EOF
GEN="sdxl_gen_img.py --ckpt /ckpts/sd_xl_base_1.0.safetensors --W 1408 --H 704 --steps 28 --scale 6 --sampler euler_a --bf16 --sdpa --batch_size 4 --sequential_file_name"
read -r PID CSV < <(../bench/gpu_log.sh uc7_gen)
T0=$(date +%s)
[ -d $R/gen/base ] || kohya $GEN --outdir /dataset/gen/base --from_file /dataset/prompts_fid.txt > $R/gen_base.log 2>&1
[ -d $R/gen/lora ] || kohya $GEN --outdir /dataset/gen/lora --from_file /dataset/prompts_fid.txt --network_module networks.lora \
  --network_weights /dataset/output/realeyes_sdxl_v1.safetensors --network_mul 1.0 > $R/gen_lora.log 2>&1
T1=$(date +%s)
for e in 000002 000004 000006 000008; do
  [ -d $R/gen/grid_e$e ] || kohya $GEN --outdir /dataset/gen/grid_e$e --from_file /dataset/prompts_grid.txt --network_module networks.lora \
    --network_weights /dataset/output/realeyes_sdxl_v1-$e.safetensors --network_mul 1.0 > $R/gen_grid.log 2>&1
done
[ -d $R/gen/grid_e000010 ] || kohya $GEN --outdir /dataset/gen/grid_e000010 --from_file /dataset/prompts_grid.txt --network_module networks.lora \
  --network_weights /dataset/output/realeyes_sdxl_v1.safetensors --network_mul 1.0 >> $R/gen_grid.log 2>&1
kill $PID
echo "{\"generovano\": $((2 * N_GEN)), \"generace_s\": $((T1 - T0))}" > $R/gen.json
../bench/summarize.py "$CSV" --label UC7_lora_gen --extra "{\"obrazku\": $((2 * N_GEN))}" --merge $R/gen.json
$PY eval_lora.py --run $R --ref $D/ref | tee $R/eval.log | tail -3
echo UC7_DONE
