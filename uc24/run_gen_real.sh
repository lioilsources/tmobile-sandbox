#!/usr/bin/env bash
# UC24 — skutečné generované klipy (queue-ready; spouští Director v GPU okně). Fáze:
#   bulk    hromadné generování katalog × modifikátory → sfxgate (bez LLM)   → slovníková analýza
#   loop    feedback smyčka nad katalogem Storytelleru (generátor + LLM na :8000 SOUČASNĚ)
#   judges  CPU/GPU soudci nad vygenerovanými klipy + blind list pro Olina
# Backend = UC24_GEN:
#   comfy-ace  ComfyUI ACE-Step (profil D, :8188; ~27 GB VRAM peak, ~5 s / 60s klip na L40S podle UC5) — hudba/smyčky
#   comfy-sao  ComfyUI Stable Audio Open 1.0 — nejdřív stáhnout gated checkpoint + t5_base (TODO uživatel: licence)
#   aistack    AiStack services/audio (MOSS-SoundEffect v2.0) na UC24_AUDIO_URL — image aistack/audio-sfx na VM není
# Příklady:
#   UC24_GEN=comfy-ace KIND=music N=2000 bash run_gen_real.sh bulk judges
#   UC24_GEN=comfy-ace KIND=music bash run_gen_real.sh loop      # + vLLM qwen3-8b (≤0.35 GPU util) vedle ComfyUI
set -uo pipefail
cd "$(dirname "$0")"
: "${UC24_GEN:?nastav UC24_GEN}"; KIND=${KIND:-sfx}; N=${N:-5000}
export HF_HOME=/srv/hf HF_HUB_OFFLINE=1 UC24_GEN
PY=/cold/venvs/uc24/bin/python; W=/cold/uc24/real_${UC24_GEN}_${KIND}; mkdir -p $W /srv/work/bench/results
TASKS=storyteller; [ "$KIND" = music ] && TASKS=storyteller-music
for phase in "$@"; do
  case $phase in
    bulk)
      read -r PID CSV < <(../bench/gpu_log.sh uc24_bulk_${UC24_GEN})
      $PY gen_bulk.py --n "$N" --out $W/bulk.jsonl --save $W/wav_bulk
      kill $PID
      ../bench/summarize.py "$CSV" --label UC24_bulk_${UC24_GEN}_${KIND} --extra "{\"klipů\": $(wc -l < $W/bulk.jsonl)}" > /dev/null
      $PY toxicity.py --bulk $W/bulk.jsonl --out /srv/work/bench/results/UC24_toxicity_${UC24_GEN}.json ;;
    loop)
      curl -sf 127.0.0.1:8000/v1/models > /dev/null || { echo "loop potřebuje LLM na :8000"; exit 1; }
      read -r PID CSV < <(../bench/gpu_log.sh uc24_loop_${UC24_GEN})
      $PY feedback.py --tasks $TASKS --save $W/wav_loop --out $W/loop.jsonl --summary $W/loop_summary.json
      kill $PID
      ../bench/summarize.py "$CSV" --label UC24_loop_${UC24_GEN}_${KIND} --merge $W/loop_summary.json \
        --extra "{\"generací\": $(python3 -c "import json;print(json.load(open('$W/loop_summary.json'))['generations_total'])")}" > /dev/null ;;
    judges)
      python3 - "$W" "$KIND" <<'PY'
import json, os, sys
w, kind = sys.argv[1], sys.argv[2]
with open(f"{w}/manifest.jsonl", "w") as f:
    for src in ("bulk.jsonl",):
        p = os.path.join(w, src)
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            if "verdict" in r:
                f.write(json.dumps({"id": f"b{r['idx']:05d}", "path": f"{w}/wav_bulk/{r['idx']:05d}.wav", "prompt": r["prompt"],
                                    "kind": kind, "split": "real", "defect": "unknown", "verdict": r["verdict"]}) + "\n")
PY
      for j in panns clap-general aesthetics; do $PY judges.py --judge $j --manifest $W/manifest.jsonl --out $W/judges --device cuda --threads 4; done
      $PY rating_sheet.py --manifest $W/manifest.jsonl --out $W/rating --n 150 ;;
    *) echo "neznámá fáze $phase"; exit 1 ;;
  esac
done
echo "HOTOVO (uc24 gen real $UC24_GEN $KIND: $*)"
