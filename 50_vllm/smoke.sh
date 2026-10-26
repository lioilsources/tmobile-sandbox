#!/usr/bin/env bash
# Smoke test aktuálního profilu.
set -u
P=$(cat /srv/work/.gpu_profile 2>/dev/null || echo ?)
echo "profil $P"
if curl -sf localhost:8000/v1/models >/dev/null; then
  M=$(curl -s localhost:8000/v1/models | jq -r '.data[0].id')
  if [ "$P" = C ]; then
    INV=/srv/work/smoke_invoice.png
    [ -f $INV ] || /srv/comfy/ComfyUI/venv/bin/python "$(dirname "$0")/make_invoice.py" $INV
    { printf 'data:image/png;base64,'; base64 -w0 $INV; } | jq -Rs --arg m "$M" '. as $u | {model:$m,max_tokens:400,messages:[{role:"user",content:[{type:"image_url",image_url:{url:$u}},
      {type:"text",text:"Vytěž z faktury JSON s poli: cislo_faktury, dodavatel_ico, odberatel_ico, datum_splatnosti, variabilni_symbol, zaklad, dph, celkem. Vrať jen JSON."}]}]}' > /tmp/smoke_body.json
    BODY=@/tmp/smoke_body.json
  else
    BODY=$(jq -n --arg m "$M" '{model:$m,max_tokens:200,messages:[{role:"user",content:"Shrň jednou větou česky, co je T-Mobile. /no_think"}]}')
  fi
  time curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d "$BODY" | jq -r '.choices[0].message.content, .usage'
fi
curl -sf localhost:8002/v1/models >/dev/null && curl -s localhost:8002/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-embed","input":["Dobrý den, volám kvůli faktuře."]}' | jq '.data[0].embedding|length'
if curl -sf localhost:8001/health >/dev/null; then
  [ -f /srv/work/smoke_cs.wav ] || { command -v espeak-ng >/dev/null && espeak-ng -v cs -w /srv/work/smoke_cs.wav "Dobrý den, chtěl bych změnit tarif a zrušit roaming."; }
  [ -f /srv/work/smoke_cs.wav ] && time curl -s localhost:8001/v1/audio/transcriptions -F file=@/srv/work/smoke_cs.wav \
    -F model=deepdml/faster-whisper-large-v3-turbo-ct2 -F language=cs | jq -r .text
fi
curl -sf localhost:8188/system_stats >/dev/null && curl -s localhost:8188/system_stats | jq '.devices[0]|{name,vram_total,vram_free}'
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
