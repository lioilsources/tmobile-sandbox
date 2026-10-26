# tmobile-sandbox

Provisioning T‑Mobile GPUaaS sandboxu (L40S 48 GB, CloudSigma KVM, Ubuntu 24.04). 
Na VM leží v `~/tmobile-sandbox` (sync: `rsync -a ./ $TMS_HOST:tmobile-sandbox/ --exclude 70_monitoring/.env --exclude 56_api/.env --exclude 56_api/tenants.json`; `TMS_HOST` = ssh alias VM).

## Skutečný HW (≠ plán)
8 vCPU · **23 GB RAM** · vda 300 GB (OS) + **vdb 650 GB `/data`** (LABEL=data) · L40S passthrough, driver 595 předinstalovaný · veřejná IP na ens3 (může se změnit po zásahu v portálu) · bez virtio kanálu pro guest agenta.
`05_data_disk.sh`: `/srv` a `/var/lib/containerd` (Docker 29 containerd store ignoruje `data-root`) leží na `/data` a jsou bind-mountnuté zpět (fstab). /data: 280 GB obsazeno, 327 GB volno.

## Pořadí
```
./00_inventory.sh            # bez sudo
./05_data_disk.sh            # /srv + containerd na druhý disk /data
./10_base.sh                 # balíky, /srv, ufw (jen 22), linger
./20_driver.sh               # přeskočí, když nvidia-smi funguje
./30_docker.sh               # Docker + NVIDIA toolkit   → SNAPSHOT „base“
tmux new -s models './40_models.sh core && ./40_models.sh extra'   # extra = Wan2.2 I2V
./60_comfy/install.sh
./70_monitoring/up.sh        # DCGM, node, Prometheus, Grafana :3001, Open WebUI :3000
./50_vllm/switch.sh A|B|C|D|off && ./50_vllm/smoke.sh
```

## Profily (ověřeno 25. 10. 2026, vLLM 0.32.0)
| | Služby | VRAM | Start | Smoke / bench |
|---|---|---|---|---|
| A | Qwen3‑32B‑FP8, ctx 32k, fp8 KV (~43k tok) | 41.3 GB | 4.5 min | 1024→256 tok @4 req/s: 0.98 req/s, 252 tok/s out, TTFT 29 s (saturováno) |
| B | Qwen3‑30B‑A3B‑2507‑FP8 + Qwen3‑Embedding‑0.6B + Whisper | 42.8 GB | 5.8 min | @8 req/s: 4.87 req/s, 1247 tok/s out, TTFT 2.9 s, TPOT 50 ms |
| C | Qwen3‑VL‑32B‑FP8, ctx 16k | 41.9 GB | 5.1 min | faktura → JSON, 8/8 polí správně, 9.3 s/strana |
| D | ComfyUI + Whisper | 19–39 GB | 20 s | FLUX‑schnell 1024² 4 kroky: cold 39 s, warm 2.5 s · Wan2.2 T2V 14B + lightx2v 832×480×33 sn.: cold 90 s, warm 16 s (RAM peak 21.8/23 GB, swap 1.1 GB) |

Whisper (large‑v3‑turbo, cs): krátká věta ~5 s včetně načtení. Bench JSONy v `/srv/work/bench/`.

## Přístup
`ssh $TMS_HOST` (alias v `~/.ssh/config` s LocalForward 3000/3001/8000/8001/8188/9090). Heslo Grafany: `70_monitoring/.env` na VM.

## Demo 3 – dokumenty (`demos/03_documents/run.sh`)
50 syntetických faktur/smluv (1/3 jako sken), Qwen3‑VL‑32B‑FP8, json_schema výstup, prompt v2:
150 dpi × souběh 16 → 944 stran/h, 100 % polí · 110 dpi × souběh 32 → 1529 stran/h, 99.75 % polí.

## Demo 1 – call centrum (`demos/01_callcenter/run.sh`, profil B + `55_tts`)
Scénáře Qwen3‑32B → validace rozhodčím (`validate.py`, 50/150) → Piper TTS (CPU) → stereo 8 kHz μ‑law → Whisper po kanálech + VAD → Qwen3‑30B‑A3B (json_schema).
WER 7.9 % / 12.0 % (operátor / zákazník), štítky 88 % (kategorie 100 %, ověření identity 98 %), 25–27× reálný čas ≈ 1 100–1 200 hovorů/h.

## Demo 2 – AI API pro tenanty (`56_api/up.sh`, `demos/02_api/run.sh 120`, profil B)
Gateway = AiStack gateway + `tenants.go` (klíče, povolené modely, limit souběhu → 429, měření tokenů vč. streamu, kniha `/srv/work/demo2/usage.jsonl`, `/metrics`),
→ LiteLLM v1.104.2 (`chat`, `embed`, `whisper`, master klíč v `56_api/.env`) → vLLM / Whisper. Klíče tenantů: `56_api/tenants.json` (jen na VM).
Grafana: dashboard „AI API – tenanti“. Load test 3×2 min: chatbot TTFT p95 94 → 133 ms během náporu C, C držen na 16 souběžných, ~1 000 výstupních tok/s.

## Reporty
`reports/*.md` + `*.html` (kopie v `/srv/work/reports/` → záloha na NAS).

## Zálohy
NAS v domácí síti si stahuje `/srv/work` → `$DST/work` (`80_backup.sh`, na NAS jako `~/bin/tmobile-backup.sh`, VM a DST v `~/.config/tmobile-backup.env`); v neděli hardlink snapshot, drží se 8 týdnů. Klíč `~/.ssh/tmobile_backup` na NAS je na VM omezený na `rrsync -ro /srv/work/`. Cron na NAS: `30 3 * * * $HOME/bin/tmobile-backup.sh >> $HOME/tmobile-backup.log 2>&1`.

## Otevřené
- ComfyUI běží s `--disable-pinned-memory` (`~/.config/comfyui.env`): RAM při FLUX+Wan 22 → 5.4 GB, rychlost stejná (`60_comfy/memtest.sh`).
- AiStack x86 build / controller-manager — zatím `switch.sh`.
- `/etc/sudoers.d/90-<user>` (NOPASSWD) — smazat, až nebude potřeba.
