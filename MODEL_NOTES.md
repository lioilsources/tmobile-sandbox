# MODEL_NOTES — co se na L40S osvědčilo a na čem to padalo

Provozní poznámky ke každému modelu a nástroji: nastavení, které funguje, naměřená
propustnost a chyby, na které jsme už jednou narazili. **Číst před každým novým během**
(a dát přečíst sub-agentům). Doplňovat hned, když se něco zjistí — s číslem a datem.

Stroj: L40S 46 GB, 8 vCPU (sdílené, ~1 jádro steal), 23 GB RAM. vLLM image
`vllm/vllm-openai:latest` = build 5df63a72 (2026-10-18) — při změně tagu ověřit níže
uvedená nastavení znovu.

## Obecná pravidla (každý z nich už jednou stál hodiny GPU)

| Pravidlo | Proč (co se stalo) |
|---|---|
| Souběh LLM klientů dimenzovat podle `num_requests_waiting`, ne odhadem. Cíl: waiting ≈ 0, KV < 90 %, preempce 0. | UC11 na Qwen3-32B se 48 simulacemi: 67 běží, 173 čeká, 902 preempcí → rozhodnutí vypršela po 180 s, 62 % náhradních tahů, 2,5 h GPU k ničemu (2026-10-28). |
| Každý běh hlídat podílem fallbacků / chyb **během** prvních minut, ne až na konci. | Tamtéž; a UC25 VLM: 3 037 / 3 037 requestů HTTP 400, běh „skončil úspěšně“. |
| Klient vždy loguje **tělo** chybové odpovědi a první chybu hned na stderr. | UC25: hláška jen `400 Bad Request`, příčina zjištěna až offline. |
| Timeout dávky dimenzovat z naměřené rychlosti × 1,5, ne z odhadu agenta. Fáze, které následují (judges, souhrny), spouštět i po timeoutu nad tím, co vzniklo. | UC24 ACE-Step: odhad ~1 200 klipů/h, skutečnost 575/h → timeout 3 h usekl i navazující fázi soudců. UC25 kalibrace: 542 / 882 za 2 h. |
| Na 8 vCPU VM omezit vlákna: `OMP/MKL/OPENBLAS_NUM_THREADS=4`, torch `set_num_threads(4)`. | UC25 trénink hlavy: 8 vláken = 23 s / 10 epoch, 4 vlákna = 4,7 s; v noci se zasekl na 1 h timeout. |
| Po spuštění GPU úlohy ověřit, že GPU opravdu jede (`nvidia-smi` util > 0 do 2 min). | onnxruntime potichu spadl na CPU, GPU hodinu stála (viz WD14). |
| Testy/benchmarky spouštět v process group a při timeoutu zabít celou skupinu (vč. vnuků). | UC15: osiřelý `dart test` z úlohy mutants-02 běžel 14 h na ~2 jádrech, ignoroval SIGTERM. |
| `pkill -f <vzor>` přes `ssh` nepoužívat — vzor je i v příkazové řádce vlastního ssh a zabije ho (exit 255). Brát PID z `pgrep` / tmux pane. | Stalo se 2× (JODA kopie, UC11 stop). |
| Bežící bash skript neupravovat na místě — bash čte soubor průběžně. Frontu zastavit a založit znovu, nebo nové kroky dát do dalšího skriptu, který čeká na marker. | Fronty `queue/night*.sh`. |
| Prometheus scrape cíle jsou jména kontejnerů — nový kontejner (`vllm-coder`, `vllm-vl8b`) přidat do `70_monitoring/prometheus.yml`. | Profil W první noc bez vLLM metrik. |
| Modely přesunuté na `/cold` mají v `/srv/hf` symlinky → každý kontejner, který čte HF cache, musí mít `/cold:/cold:ro`. | 2026-10-27 přesun Qwen3-VL-32B, Qwen2.5-VL-7B, Wan T2V. |

## vLLM modely

| Model (profil, port) | Osvědčené nastavení | Naměřeno | Pozor |
|---|---|---|---|
| Qwen3-30B-A3B-Instruct-2507-FP8 (L, :8000) | `--max-num-seqs 256`, prefix caching, fp8 KV; klient 64 souběžných simulací, `llm_max_conc 256` | UC11: 317 sim./h, 35,8 rozhodnutí/s, 1 840 gen tok/s, 25 700 prompt tok/s, GPU 100 %, 343 W, KV 70 %, 0 preempcí | Instruct-2507 nemá thinking → bez `chat_template_kwargs`. Kompaktní prompt rozhodnutí (51 místo 123 výstupních tokenů) = 1,8× propustnost, výsledky se posunou o −0,02 až −0,04 × ceny. |
| Qwen3-32B-FP8 (A, :8000) | `--max-model-len 32768 --gpu-memory-utilization 0.92 --kv-cache-dtype fp8 --reasoning-parser qwen3`; klient **max 16** simulací | UC11: 16 sim. → 0 fallbacků, 33–40 sim./h, 168 gen tok/s | 48 simulací přetíží (viz výše). Hybridní thinking: vypínat `chat_template_kwargs: {"enable_thinking": false}`. |
| Qwen3-8B-FP8 (S, :8000) | `--max-model-len 8192 --max-num-seqs 256`, prefix caching, `--reasoning-parser qwen3` | UC11: 48 sim. → 180 sim./h, 21,6 rozhodnutí/s, 1 108 gen tok/s, fallback 0,06 % | S `--reasoning-parser qwen3` vLLM ignoruje `/no_think` v textu a vrací prázdný `content`. Thinking jen přes `chat_template_kwargs`; klient, který to neumí (OpenEvolve), přes `uc24/evolve/nothink_proxy.py`. |
| Qwen3-Coder-30B-A3B-Instruct-FP8 (W, **:8003**) | `--max-model-len 65536 --gpu-memory-utilization 0.92 --kv-cache-dtype fp8 --max-num-seqs 32`, prefix caching, `--enable-auto-tool-choice --tool-call-parser qwen3_coder` | UC15: 4 úlohy paralelně, GPU 99–100 % | :8001 drží whisper. Kontejner `vllm-coder` → musí být v Prometheu. |
| Qwen3-VL-8B-Instruct (dočasný `vllm-vl8b`, :8004) | `--max-model-len 8192 --gpu-memory-utilization 0.80 --max-num-seqs 64 --limit-mm-per-prompt '{"image":1,"video":0}' --mm-processor-kwargs '{"max_pixels":802816}'`; klient `max_tokens 1024` | obraz 896² = 784 vizuálních tokenů, prompt ~1 500 tokenů, odpověď ≤ 344 | **Guided JSON:** schéma musí být validní JSON Schema (Draft 2020-12), jinak vLLM (xgrammar i guidance) vrátí 400 na každý request ještě před generováním. Před během: `jsonschema` metaschema check + test `test_vlm_schema_well_formed`. Příčina 2026-10-28: uzel s potomkem jménem `type`. ~37 GB VRAM → profil off. |

## Ostatní modely a knihovny

| Co | Nastavení | Naměřeno | Pozor |
|---|---|---|---|
| WD14 `wd-swinv2-tagger-v3` (onnxruntime) | **`onnxruntime-gpu==1.22.0`** v venv s torch cu128 (CUDA 12); ověřit `session.get_providers()` obsahuje CUDA | 71 851 obr./h na GPU (CPU ~2 200/h) | onnxruntime-gpu 1.31 chce CUDA 13 (`libcublasLt.so.13`) → bez chyby přejde na CPU. |
| NSFW Falconsai / AdamCodd (torch) | batch 64 | 59 434 / 63 523 obr./h | Jednotlivě se pletou (Falconsai 0,96 na výřez oka); brát jen ensemble. |
| SigLIP 2 so400m-512 (torch) | batch 32 | 54 801 obr./h (CPU 426/h) | Zero-shot k ničemu u účesu (shoda 0,03) a výrazu (0,13). |
| Qwen2-Audio-7B-Instruct (transformers, bf16) | batch 8, ≥ 21 GB volné VRAM | 9 918 klipů/h, VRAM špička 18,2 GB | **0 / 400 odpovědí parsovatelných** (2026-10-28) — prompt/parsing neopravený, výsledky soudce zatím nepoužitelné. |
| ACE-Step přes ComfyUI (profil D) | `uc24/gen_bulk.py` posílá klipy po jednom | 575 klipů/h, GPU jen 38 % | Pro objem posílat víc promptů do fronty ComfyUI naráz (batch), gate na CPU dělat mimo smyčku. |
| JAX fyzika (UC14) | float32, batch 1 M | 1,47 M kopů/s | `fma` mění poslední bit (max 9,5e-7) — v testech tolerance 1e-4. |
| Torch head nad SigLIP (UC25) | 4 vlákna, 200 epoch | 91 s, ONNX 0,19 ms/obr. | viz pravidlo o vláknech. |

## Nástroje kolem

- **OpenCode** (UC15): v agentech je pole `permission` (jednotné číslo), sub-agent se volá nástrojem `task`. `opencode session list` po běhu občas vrátí prázdno → exportovat podle `sessionID` z transcriptu (`run --format json`), tokeny dopočítat ze `step_finish` (`uc15/bench/run_bench.py --backfill`).
- **Godot headless** (UC14): reference pro validaci, ne pro trénink.
- **HF downloady**: nepřihlášené (rate-limit warning); velikost repa z API zahrnuje všechny formáty — reálně se stáhne méně (`--exclude 'original/*'`).
