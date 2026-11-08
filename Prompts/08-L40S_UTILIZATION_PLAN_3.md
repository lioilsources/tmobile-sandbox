# L40S_UTILIZATION_PLAN_3.md — třetí vlna: sandbox plně vytížený až do konce, podklady pro pitch T‑Mobile

Handoff pro Opus (Claude Code, Director + sub-sessions). Navazuje na `01-L40S_BEYOND_LLM_USECASES_PLAN.md`
(vlna 1, UC1–UC10) a `04-L40S_USECASES_PLAN_2.md` (vlna 2, UC11–UC25). Stav k 8. 11. 2026 je v `README.md`,
`reports/*.md`, `MODEL_NOTES.md` a `uc*/README.md`. **MODEL_NOTES.md číst před každým během a doplňovat hned.**

Cíl vlny 3 (do konce sandboxu, odhad 2–3 týdny):
1. GPU nikdy nestojí: každé okno má za sebou přerušitelnou výplň (`qctl`, SIGTERM → checkpoint → resume).
2. Vzniknou podklady pro prezentaci T‑Mobile (čísla, grafy, videa, Kč na jednotku) — jsou vedlejší produkt běhů, ne práce navíc.
3. Dokončí se rozdělané UC (27, 18/19, 12, 24/25, 26) tak, aby každé mělo report a JSON v `bench/results/`.

---

## 0. Pravidla (každé už jednou stálo hodiny GPU nebo bylo zablokováno)

- Fronta jen přes `queue/qctl` (`window` pro krátké úlohy s `guard_min`, `cmd` pro dávky, dlouhé běhy jako `filler`). Stav v `/srv/work/queue/jobs/`. Nikdy ruční `tmux` fronty vedle qctl.
- Výstupy na `/cold` (ComfyUI `/srv/work/comfy-out` je symlink). `/srv` jen pro horké cache. Před dlouhým během `df -h /srv /cold`.
- 23 GB RAM je strop: dvě RAM‑těžké úlohy naráz ne; dlouhé CPU/RAM úlohy pod `bench/memguard.sh`. Vlákna `OMP_NUM_THREADS=4`.
- Souběh LLM klientů podle `num_requests_waiting` (cíl waiting ≈ 0, KV < 90 %, preempce 0). Dense Qwen3‑32B max ~7 klientů; MoE 30B‑A3B 64+.
- Každý skript nanečisto (vstupy + 1 položka) před zařazením. Timeouty z naměřené rychlosti × 1,5. Chybovost hlídat v prvních minutách.
- Nic z firemních dat T‑Mobile. Žádný NSFW obsah na firemní GPU. Nekomerční modely (FLUX.1‑dev, Kontext, Fill, XTTS, FFHQ) jen pro interní měření, do podkladů pro T‑Mobile **jen Apache/MIT/CC‑BY** (Qwen*, FLUX‑schnell, Whisper CT2, lc0, Lichess, Registr smluv). Každý nový model: licenci ověřit z HF metadat a zapsat do MODEL_NOTES → „Licence“.
- Repo je PUBLIC: žádné IP, domácí cesty, jména NAS, e‑maily, tokeny. `Prompts/02-*`, `03-*`, `05-*`, `06-*`, `uc25/calib_gen.py` nikdy necommitovat; `04`, `07` a tento soubor commitnout jen na pokyn uživatele.
- SPARK jen přes `sparkctl`. NAS, Spark a domácí soubory na VM nekopírovat bez výslovného OK uživatele.
- Po rebootu VM: `/srv/work/.gpu_profile` resetovat, qctl a hlídače (crashwatch, memguard) spustit znovu.
- Rozhodnutí uživatele řešit souběžně s během výplně, ne místo něj.

---

## 1. Výplně (fillers) — co drží GPU, když nic jiného neběží

Pořadí = priorita. Každá výplň musí umět SIGTERM → checkpoint ≤ 60 s → resume od posledního checkpointu.

| # | Výplň | Profil | Stav | Checkpoint |
|---|---|---|---|---|
| F1 | **UC27 agenttune LoRA SFT → DPO** (`/cold/agenttune/data/{sft,dpo}.jsonl`, Qwen3‑Coder‑30B‑A3B QLoRA, Unsloth) | GPU sama (trénink) | dataset na VM, skript chybí | adapter každých N kroků do `/cold/agenttune/ckpt/` |
| F2 | **UC12 AlphaZero, jeden pokus s větší sítí** 8×256, `--sims 32`, 24 GPU h, start z iter 22 346 | GPU sama (JAX frakce 0,90) | `uc12/run_az.sh` | hotovo (resumable) |
| F3 | **UC18/19 scoring na GPU** (PickScore, SigLIP 2, WD14, Qwen3‑VL‑8B párový soudce) nad hotovou maticí v `/cold/w2/uc18` | D + `vllm-vl8b` :8004 | matice vygenerovaná, scoring jen CPU smoke | po blocích (manifest.jsonl) |
| F4 | **UC25 plný re‑tag** galerie FinetuneGallery (jen SFW tiery) | GPU (WD14/SigLIP/NSFW), 60–70 k obr./h | hotový harness `uc25/gpu_batch.sh` | po dávkách |
| F5 | **Lab stylů vlna 8** (Ol1nLLM, SDXL rodina) — jen když uživatel dodá zadání | D | čeká na zadání | po blocích |

Pravidlo: F1 má přednost, jakmile existuje tréninkový skript (sekce 2.2). Do té doby běží F2. F3/F4 jsou „denní“ výplně vedle profilu D, F2 potřebuje GPU samu.

---

## 2. Pracovní balíky (každý = sub-session, končí `reports/<uc>.md` + `bench/results/<UC>_*.json` + řádky v MODEL_NOTES)

### 2.1 P1 — Podklady pro pitch (první týden, krátká okna přes den) — **nejvyšší priorita**

Cíl: ke každému ze tří produktových dem a k UC9 mít **reprodukovatelný běh + Grafana PNG + 60–90 s screencast + tabulku Kč/jednotku**. Modely jen s komerční licencí.

1. **Demo 3 dokumenty** (profil C): znovu `demos/03_documents/run.sh` 110 i 150 dpi; uložit `/cold/pitch/demo3/{grafana.png, metrics.json, sample_in.png, sample_out.json}`. Screencast: faktura → JSON v Open WebUI nebo curl.
2. **Demo 1 call centrum** (profil B): znovu s 200 hovory; výstup transcript + štítky jednoho hovoru jako ukázka; graf hovorů/h.
3. **Demo 2 multi‑tenant API** (profil B): load test 3×2 min; Grafana dashboard „AI API – tenanti“ PNG; usage.jsonl → tabulka tokenů/tenant/Kč.
4. **UC9 Registr smluv**: nepřepočítávat; z existujících logů/CSV udělat graf propustnosti a tabulku Kč (≈ 55 Kč/h VM). Ověřit, že CSV neobsahuje nic mimo veřejná data registru.
5. **Beyond‑LLM vizuály** (profil D / CPU): Kirian replay GIF (`uc4/replay_gif.py`), UC12 Elo graf z ligy, 200 3D předmětů jako turntable video (6–8 ks), mozaika 7 stylů UC10, 21 ikon UC28 kontaktní arch. Ukládat do `/cold/pitch/visuals/`.
6. **Tabulka ekonomiky** `reports/pitch-ekonomika.md`: jednotka, propustnost/h, Kč/jednotku při 55 Kč/h, kWh (z DCGM), pro: strana dokumentu, hovor, smlouva, ilustrace, 3D předmět, šachová pozice, QLoRA doladění. Bez odhadů — jen naměřené.
7. **Zpětná vazba k platformě** `reports/pitch-platforma.md`: incidenty (hard stop 25. 10., RAM OOM, steal CPU, změna IP, netplan), co pomohlo, co doporučit (RAM 64 GB, dedikované vCPU, snapshoty, 2. GPU). Bez IP a časů mimo datum.

Hotovo = `/cold/pitch/` kompletní, kopie PNG/GIF/MD v `reports/pitch/` (repo), screencasty jen na /cold + NAS (ne do repa).

### 2.2 P2 — UC27 agenttune: trénink a eval (výplň F1)

1. `uc27/train_lora.sh`: Unsloth QLoRA na Qwen3‑Coder‑30B‑A3B‑Instruct (ověřit, že Unsloth podporuje MoE; pokud ne, fallback Qwen3‑Coder‑7B nebo Qwen2.5‑Coder‑14B — zapsat do MODEL_NOTES). SFT 2 epochy → DPO 1 epocha. Checkpoint každých 200 kroků, SIGTERM handler.
2. Merge adapteru → vLLM profil W2 (`50_vllm/switch.sh W2`), stejný harness jako baseline (`uc15/bench/run_bench.py`, 25 úloh) → pass@1 před/po.
3. Promotion gate podle `uc27/README.md`. Report `reports/uc27-agenttune.md`.

### 2.3 P3 — UC18/19 dokončit (výplň F3)

Scoring na GPU, Bradley‑Terry router `uc18/route.py`, český prompt → ideální prompt per checkpoint (`uc19/translate.py`), vyhodnocení na 50 ručních promptech. Report s `licence_restricted` u FLUX‑dev obrázků (do repa nikdy). `uc18/to_ingest.py` bez `--include-restricted` → PR do FinetuneGallery (větev `l40s-w2/uc18`).

### 2.4 P4 — UC12 rozhodnutí (výplň F2)

Jediný pokus 8×256 (24 GPU h) → liga 9 000 partií proti Dart Easy/Medium/Hard. Kritérium: Elo proti Hard ≥ +50 (CI nad 0) → pokračovat dalších 48 h; jinak UC12 ukončit, nejlepší checkpoint exportovat (`uc12/export.py` → ONNX pro Flutter), report `reports/uc12-orbitron-alphazero.md`, výplň přepnout na F1/F3.

### 2.5 P5 — UC24 / UC25 na reálných datech

- UC24: 500 skutečných klipů (ACE‑Step hudba + Stable Audio Open SFX — **licenci SAO ověřit, Stability AI Community License má limit obratu**; pokud nevyhovuje, jen ACE‑Step) → sfxgate + soudci → `rating_sheet.py` 150 klipů pro lidské hodnocení (uživatel).
- UC25: fáze B (VLM Qwen3‑VL‑8B) znovu s opraveným schématem; 300 lidských labelů (uživatel); conformal prahy na reálných datech.
- Hotovo = aktualizované reporty uc24/uc25 s reálnými čísly, κ z lidských labelů.

### 2.6 P6 — UC26 BestBet (CPU, překrývá se s GPU výplněmi)

Ingest fáze 1 podle `Prompts/07-BESTBET_VALIDATION_PLAN.md` (zdroj dat dodá uživatel), walk‑forward backtest, verdikt GO/NO‑GO. Na GPU jen pokud LLM‑features (30B‑A3B) — jinak celé na CPU pod memguard.

### 2.7 P7 — Nové UC pro pitch (jen po souhlasu uživatele; každé ≤ 2 dny)

| UC | Co | Proč pro T‑Mobile |
|---|---|---|
| UC29 | **RAG nad vlastními dokumenty česky**: Qwen3‑Embedding‑0.6B + bge‑reranker + Qwen3‑30B‑A3B, 200 otázek/odpovědí nad veřejnými českými dokumenty (zákony, výroční zprávy), přesnost + latence + Kč/dotaz | nejčastější B2B poptávka, doplní Demo 2 |
| UC30 | **Schůzky**: Whisper diarizace (pyannote — licence ověřit) + souhrn + úkoly; vlastní nahrávky | druhý produkt vedle call centra |
| UC31 | **Kalibrace LLM‑soudce češtiny**: Gemma 3 27B vs Qwen3‑32B vs Mistral Small jako hodnotitel českých odpovědí proti 100 lidským hodnocením | rozhodne, který model nasadit v Demo 2 |

### 2.8 P8 — Infra, které musí doběhnout

- Witness VM založit (bootstrap v `90_witness/`), Telegram token dodá uživatel. Alerty: GPU util < 5 % déle než 15 min = „fronta vyschla“.
- Zálohy: ověřit, že `/cold/pitch` a `/cold/agenttune/ckpt` jsou v záloze na NAS (`80_backup.sh` vzor).
- Tailscale na L40S (hotovo 8. 11.): nic neměnit; relay token jen na VM.

---

## 3. Rozvrh (týden = 7 × 24 h GPU; okna přes den krátká, noc = výplň)

| Den | Den (okna ≤ 3 h) | Noc (výplň) |
|---|---|---|
| 1–2 | P1.1–P1.3 (dema C, B), P1.5 vizuály (D) | F2 UC12 8×256 |
| 3 | P1.4, P1.6, P1.7 (CPU), P2.1 skript + smoke | F2 dokončit → liga (CPU) |
| 4–6 | P3 scoring (D) jako denní výplň; P5 UC24 generování (D) | F1 UC27 SFT → DPO |
| 7 | P2.2 eval W2; P4 rozhodnutí | F1 / F3 |
| 8–10 | P5 UC25 fáze B; P7 (pokud schváleno) | F1 pokračování nebo F4 |
| 11+ | P6, zbytek P7, finální reporty | F4 / F5 |

Každý den ráno: `qctl status`, `nvidia-smi`, `df`, `tail` logů; GPU util za posledních 24 h z Prometheu do `reports/pitch/utilizace.md` (graf na konec vlny = důkaz 24/7).

---

## 4. Rozdělení pro Director

- `w3-pitch` — P1 (má přednost na GPU přes den, krátká okna s `guard_min`).
- `w3-train` — P2 + F1 (noc).
- `w3-score` — P3 + F3/F4 (denní výplň vedle D).
- `w3-rl` — P4 + F2 (dokud neskončí rozhodnutí).
- `w3-quality` — P5 (čeká na licenci SAO a lidské labely).
- `w3-bestbet` — P6 (CPU).
- Každá sub-session: před během číst MODEL_NOTES, po běhu doplnit řádek; výsledky JSON do `bench/results/`; report v češtině bez IP a cen portálu (jen „≈ 55 Kč/h“).

## 5. Definice hotovo

- GPU util průměr za vlnu ≥ 80 % (graf z Prometheu v `reports/pitch/utilizace.md`).
- `/cold/pitch/` + `reports/pitch/` obsahují pro Demo 1/2/3 a UC9: PNG z Grafany, metrics JSON, screencast, Kč/jednotku.
- UC27 má pass@1 před/po; UC12 má rozhodnutí a export; UC18/19, UC24, UC25 mají finální report.
- MODEL_NOTES má řádek ke každému novému modelu/nástroji včetně licence.
- Nic v repu neporušuje pravidla sekce 0.
