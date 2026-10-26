# L40S_USECASES_PLAN.md — T-Mobile GPUaaS sandbox: utilizace, benchmark, report

Handoff pro Opus (Claude Code, Director + sub-sessions). Cíl: za ~4 týdny na L40S sandboxu
(T-Mobile Business Cloud) rozjet 5 use casů napojených na existující projekty, změřit utilizaci
a throughput, a vyprodukovat report + demo, které přesvědčí T-Mobile ke spolupráci.

Vlákno v Director session: `l40s-sandbox`. Každý UC = vlastní sub-session, každá končí
`report/UC<n>.md` + JSON metrikami v `bench/results/`.

---

## 0. Kontext a omezení

| Položka | Hodnota |
|---|---|
| GPU | 1× NVIDIA L40S, 48 GB GDDR6, Ada (sm_89), x86_64 |
| VM | KVM na t-cloud.eu, Ubuntu z vlastního ISO (viz `TMOBILE_SANDBOX_PROVISION_PLAN.md` v2), 1 TB "superfast" disk |
| Čas | 1 měsíc sandboxu → držet se timeline v §6, snapshoty VM po každé fázi |
| Doma | DGX Spark (GB10, ARM64, 128 GB unified, sm_121) — slouží jako **srovnávací baseline** a zdroj už vyladěných pipeline (ComfyUI, vLLM, AiStack) |
| Co L40S umí lépe než Spark | ~3–4× vyšší compute a bandwidth, x86 ekosystém bez wheel-martyria (JAX, lc0, Triton, FlashAttention fungují z PyPI) |
| Co Spark umí lépe | 128 GB paměti → modely >40 GB; na L40S nutno kvantizovat / offloadovat |

Pravidla:
- Žádný stav mimo VM se neztratí: `/data` = jediné místo pro váhy, datasety, výsledky; denní `rsync` do domácí NAS (domácí NAS).
- Každé spuštění loguje `nvidia-smi dmon` + DCGM → report utilizace je vedlejší produkt, ne dodatečná práce.
- Nic z firemních dat T-Mobile; jen vlastní data (matriky, assety her, vlastní video).

---

## 1. Provisioning

### 1.1 Base OS (po bootu z ISO)
```bash
sudo apt update && sudo apt install -y build-essential git curl htop tmux zstd jq ffmpeg \
  xvfb mesa-utils libgl1-mesa-dev libegl1-mesa-dev python3-venv
# NVIDIA driver 570+ (podpora CUDA 12.8) — z NVIDIA repo, ne z ubuntu-drivers
# ověřit: nvidia-smi → L40S, 49140 MiB
```
- Docker + `nvidia-container-toolkit`, `docker run --gpus all nvidia/cuda:12.8.1-base nvidia-smi` musí projít.
- `uv` pro Python venvy (jedno venv na UC, Python 3.11; DreamerV3 vyžaduje 3.11+).
- `/data` layout:
```
/data/models/hf        # HF_HOME=/data/models/hf
/data/models/comfy     # symlink do ComfyUI/models
/data/datasets/{matriky,kirian,chess,shaders}
/data/bench/{logs,results}
/data/snapshots-notes.md
```
- Secrets: `~/.env` (HF_TOKEN, CIVITAI_TOKEN, SSH klíč na domácí NAS + Spark). Nic do repo.

### 1.2 Monitoring (povinné před prvním UC)
- `dcgm-exporter` (docker, port 9400) + Prometheus + Grafana (docker-compose v `infra/monitoring/`).
- Dashboard: GPU util %, SM occupancy, mem used, power W, temp, PCIe TX/RX — exportovat PNG per UC pro report.
- `bench/gpu_log.sh <label>` → spustí `nvidia-smi dmon -s pucvmt -d 1 > /data/bench/logs/<label>_<ts>.csv` na pozadí; každý benchmark skript ho volá.
- `bench/summarize.py` → z CSV vyrobí: avg/p95 util, avg power, energie (Wh), čas, a zapíše do `results/<label>.json`.

### 1.3 Servery
| Služba | Jak | Port |
|---|---|---|
| vLLM | `vllm/vllm-openai:latest` (x86 → bez forku!) | 8000 |
| ComfyUI | git clone + venv, modely přes symlink `/data/models/comfy` | 8188 |
| LiteLLM gateway | stejná config jako doma, upstream = lokální vLLM | 4000 |
| Grafana | docker | 3000 |
| lc0 (šachy) | build s `-DUSE_CUDNN`, nebo oficiální binárka cuda | — |
Vystavit přes Cloudflare Tunnel jako `l40s-*.<doména>` (stejný pattern jako doma) — demo pro T-Mobile pak běží z prohlížeče.

### 1.4 Co stáhnout (odhad ~250 GB, pořadí = priorita)

| # | Model / data | Zdroj | Velikost | Pro UC |
|---|---|---|---|---|
| 1 | Qwen/Qwen3-VL-8B-Instruct (nebo Qwen2.5-VL-7B-Instruct jako fallback) | HF | ~17 GB | UC1 |
| 2 | rednote-hilab/dots.ocr | HF | ~3 GB | UC1 |
| 3 | allenai/olmOCR-2-7B (ověřit aktuální název) | HF | ~16 GB | UC1 |
| 4 | Kraken HTR modely (`kraken get`), zkusit model pro německou kurentu/novogotickou | Zenodo přes kraken | <1 GB | UC1 |
| 5 | Qwen/Qwen3-32B-AWQ nebo Qwen3-30B-A3B (text LLM pro OpenEvolve + kvízy) | HF | ~18 GB | UC2, UC3, UC5 |
| 6 | facebook/vjepa2-vitl-fpc64-256 + vith | HF | ~3 GB | UC2, UC5 |
| 7 | lc0 síť (BT4 / T82 series, `lczero.org/play/networks`) + Stockfish 17 | web | <1 GB | UC3 |
| 8 | Lichess puzzle DB (`lichess_db_puzzle.csv.zst`) | database.lichess.org | ~1 GB | UC3 |
| 9 | FLUX.1-dev FP8 (Comfy-Org), + text encodery, VAE | HF | ~17 GB | UC5 |
| 10 | Wan2.2 T2V/I2V A14B FP8 (high+low noise, Comfy-Org repack) + Lightning LoRA | HF | ~30 GB | UC5 |
| 11 | ACE-Step (hudba), stabilityai/stable-audio-open-1.0 (SFX) | HF | ~10 GB | UC5 |
| 12 | yuvalkirstain/PickScore_v1, THUDM/ImageReward, laion/CLIP-ViT-H-14 | HF | ~6 GB | UC2, UC5 |
| 13 | DreamerV3 (`pip install dreamerv3`, JAX cuda12) | PyPI | — | UC4 |
| 14 | OpenEvolve (`pip install openevolve`) | PyPI | — | UC1, UC2, UC5 |
| 15 | Vlastní data: sken matrik ze `genealogy-matriky`, Kirian Linux build, MirrorBooth shadery | domácí NAS/Spark | ~50 GB | všechny |

Pozn.: Wan 2.2 A14B FP8 = 2×~14 GB; na 48 GB jde, ale s `--lowvram` / block swap. HunyuanVideo vynechat (EU licence).

---

## 2. UC1 — Genealogie OCR (`genealogy-matriky`)

**Cíl:** rozhodnout, kterým modelem přepisovat matriky, a změřit, co to stojí. Plus přesunout
record-linkage (ML_PLAN.md) z ručního skóre na trénovaný/evolvovaný model.

**Vstup:** 200 stran ze `/data/datasets/matriky`, z toho 30 stran s ručním ground-truth
přepisem (Olin dodá; pokud není, vybrat strany, kde existuje indexace archivu → polo-GT).

**Kroky:**
1. Harness `uc1/run_ocr.py --engine {qwen3vl,dotsocr,olmocr,kraken}` → JSONL (strana, text, latence, tokens).
2. Metriky: CER/WER proti GT (jiwer), **strukturovaná extrakce** (jméno, datum, obec) jako F1 — to je skutečný cíl, ne surový text. Prompt pro VLM vrací JSON podle schématu z `genealogy/`.
3. Throughput: stran/hod při batchi (vLLM pro Qwen3-VL, `max_num_seqs` ladit), GPU util z DCGM.
4. **Non-LLM část:** OpenEvolve evolvuje skórovací funkci record-linkage: evaluátor = Go test `go test ./genealogy -run TestMatchBench` nad označenými páry (precision/recall), kandidát = `match_score.go`. LLM backend = lokální Qwen3-32B přes vLLM. 200 generací, island=4.
5. Alternativa v ML_PLAN: trénovat malý gradient-boost / MLP na stejných párech a porovnat s evolvovanou heuristikou.

**Výstup:** tabulka engine × (CER, F1 extrakce, stran/hod, Wh/stranu, GPU util), doporučení, PR do `genealogy-matriky` s nejlepším enginem za flagem.

---

## 3. UC2 — Pipeline shaderů pro MirrorBooth (ShaderGen + ShaderBattle)

**Cíl:** z LangGraph pipeline (ShaderGen) a benchmarku (ShaderBattle) udělat **evoluční**
generátor GLSL efektů pro MirrorBooth s automatickým hodnocením — bez člověka v loopu.

**Architektura:**
- Headless render: `moderngl` + EGL na L40S (bez X), 512×512, 60 frames per kandidát; sanity přes `glslViewer --headless` pokud moderngl zlobí.
- Evaluátor (`uc2/evaluate.py`) vrací skalár + složky:
  - **kompiluje + ≥30 FPS** na referenčním mobilním rozpočtu (odhad: čas na L40S × faktor; kalibrovat na reálném telefonu s 5 shadery),
  - **estetika** PickScore/ImageReward na 5 vzorcích snímků proti textovému zadání ("kaleidoscope mirror, neon, smooth"),
  - **pohyb** V-JEPA 2 embedding 64-frame klipu: vzdálenost od referenčních efektů (chceme "živé", ne statické ani blikající) + penalizace za flicker (rozptyl embeddingů mezi okny),
  - **validita pro MirrorBooth**: shader musí respektovat uniformy `iTime, iResolution, iChannel0` (kamera) — statická kontrola.
- OpenEvolve: seed = 10 existujících shaderů z MirrorBooth, LLM = Qwen3-32B (fallback: Claude přes LiteLLM pro srovnání kvality), MAP-Elites dimenze = (složitost instrukcí, "energie" pohybu).

**Kroky:** 1) render harness + evaluátor na stávajících shaderech → baseline skóre; 2) evoluce 3× 300 generací s různými zadáními; 3) top-20 vyrenderovat do galerie (HTML) pro ruční výběr; 4) export vybraných do Flutter `fragment_program` formátu + test na iPhone.

**Výstup:** galerie, GPU util profil (evaluátor je GPU-bound: render + V-JEPA + PickScore paralelně → cíl >80 % util), `SHADERGEN_EVOLVE.md` s nastavením, PR do MirrorBooth s novými efekty.

---

## 4. UC3 — Kvízy do PrincessChess

**Cíl:** generátor výukových úloh ("kvízů") s kalibrovanou obtížností pro 5leté dítě + storytelling obal.

**Non-LLM jádro = lc0 na L40S (batch analýza):**
1. Lichess puzzle DB filtrovat na rating < 1000, témata `mateIn1, hangingPiece, fork, pin` + **vlastní pozice** z mini-her (pouze králové+pěšci, dáma vs král atd.) generované skriptem.
2. lc0 ohodnotí každou pozici (policy + Q): obtížnost = (a) kolik kandidátních tahů má policy >5 % (méně = snazší), (b) hloubka, v jaké správný tah vyhraje nad druhým nejlepším, (c) Stockfish multipv pro ověření jednoznačnosti. Cíl: 1M pozic/hod — lc0 na L40S s batch 512 to zvládne; to je zároveň dobrý utilizační benchmark.
3. Výsledné "kid-level" pásmo: řešení = 1 tah, jediný správný, motiv jasný. Clustrovat podle lekcí z `PRINCESSCHESS_LESSONS.md`.
4. **Protivník:** z lc0 udělat nastavitelného soupeře — temperature sampling z policy + omezení na 1-ply → "princezna úroveň 1–5". Měřit, jak často dítě-simulant (random + 1 heuristika) vyhraje.
5. LLM (Qwen3-32B, česky) vygeneruje ke každému kvízu 1 větu příběhu + nápovědu v jazyce princezen; validace přes JSON schema; 2000 kvízů.

**Výstup:** `princess_quizzes.json` (FEN, řešení, obtížnost 1–5, lekce, text CZ), throughput pozic/hod a util, PR do PrincessChess (datový soubor + loader).

---

## 5. UC4 — Hraní Kirianu a hledání těžkých levelů

**Cíl:** RL agent hraje Kirian, výstup = křivka obtížnosti per level, heatmapa smrtí, nalezené exploity. Tohle je **hlavní demo pro T-Mobile** (vizuálně nejsrozumitelnější).

**Problém:** Flutter/Flame nemá gym rozhraní. Řešení ve 2 fázích:
1. **Debug build Linux** (`flutter build linux --dart-define=RL=true`): v RL módu hra otevře WebSocket (`localhost:7000+i`), přijímá akce (left/right/fire/none) na fixním ticku 20 Hz, posílá stav (pozice hráče, nepřátel, střel, HP, score, level) + volitelně 96×96 grayscale frame. Běží pod `xvfb-run`, 8–16 instancí paralelně (CPU ~20 jader na VM — ověřit, jinak snížit). Deterministický seed per epizoda.
2. Python `KirianEnv(gym.Env)` nad WebSocketem; vektorizace přes `gymnasium.vector.AsyncVectorEnv`.

**Agenti:**
- Baseline: PPO (CleanRL, stavový vektor) — rychle, ověří env.
- Hlavní: **DreamerV3** (`pip install dreamerv3`, JAX cuda12) na obrázcích 96×96, config `size50m`, `train_ratio 32`; budget 48 h GPU. Dreamer tu dává smysl: sběr dat je drahý (reálná hra), world model se naučí fyziku hry.
- Pro srovnání utilizace: Dreamer ~GPU-bound, PPO ~env-bound → reportovat oba profily.

**Analýza:** per level: úspěšnost, průměrné HP při dohrání, počet pokusů do dokončení, místa smrti (heatmapa na pozadí levelu), "cheesy" strategie (agent se nehýbe a vyhrává → level je špatně navržený). Levely s úspěšností <10 % po 200 epizodách = kandidáti na zjednodušení; >90 % hned = nudné.

**Výstup:** `kirian_difficulty_report.html` (grafy + heatmapy + 5 GIFů z hraní agenta), PR do Kirian (RL mód za flagem, `KirianEnv`), srovnání Dreamer vs PPO.

**Kolo 1 (hotovo 26. 10.):** PPO se stavovým vektorem, headless `flutter test` místo Linux buildu, 12 M kroků za 1,5 h na CPU.
Agent dokončí 10 z 18 sektorů. V 8 sektorech našel bezpečný roh a kempuje do limitu, protože odměna platila za přežití.

**Kolo 2 — „vystřílej všechno“:** agent nemá jen přežít, ale sestřelit všechny nepřátele v sektoru.
- **Pozorování ze hry:** server posílá `k = [sestřeleno, celkem, vyčištěných formací, skončených formací, formací, uletělo]`
  ze `Sector.fleets` (`Fleet.kills/count`). Formace je vyčištěná, když padli všichni (ve hře to spouští bonus).
- **Odměna:** +0,3 za sestřel, −0,3 za nepřítele, který uletěl, −0,003 za krok (místo +0,002), −5 za smrt i vypršený čas,
  dokončení +5 + 10 × podíl sestřelených + 5 navíc za 100 %. Kempování je tím ztrátové.
- **Agent:** pozorování navíc 3 čísla (podíl sestřelených, skončených formací, uletělých). Start z agenta kola 1,
  nové vstupy mají nulové váhy, takže agent začne s chováním z kola 1. Sektory, kde agent kempoval, se losují 2× častěji. 16 M kroků.
- **Metriky po sektorech:** dokončeno %, **vyčištěno %** (dokončeno a sestřeleni všichni), sestřeleno %, uletělo,
  smrt. Srovnání s agentem kola 1 na stejných metrikách.
- **Otevřená otázka:** každá epizoda začíná s novou lodí a základní zbraní. V pozdějších sektorech 100 % sestřelů nemusí jít
  bez vylepšení zbraní z kampaně. Pokud vyčištěno zůstane 0 % i při vysokém podílu sestřelů, další kolo dá lodi zbraň
  odpovídající postupu v kampani.

**Kolo 3 — kampaň s obchodem:** agent si za vystřílený kredit kupuje zbraně a nevyužitý kredit si nechává na další sektor.
- **Průběh běhu:** sektory od 0 za sebou se stejnou lodí. Kredit z kola 2 i výbava zůstávají, jako v endless režimu hry.
  Po dokončeném sektoru server udělá totéž co aplikace: `advanceToNextSector` + `openComCenter`. Pak přijdou nákupy a odlet
  (`resetVessel` + `resumeFromComCenter`). Běh končí smrtí, vypršeným časem v sektoru nebo sektorem 17.
- **Obchod:** 8 akcí, které odpovídají tlačítkům ComCenteru (bez prodeje): odlet, upgrade přední zbraně, další přední zbraň,
  levá a pravá boční zbraň (koupit nebo upgradovat), upgrade generátoru, další boční zbraň vlevo a vpravo. Akce, na které není
  kredit, je na max. úrovni nebo není odemčená skóre, jsou maskované. Nejvýš 12 nákupů na návštěvu.
- **Dva agenti:**
  - **pilot:** PPO jako v kole 2 (epizoda = sektor). Startuje z pilota kola 2 a navíc vidí výbavu a kredit (OBS_V3).
  - **obchodník:** malá síť (PPO). Každý nákup před sektorem k+1 dostane výnos Σ 0,8^m · R(k+1+m) z odměn pilota
    v dalších sektorech, takže se vyplatí i ušetřit kredit na později.
- **Vyhodnocení:** 30 kampaní pro každou variantu (naučený obchodník, pravidlo „nejdřív lepší přední zbraň, pak upgrady“,
  bez nákupů). Měří se počet dokončených sektorů, sestřely po sektorech, utracený a nevyužitý kredit a výbava na konci.
  Kód: `uc4/ppo_campaign.py`, server `test/rl/rl_server_test.dart` (příkazy `next`, `shop`, `launch`).

---

## 6. UC5 — Promptování grafických / video / hudebních / SFX modelů

**Cíl:** (a) throughput benchmark L40S vs Spark pro ComfyUI stack, (b) **automatická evoluce promptů** proti objektivním hodnotitelům, (c) náhrada ElevenLabs pro Kirian (`AUDIO_OSS_PLAN.md`).

**Benchmark (a):** identické workflow JSONy ze Sparku: FLUX.1-dev 1024², 20 kroků; Wan 2.2 I2V 480p 81 frames; ACE-Step 60 s skladba; Stable Audio Open 5 s SFX. 10 opakování, medián s/obrázek, s/klip, VRAM peak, W. Tabulka L40S vs Spark (Spark čísla změřit doma stejným skriptem `bench/comfy_bench.py`).

**Evoluce promptů (b):** OpenEvolve, kde "program" = prompt + parametry (cfg, steps, sampler), evaluátor:
- obrázky: PickScore + ImageReward vůči zadání (např. "Kirian skin: Sega Genesis 1992 style ship sprite sheet") + CLIP konzistence stylu mezi 4 vzorky,
- video: V-JEPA 2 — plynulost (nízký rozptyl embeddingů), "sémantika pohybu" (podobnost s referenčním klipem),
- hudba/SFX: CLAP skóre (laion/clap-htsat-unfused) vůči textu + loudness/klip detekce.
Výsledek: pro 5 typů assetů Kirianu nalezený "nejlepší prompt-recept" a číslo, o kolik je lepší než ručně psaný. To je B2B příběh: *optimalizace generování bez člověka*.

**Audio (c):** vygenerovat kompletní sadu SFX pro Kirian (30 položek) a 3 skladby; A/B poslech proti ElevenLabs; cena per asset = Wh × tarif.

**Výstup:** benchmark tabulka, `prompt_recipes.json`, audio pack do Kirian repo, util profil.

---

## 7. Metodika benchmarku a report pro T-Mobile

Report `report/T-MOBILE_L40S_REPORT.md` (+ PDF, + 10 slidů) obsahuje:
1. **Shrnutí** — 5 use casů, 1 stránka, pro business publikum.
2. **Utilizace** — per UC: avg/p95 GPU util, power, Wh, "idle gap" (kde pipeline GPU nevytěžuje a proč). Cíl ukázat: správně postavená pipeline drží L40S >80 %; naivní (sekvenční) <40 %.
3. **Throughput + cena** — jednotka práce (strana OCR, obrázek, klip, 1M pozic, 1 h RL) → s a Wh → Kč při tarifu GPUaaS vs cena API (OpenAI/Replicate/ElevenLabs). Jedna tabulka.
4. **L40S vs Spark** — kde je L40S 3× rychlejší, kde Spark vyhrává pamětí; tvrzení "jeden L40S za měsíc obslouží X".
5. **Co je unikátní** — non-LLM modely (V-JEPA, Dreamer, lc0, evoluce) jako evaluátory a optimalizátory; to konkurence na GPUaaS nenabízí.
6. **Demo** — 3 živé URL: Kirian agent hraje (stream), galerie evolvovaných shaderů, OCR matrik s extrakcí.
7. **Nabídka** — co Olin dodá (use case workshop, pipeline, utilizační audit), ceník.

Skripty: `bench/report_build.py` generuje tabulky z `results/*.json` a grafy (matplotlib) → žádná ruční čísla.

---

## 8. Timeline (4 týdny)

| Týden | Co | Snapshot |
|---|---|---|
| 1 | Provisioning §1 kompletní, monitoring, downloads 1–8, UC1 harness + první čísla, UC3 lc0 batch běží přes noc (utilizační test) | `w1-base` |
| 2 | UC1 OpenEvolve, UC3 kvízy hotové, UC4 Kirian RL mód + PPO baseline, downloads 9–12 | `w2-uc13` |
| 3 | UC4 DreamerV3 48 h run, UC2 render harness + evoluce, UC5 benchmark L40S vs Spark | `w3-rl` |
| 4 | UC5 evoluce promptů + audio pack, UC2 galerie, report + slidy + 3 demo URL, PR do všech repo | `w4-final` |

Pořadí priorit, když dojde čas: UC4 (demo) > UC1 (praktická hodnota) > UC5 (benchmark) > UC3 > UC2.

---

## 9. Rozdělení pro Director

- `l40s-infra` — §1, monitoring, downloads, Cloudflare tunnel; vlastní `infra/` repo `lioilsources/l40s-sandbox`.
- `l40s-uc1` … `l40s-uc5` — každá vlastní venv, vlastní adresář `uc<n>/`, výsledky jen do `/data/bench/results/UC<n>_*.json`.
- `l40s-report` — startuje v týdnu 3, konzumuje JSONy, nesahá do GPU.
- Sdílený hub: stav modelů ke stažení (`downloads.yaml` se status sloupcem), GPU rezervace (jen jedna session smí mít GPU-těžký job; ostatní čekají — L40S nesdílet mezi UC4 a UC5 naráz, oba jsou >30 GB VRAM).

## 11. Doplněné projekty, které na L40S fitují (nižší priorita, zapojit když zbyde čas nebo jako náhrada za UC2)

### UC6 — Hajánek: klonování hlasu rodiče (silný B2B příběh "on-prem hlas, data neopouští firmu")
- Modely: `SWivid/F5-TTS` (+ multilingual/CZ fine-tune pokud existuje — ověřit), `coqui/XTTS-v2` (CZ nativně), GPT-SoVITS jako třetí.
- Benchmark: 30 s referenčního hlasu → 10 min usínací meditace CZ; metriky: speaker similarity (ECAPA/resemblyzer cos), WER zpětným ASR (Whisper large-v3), RTF, Wh/min audia.
- Výstup: nejlepší engine za flagem v Hajánek backendu, PR, čísla do reportu. Kód-sdílení s UC5(c) audio.

### UC7 — RealEyes LoRA: trénink místo inference
- L40S je pro trénink LoRA (FLUX / SDXL) výrazně rychlejší než Spark a `kohya_ss`/`ai-toolkit` jdou z PyPI bez buildů. Spustit stejný `EYES_TRAINING_PLAN.md` dataset: čas/epochu, VRAM, kvalita (FID + ruční grid) — přímé srovnání Spark vs L40S pro **trénink**, které v reportu chybí (UC5 měří jen inference).
- Rozšířit: hromadný trénink 5 LoRA stylů pro Kirian skiny → vstup do UC5 evoluce promptů.

### UC8 — UGC Factory / FantasyCharacter: img-to-3D batch
- TRELLIS (`microsoft/TRELLIS-image-large`) a `tencent/Hunyuan3D-2.1` na L40S (48 GB stačí); 200 obrázků → GLB, s/model, Wh/model, poměr validních meshů (manifold, poly budget pro Roblox).
- Napojení: výstup rovnou do `ugc-backend` fronty → market.ol1n.com dostane 200 nových položek. Dobré demo "GPU jako výrobní linka".

### UC9 — Leads z Registru smluv: batch LLM extrakce (největší throughput benchmark)
- 500k smluv z Registru smluv (open data) → vLLM Qwen3-30B-A3B, strukturovaná extrakce (odvětví, předmět, hodnota, dodavatel) s guided JSON; `max_num_seqs` 256+, prefix caching.
- Metriky: dokumentů/hod, tokenů/s, GPU util (cíl >90 %, batch inference je ideální utilizační showcase), cena/1000 dokumentů vs API.
- Výstup: naplněná DB pro `leadscout` (LEADS_PLAN.md) + segmentace firem pro olin.now/business — tj. report sám generuje leady.

### UC10 — Storyteller / anime příběhy: dávková generace ilustrací a klipů
- FLUX-schnell 10k ilustrací pro Storyteller (země × fáze příběhu) a 50 jednominutových klipů z Wan 2.2 pro anime-story videa; sdílí harness s UC5(a), jen jiný workload — sleduje udržení utilizace při dlouhém běhu (24 h soak test, teploty, throttling).

Doplnit do §1.4: F5-TTS + XTTS-v2 (~5 GB), TRELLIS + Hunyuan3D-2.1 (~20 GB), FLUX.1-schnell FP8 (~12 GB), Whisper large-v3 (~3 GB), ai-toolkit/kohya (PyPI).

Nefitují / nechat na Sparku: vLLM GB10 fork (sm_121-specific), Roomscan (mikrofon telefonu), Předávka, TerraReality (není GPU úloha), AILineage.

## 10. Definice hotovo
- 5× `report/UC<n>.md` s čísly z `results/`, ne odhady.
- Grafana export PNG per UC.
- 4 PR (genealogy-matriky, MirrorBooth, PrincessChess, Kirian), všechny za feature flagem, CI zelené.
- `T-MOBILE_L40S_REPORT.md` + PDF + slidy, 3 demo URL živé.
- Snapshot `w4-final` + rsync `/data` na domácí NAS.
