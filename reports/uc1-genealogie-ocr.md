# UC1 – Přepis matrik (genealogie OCR) na L40S

*26. 10. 2026 · T‑Mobile GPUaaS sandbox, 1× NVIDIA L40S 48 GB · vLLM 0.32.0 · Unsloth 2026.10.8*

## Shrnutí
Čtyři hotové vision‑language modely přepisují rukopisné matriky (19. století, kurent i latina) do strukturovaného JSONu
nejvýš s **35 % přesně trefených polí** (validační sada). Po **QLoRA doladění na 180 půlstranách vlastních přepisů**
(11–17 minut tréninku na téže L40S, ≈ 15 Kč) vyskočí **Qwen2.5‑VL‑7B z 24 % na 67 %** a Qwen3‑VL‑8B z 35 % na 65 %
přesných polí. Doladěný 7B model běží **2 133 půlstran/h** (≈ 26 Kč / 1000 půlstran), tedy 5× rychleji a 5× levněji
než největší hotový model Qwen3‑VL‑32B, který je přitom o polovinu méně přesný.

**Doporučení:** Qwen2.5‑VL‑7B + vlastní LoRA, výstup omezený JSON schématem + `presence_penalty 1.5`.

## Data a metoda
- **Sada:** export z `genealogy-matriky` (`finetune/export_pages.py`): 208 půlstran ze dvou veřejných knih SOA Praha
  (Kladno ev. 01 – oddaní, Pchery 12 – narození), cílové přepisy = existující strukturované přepisy projektu
  (polo‑GT podle `PREPIS-KONVENCE.md`, ne ruční ground truth). **Train 180** (Kladno), **val 28** = posledních
  24 folií Kladna (tréninkem neviděné) + **4 půlstrany z Pcher** (kniha, kterou model při tréninku vůbec neviděl).
- **Metriky** (`eval_cer.py` projektu): JSON validní, shoda počtu záznamů, CER neprázdných polí, **přesná shoda pole**
  (hlavní metrika – je to to, co jde do rodokmenu), skupiny polí jména / data / místa / ostatní.
- **Inference:** vLLM, souběh 16, `max_tokens 4096`. Dekódování *fix* = `presence_penalty 1.5` + JSON schéma
  (omezené délky polí a max. 40 řádků). Hladový (greedy) Qwen3‑VL‑8B se zacyklil u 100 z 208 stran
  (JSON validní jen 52 %); *fix* to srazil na 3 smyčky a 98,6 % validních JSONů – proto jedou všechny běhy s *fix*.
- **Fine‑tune:** `uc1/train_qlora.py` (odvozeno z `finetune/train_unsloth.py`): 4bit QLoRA, r=16, jen jazykové vrstvy,
  3 epochy (69 kroků, batch 2×4), lr 2e‑4, loss jen na odpovědi; obrázky v plném rozlišení, max. sekvence 8192.
  Následně merge do bf16 a servírování ve vLLM stejně jako hotové modely.
- Každý běh loguje `nvidia-smi dmon` → `/srv/work/bench/results/UC1_*.json`.

## Výsledky (validační sada, 28 půlstran)

| Model | Přesná pole | Shoda počtu záznamů | CER polí | JSON validní | Půlstran / h | Wh / půlstranu | Kč / 1000 |
|---|---|---|---|---|---|---|---|
| Qwen2.5‑VL‑7B | 24,2 % | 35,7 % | 0,79 | 100 % | 2 837 | 0,10 | 19 |
| Qwen3‑VL‑8B | 34,6 % | 25,0 % | 2,55 | 100 % | 1 678 | 0,16 | 33 |
| Qwen3‑VL‑30B‑A3B (FP8, MoE) | 17,8 % | 57,1 % | 1,81 | 96,4 % | 1 177 | 0,21 | 47 |
| Qwen3‑VL‑32B (FP8) | 30,9 % | 78,6 % | 1,94 | 100 % | 422 | 0,68 | 130 |
| **Qwen2.5‑VL‑7B + LoRA** | **67,4 %** | **85,7 %** | **0,42** | 100 % | **2 133** | 0,13 | **26** |
| Qwen3‑VL‑8B + LoRA | 65,2 % | 82,1 % | 0,49 | 100 % | 1 811 | 0,15 | 30 |

Propustnost a energie jsou z celého běhu (208 půlstran, GPU 93–99 %, 260–300 W). CER > 1 znamená, že model
do polí přidává víc textu, než tam je (halucinované dovětky).

### Neviděná kniha (Pchery, 4 půlstrany – orientační)
| Model | Přesná pole | Shoda počtu záznamů |
|---|---|---|
| Qwen2.5‑VL‑7B | 24 % | 50 % |
| Qwen3‑VL‑8B | 10 % | 50 % |
| Qwen3‑VL‑32B | 22 % | 100 % |
| **Qwen2.5‑VL‑7B + LoRA** | **57 %** | **100 %** |
| Qwen3‑VL‑8B + LoRA | 56 % | 75 % |

### Skupiny polí (všech 208 půlstran, u LoRA včetně tréninkových – jen pro představu, kde jsou chyby)
| Model | Jména | Data | Místa | Ostatní |
|---|---|---|---|---|
| Qwen3‑VL‑32B | 29 % | 36 % | 20 % | 46 % |
| Qwen2.5‑VL‑7B + LoRA | 57 % | 73 % | 65 % | 88 % |

### Trénink
| Základ | Doba (vč. načtení a merge) | Čistý trénink | Loss | Energie | GPU util | VRAM |
|---|---|---|---|---|---|---|
| Qwen2.5‑VL‑7B | 20 min | 17 min | 0,46 → 0,16 | 82 Wh | 59 % | 10 GB |
| Qwen3‑VL‑8B | 14 min | 11 min | 0,56 → 0,18 | 61 Wh | 66 % | 10 GB |

## Ukázka – kniha Pchery, záznam 1 (neviděná kniha)
| Pole | Přepis (polo‑GT) | 7B hotový | 7B + LoRA | 32B hotový |
|---|---|---|---|---|
| datum narození | 27. červen 1840 | Roku 1840 Jun 27 Ev. | **27. červen 1840** | Rok 1840 dne 27. Června |
| jméno dítěte | Maria Ludmila | Maria Ludmila Stuljk. | ✗ (jméno kaplana) | Maria Ludmila Hulzík |
| otec | Josef Kutík(?), šafář v Pcherách č.4 … | Josef Rutik, mistr … (smyšlený text) | Josef Šilhák … (smyšlený text) | ✗ (kaplan) |

LoRA naučila model **konvenci přepisu** (formát dat, normalizace, správný počet řádků) – odtud většina zisku.
U neznámé ruky ale pořád **zaměňuje sloupce a vymýšlí jména** – čtení samotného písma se z 180 stran nenaučí.

## Co jsme se naučili
1. **Doladění porazí velikost.** 7B + LoRA za 15 Kč tréninku je 2× přesnější a 5× levnější než 32B. Pro matriky nemá
   smysl platit za větší hotový model – smysl má sbírat přepisy.
2. **Dekódování je povinné.** Bez `presence_penalty` a schématu se VLM na tabulkových rukopisech zacyklí
   (opakuje řádky do 4096 tokenů) – půlka stran je pak k ničemu a propustnost padne na třetinu.
3. **MoE (30B‑A3B) tu selhává** – nejhorší přesnost ze všech, i když je rychlejší než 32B.
4. **Trénink GPU nevytíží** (59–66 %, 10 GB VRAM) – batch 2 + gradient checkpointing; větší batch nebo trénink
   souběžně s inferencí se na L40S vejde.
5. Unsloth tiše zkracoval sekvence na 2048 tokenů a zmenšoval obrázky na 512 px – původní skript projektu by
   uřízl cílové JSONy. Opraveno v `uc1/train_qlora.py` (zpětně použitelné pro Spark).

## Omezení
- Validace je malá (28 půlstran, z toho 4 z neviděné knihy) a z jediné kombinace písař/typ knihy pro train.
  Čísla berte jako směr, ne jako přesnou hodnotu.
- Cíle jsou přepisy projektu (polo‑GT), ne nezávislá ruční kontrola; část „chyb“ mohou být chyby přepisu.
- Specializované OCR modely (dots.ocr, olmOCR‑2) a Kraken nebyly změřeny – nevracejí strukturu podle schématu;
  dávají smysl jen jako první krok dvoustupňové pipeline (OCR → textový LLM), což při 67 % LoRA nemá prioritu.
- Část plánu *record‑linkage + OpenEvolve* neběžela: v repu zatím chybí `match_score.go` a označené páry
  (`TestMatchBench`).
- **Cena:** předpokládaný náklad VM ≈ 55 Kč/h za celý systém; Kč/1000 půlstran = 55 Kč/h ÷ propustnost.

## Další kroky
1. Víc tréninkových dat z jiných knih a písařů (aktivní učení podle `finetune/how-to.md`: žák přepíše, špatné strany
   přepíše učitel) – největší očekávaný zisk je u neviděných rukou.
2. Druhé kolo LoRA s rank 32 / 5 epoch a větším batchem; vyhodnocení na větší val (≥ 100 půlstran z ≥ 3 knih).
3. PR do `genealogy-matriky`: `train_qlora.py` opravy + flag pro endpoint s LoRA modelem.

*Reprodukce:* `uc1/bench_engine.sh <label> <hf_model|profil:C> 16 "" fix`, fine‑tune `uc1/ft_chain.sh <label> <hf_base> 3`.
Data: `/srv/work/uc1/` (runs, dataset), modely `/srv/hf/ft/*-merged`, výsledky `/srv/work/bench/results/UC1_*.json`.
