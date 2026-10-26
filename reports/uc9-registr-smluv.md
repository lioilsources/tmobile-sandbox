# UC9 – Registr smluv: dávková LLM extrakce půl milionu smluv (L40S)

*26. 10. 2026 · T‑Mobile GPUaaS sandbox, 1× NVIDIA L40S 48 GB · vLLM 0.32.0 · Qwen3‑30B‑A3B‑Instruct‑2507‑FP8*

## Shrnutí
Jedna L40S za **3 h 30 min** zpracovala všech **568 766 platných smluv** z Registru smluv za duben–září 2026.
Ke každé určila typ dokumentu, druh plnění, odvětví, IČO dodavatele a míru souvislosti s IT a připsala krátký popis.
Propustnost byla **162 800 smluv/h** (45 smluv/s), bez jediné chyby nebo nevalidního výstupu. Náklad VM ≈ **192 Kč**,
tedy **0,34 Kč za 1 000 smluv**. Stejný objem tokenů by přes Claude Haiku 5.5 stál ≈ 850–1 700 Kč a přes Sonnet 5.5
≈ 17–34 tisíc Kč. Výsledek je SQLite databáze `leads.sqlite` pro segmentaci zadavatelů a dodavatelů.

## Data a metoda
- **Vstup:** měsíční XML dumpy `data.smlouvy.gov.cz` (ISRS 1.2), duben–září 2026: 585 999 záznamů, po vyřazení
  neplatných a starších verzí 568 766 smluv (`uc9/parse.py`). Veřejná otevřená data, žádná data T‑Mobile.
- **Do modelu jdou jen metadata:** zveřejňující subjekt, protistrany (název, IČO, příznak příjemce platby), předmět,
  hodnota, názvy příloh. Plné texty příloh (PDF) nestahujeme.
- **Výstup:** `typ` (8 hodnot), `plneni` (7), `odvetvi` (20), `dodavatel_ico`, `it_relevance` 0–3, `popis` (≤ 12 slov).
- **Prompt:** dlouhá statická instrukce první, proměnná část na konci, takže ji prefix caching počítá jednou.
- **Model:** Qwen3‑30B‑A3B (MoE, 3B aktivních parametrů) ve vLLM, profil L: `max-num-seqs 512`,
  `max-num-batched-tokens 16384`, `gpu-memory-utilization 0.90`, FP8 KV cache, souběh klienta 768.

## Ladění propustnosti (pilot 2–6 tisíc smluv)

| Nastavení | Smluv/s | Výstup tok/s | GPU util | Poznámka |
|---|---|---|---|---|
| JSON schéma u každého dotazu, 256 sekvencí | 20,0 | 1 832 | 41 % | xgrammar brzdí na 8 vCPU |
| volný JSON (`json_object`), 256 sekvencí | 33,3 | 3 063 | 71–83 % | 3 % odpovědí mimo výčet |
| **JSON + oprava schématem, 512 sekvencí** | **43,5** | **4 156** | 73 % | zvoleno |
| 768 sekvencí | 38,6 | 3 474 | 51 % | preempce, 7 % požadavků selhalo |
| 512 sekvencí, 32k tokenů na krok | — | — | — | OOM v aktivacích MoE |

**Oprava schématem:** odpověď se nejdřív zkontroluje proti výčtům, nevalidní (2,4 % v ostrém běhu) se pošle znovu
s vynuceným JSON schématem. Výsledek: 0 nevalidních výstupů při rychlosti volného generování.

## Ostrý běh

| Metrika | Hodnota |
|---|---|
| Smluv | 568 766 (0 chyb, 0 nevalidních) |
| Doba | 12 576 s = 3 h 30 min |
| Propustnost | **162 800 smluv/h** (45,2/s) |
| Tokeny vstup / výstup | 37 400 / 4 300 tok/s; průměr 827 / 95 na smlouvu |
| Prefix cache hit | 86 % |
| Latence p50 / p95 | 17 s / 25 s (fronta 768 požadavků) |
| GPU | průměr 75 %, p95 100 %, 279 W, max 58 °C |
| Energie | 929 Wh = 0,0016 Wh na smlouvu |
| Opraveno schématem | 13 574 (2,4 %) |

## Kvalita: shoda s větším modelem
Bez ručně anotovaných dat jsme na náhodném vzorku 600 smluv nechali stejnou extrakci udělat Qwen3‑32B
(vynucené schéma, bez reasoning) a měřili shodu. Rozhodčí není ground truth, ale ukazuje, kde je extrakce nejistá.

| Pole | Shoda 30B‑A3B ↔ 32B |
|---|---|
| typ dokumentu | 89 % |
| IČO dodavatele | 93 % (99,7 % IČO je skutečně mezi stranami smlouvy) |
| IT ano/ne (relevance ≥ 2) | 96 % |
| druh plnění | 76 % |
| odvětví (20 tříd) | 74 % |
| relevance IT 0–3 | 76 % |

Nejčastější neshoda v odvětví je „konkrétní obor ↔ jiné“ (zdravotnictví, IT, kancelář). 30B‑A3B značí IT
velkoryseji: z 48 smluv, které označil jako IT, jich 32B uznal 26 (přesnost 54 %, záchyt 90 %). Pro leady proto
filtrujeme `it_relevance = 3` a seznam se má před oslovením projít.

## Co databáze ukazuje (duben–září 2026)

| Odvětví | Smluv | Hodnota (mld. Kč, kde uvedena) |
|---|---|---|
| zdravotnictví a léčiva | 196 232 | 106,9 |
| stavebnictví a údržba | 104 333 | 450,5 |
| kultura, sport a volný čas | 33 290 | 11,4 |
| IT a software | 32 625 | 41,9 |
| kancelář a spotřební materiál | 27 416 | 9,8 |
| doprava a vozidla | 21 673 | 758,9 |
| telekomunikace | 5 756 | 3,9 |

Typy: objednávky 274 340, smlouvy 185 430, dodatky 50 316, dotace a dary 40 219.
IT relevance: 3 (vývoj SW, data, AI, kyberbezpečnost, cloud) = **9 118 smluv za 29,8 mld. Kč**;
2 (HW, licence, sítě, servis) = 44 860 smluv za 74,7 mld. Kč.

**Zadavatelé s nejvíc smlouvami na vývoj SW / data / AI (it = 3):** ČVUT (208), Ministerstvo vnitra (171),
Grantová agentura ČR (165), VUT v Brně (162), Univerzita Karlova (130), hl. m. Praha (127), Masarykova univerzita
(123), Dopravní podnik hl. m. Prahy (113), Správa železnic (112), Česká televize (107).
**Podle objemu:** MPO (2,7 mld. Kč), TA ČR (1,9), GA ČR (1,5), MV (1,4), Digitální a informační agentura (1,2).
**Nejčastější IT dodavatelé:** CENDIS (80 smluv u 72 zadavatelů), Aricoma Systems, Seyfor, GORDIC, Asseco.

## Cena proti API
Stejný objem tokenů (470 M vstupních, 54 M výstupních) přes Claude API, kurz 23 Kč/USD:

| Varianta | Cena |
|---|---|
| **L40S (náklad VM 3,5 h)** | **≈ 192 Kč** |
| Claude Haiku 5.5 ($0,10 / $0,50 za M), Batch API (−50 %) | ≈ 850 Kč |
| Claude Haiku 5.5, standardně | ≈ 1 700 Kč |
| Claude Sonnet 5.5 ($2 / $10 za M), Batch API | ≈ 17 000 Kč |
| Claude Sonnet 5.5, standardně | ≈ 34 000 Kč |

Prompt caching by u API vstup ještě zlevnil. Kvalitu Claude jsme na téhle úloze neměřili; srovnání je jen cenové.

## Co jsme se naučili
1. **Vynucené schéma u každého dotazu stojí 40 % výkonu.** Kontrola po vygenerování a opakování jen u 2,4 % dotazů
   dá stejnou validitu za plnou rychlost.
2. **Více souběžných sekvencí pomáhá do 512.** Při 768 se KV cache nevejde, vLLM preemptuje a klientovi padají spojení.
3. **MoE 30B‑A3B je pro dávky správná volba:** 160 tisíc smluv za hodinu na jedné GPU. (32B jako rozhodčí běžel jen se souběhem 64, 14 tisíc/h; naplno jsme ho neměřili.)
4. **GPU nebyla na 100 % celou dobu** (průměr 75 %): úzké hrdlo je CPU (8 vCPU dělí tokenizaci, plánování a klienta).

## Omezení
- Jen metadata, ne plné texty smluv; předmět bývá krátký („objednávka spotřebního materiálu“).
- Kvalita měřena shodou s větším modelem, ne proti ruční anotaci.
- Hodnoty smluv jsou, jak je zadavatelé vyplnili, včetně extrémů; částky v Kč bez DPH (nebo s DPH ÷ 1,21).
- Předpokládaný náklad VM 55 Kč/h za celý systém.

## Další kroky
- Ruční kontrola 200 smluv pro skutečnou přesnost; podle ní upravit definici `it_relevance`.
- Plné texty u vybraného segmentu (it = 3) přes Qwen3‑VL (Demo 3 pipeline) pro rozpočty a termíny.
- Měsíční přírůstek (~100 tisíc smluv) je na L40S práce na 40 minut.

*Reprodukce:* `uc9/parse.py` → `uc9/run_full.sh` (profil L) → `uc9/post_full.sh` (rozhodčí + `build_db.py`).
Data: `/srv/work/uc9/` (`contracts.jsonl`, `runs/full/extracted.jsonl`, `leads.sqlite`, `summary.json`).
