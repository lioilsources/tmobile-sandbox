# Demo 1 – Analýza hovorů call centra (L40S sandbox)

*25. 10. 2026 · T‑Mobile GPUaaS sandbox, 1× NVIDIA L40S 48 GB · vLLM 0.32.0 · speaches (faster‑whisper)*

## Shrnutí
Jeden L40S přepíše a vyhodnotí **~1 100–1 200 hovorů za hodinu** (25–27 hodin audia za hodinu GPU) v telefonní kvalitě.
Přepis češtiny má **WER 7,9 %** (operátor) a **12,0 %** (zákazník). LLM analýza trefí **kategorii hovoru na 100 %**,
**ověření identity na 98 %**, riziko odchodu na 90 % a průměrně **88 %** všech šesti štítků.
Rozpoznávání řeči ubírá jen ~2 p. b. proti analýze čistého textu (90 %). Úzkým místem propustnosti je Whisper, LLM analýza trvá 1,9 s na hovor.

## Pipeline (vše na jednom stroji, profil B)
1. **Scénáře:** Qwen3‑32B píše hovory se zadanými vlastnostmi (téma, nálada, vyřešeno, riziko odchodu, ověření identity, další krok).
2. **Validace:** rozhodčí Qwen3‑32B (s přemýšlením) znovu ohodnotí čistý text; ponechány jen hovory, kde souhlasí všech 6 štítků (50 ze 150).
3. **Hlas:** Piper (AiStack audio‑tts, CPU, ~15× rychleji než reálný čas) – operátorka `kasandra`, zákazník `jirka`.
4. **Telefonní kvalita:** stereo (L = operátor, R = zákazník), 8 kHz, pásmo 300–3400 Hz, G.711 μ‑law, šum linky.
5. **Přepis:** Whisper large‑v3‑turbo po kanálech s VAD filtrem, dialog složen podle časových značek.
6. **Analýza:** Qwen3‑30B‑A3B‑Instruct‑2507‑FP8, výstup vynucený JSON schématem + shrnutí do CRM.

Sada: 50 hovorů, 67,8 min audia (57–112 s, průměr 81 s), 8 témat.

## Výsledky (souběh 8)

| Metrika | Hodnota |
|---|---|
| WER operátor / zákazník | **7,9 % / 12,0 %** |
| Kategorie hovoru | **100 %** |
| Ověření identity (compliance) | **98 %** |
| Riziko odchodu | 90 % |
| Vyřešeno v hovoru | 88 % |
| Další krok | 82 % |
| Nálada zákazníka na konci | 72 % |
| **Průměr 6 štítků** | **88,3 %** (na čistém textu bez Whisperu 90,3 %) |
| Propustnost | 25,2× reálný čas = 1 114 hovorů/h (souběh 16: 27,5× = 1 216 hovorů/h) |
| Latence analýzy LLM | 1,9 s na hovor |

Kapacita jedné GPU: ~600 hodin audia / ~27 000 hovorů denně.

## Jak jsme se k tomu dostali

| Krok | Průměr štítků | WER op. / zák. | Co se změnilo |
|---|---|---|---|
| v1 | 80,0 % | 10,7 / 17,9 % | scénáře psal 30B‑A3B, bez validace – zadání často nedodržel |
| v3 | 84,3 % | 24,3 / 20,0 % | scénáře Qwen3‑32B + validace + jednotné definice štítků; Whisper v tichu kanálu halucinoval |
| **finál** | **88,3 %** | **7,9 / 12,0 %** | VAD filtr ve Whisperu; přesná definice „ověření identity“ (70 → 98 %) |

**Klíčová zjištění**
- **Stereo nahrávky + VAD jsou nutnost.** Bez VAD Whisper v tichu druhého kanálu opakuje fráze; tři hovory měly WER > 100 %.
- **Přesnost analýzy určuje definice štítků víc než model.** Upřesnění „ověření identity = operátorka výslovně požádá o údaj a zákazník ho sdělí“ zvedlo přesnost ze 70 na 98 %.
- **Nálada a další krok jsou ze své podstaty nejednoznačné:** na 90 ze 150 vygenerovaných hovorů se neshodly dva běhy Qwen3‑32B. Pro produkci je potřeba společně s T‑Mobile definovat taxonomii a anotovat referenční vzorek.
- Zbývající chyby: neutrální konec hodnocen jako negativní (8×), „žádný další krok“ vs. „zaslání dokumentu“ (5×).

## Ukázka (call_048, přepis Whisperem včetně chyb rozpoznávání)
> **Operátor:** Dobrý den, tuto hovorovou linku obsluhuje Kateřina… Mohu vám prosím ověřit identitu? Mohu vás odkázat na datm narození…
> **Zákazník:** Datum narození je 25. června 1987. Chci ukončit smouvu. A já zvažuju, že se přesunu ke konkurenci…
> **Operátor:** … někdo z naší administrativy vás zavolá do pracovního dne.
> **Zákazník:** To zní stejně jako všichni znovu stejný klasický trik.

Analýza (vše správně): `vypoved_smlouvy · negativni · nevyřešeno · riziko odchodu · identita ověřena · zpětné volání`
*„Zákazník chce ukončit smlouvu kvůli opakujícím se problémům se signálem a chybám ve fakturaci. Vyjádřil zájem o přesun ke konkurenci. Operátorka slíbila zpětné volání.“*

## Omezení
- Syntetická data: hlas Piper je čistší než skutečný člověk (dialekty, překřikování, hluk); skutečné WER bude vyšší. Další krok = anonymizovaný vzorek reálných nahrávek.
- Hlas `jirka` má neověřenou licenci – použit jen pro interní test.
- **Cena:** 45–49 Kč / 1000 hovorů (≈ 2,20 Kč za hodinu audia) při plném vytížení. Počítáme s předpokládaným nákladem asi 55 Kč za hodinu za celou VM (GPU, 8 vCPU, 23 GB RAM, disky). S větší RAM nebo diskem cena roste.

## Další kroky
- Whisper je úzké hrdlo: batched inference nebo víc instancí (VRAM je volná) by měly propustnost zvednout 2–3×.
- Diarizace pro mono nahrávky (pokud T‑Mobile stereo nemá).
- Společná taxonomie štítků s T‑Mobile + referenční anotace 100–200 reálných hovorů.

*Reprodukce:* `~/tmobile-sandbox/demos/01_callcenter/run.sh 8` (profil B + `55_tts`). Data: `/srv/work/demo1/`.
