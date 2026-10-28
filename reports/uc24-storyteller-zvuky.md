# UC24 · Storyteller: zvuky, které nešumí, nechrčí a odpovídají popisu

Brána kvality pro generované SFX, smyčky a hudbu. Skládá se z pěti kroků:

1. signálové příznaky na CPU (`sfxgate`),
2. soudci (CLAP, PANNs, Audiobox Aesthetics, Qwen2-Audio),
3. učený gate s konformním prahem,
4. smyčka, která podle důvodů zamítnutí přepíše prompt lokálním LLM,
5. oprava hraničních klipů ffmpeg + noisereduce.

Do Storytelleru je zapojená za flagem, do audio služby AiStacku jako facet. **Všechno je změřené jen na syntetických datech se známou vadou.** Skutečné klipy z MOSS ani ze Stable Audio Open se na L40S nevygenerovaly, protože generátor na VM neběží a GPU drží UC11. Olinových 150 hodnocení zatím neexistuje.

**Hlavní čísla (test část syntetické sady, 1 035 klipů):**
- `sfxgate` stojí 4 ms na 3s SFX na Macu M2 a 7 ms na VM. Celou sadu prošel za 20 ms na klip (medián; VM v zátěži, klipy 3–8 s).
- Na SFX zachytí 100 % vložených vad kromě mírného ticha na začátku (92 %). Falešné poplachy na čistých klipech: 14 %.
- Učený gate ONNX (jen signál) drží recall 0,958 na nepříjemné vady a falešné poplachy jsou 11,6 %. Ruční prahy mají recall 0,923 a falešné poplachy 14,5 %.
- Smyčka se simulovaným generátorem přijme do 3 iterací 87,9 % úloh s přepisem promptu. Kontrola jen s novým seedem: 76,3 %.

## Data (jen vlastní)
| sada | obsah | k čemu |
|---|---|---|
| `uc24/synth.py` | 19 procedurálních zdrojů: 12 SFX (zvon, klepání, kroky, pták, sova, kapky, švih, mince, vrzání, zvonkohra, štěkot, mňoukání) a 7 ambientů (vítr, déšť, potok, oheň, les, cvrčci, vlny), 6 seedů. K tomu 24 řezů 8 s z 6 ACE-Step klipů z UC5. Celkem **138 výchozích klipů** | čisté reference |
| vady | 7 typů × 2 síly na každý výchozí klip: rumble (30–90 Hz, −6/+4 dB vůči RMS), clipping (3×/8×, polovina pak ztišena), hiss (SNR 30/15 dB), mlha nad 8 kHz (25/12 dB), kliky (2/8 za s), DC (0,02/0,05), ticho na začátku (0,5/1,5 s). Celkem **2 070 klipů** | známá pravda |
| mismatch | čistý klip × popis jiného zdroje stejného druhu, **114 párů** | shoda s popisem |
| rozdělení | podle výchozího klipu: dev = seed 0–2 (ACE 1–3), test = seed 3–5 (ACE 4–6) | prahy se ladí na dev, čísla jsou z testu |

Veřejné váhy (licence): PANNs Cnn14 (Zenodo 3987831, CC BY 4.0), Audiobox Aesthetics (CC BY 4.0), CLAP (LAION, Apache-2.0), Qwen2-Audio-7B-Instruct (Apache-2.0). **NISQA nepoužito.** Váhy jsou pod CC BY-NC-SA 4.0 (nekomerční) a model je řečový, takže pro SFX je mimo doménu. BEATs nepoužit: oficiální checkpoint se distribuuje mimo HF (odkaz z repa microsoft/unilm) a nestahoval jsem ho. Úlohu třídní kontroly převzaly PANNs.

## Krok 1 — `sfxgate` (Python, CPU)
**Příznaky:**
- poměr energie pod 250 Hz a pod 80 Hz,
- true-peak (4× FFT okolo maxim), LUFS (BS.1770-4, shoda s pyloudnorm ±0,05 LU), crest factor,
- clipping (plná škála i "plochý vrchol" po ztišení), DC offset,
- spektrální plochost a stacionární podlaha ve 4 pásmech, noise floor (p5),
- ticho na začátku a na konci, kliky (2. diference proti lokálnímu RMS).

**Simulace telefonu** (fallback, dokud není `phone_ir.wav`):
- peak na −1 dBTP,
- měkký clip nad 0,8 FS (limiter),
- mez výchylky basové složky (LP 250 Hz, Xe = 0,12),
- HP 300 Hz 4. řádu a LP 12 kHz.

Ze simulace vychází zkreslení (`phone_dist_db`) a ztráta hlasitosti (`phone_drop_db`). Xe a koleno limiteru jsou odhad, ne měření.

**Výstup:** JSON `{pass, verdict: pass|repair|reject, score, reasons[{code, metric, value, threshold, stage, severity, message}], features, ms}` a HTTP sidecar `python -m sfxgate.server --port 8096`. 13 unit testů prochází, jeden z nich drží κ tabulky z `kappa_test.go`.

| ms/klip, 1 vlákno | 3 s SFX | 8 s ambient | 30 s hudba |
|---|---|---|---|
| Mac M2 | 4,0–4,2 | 10,5 | 41 |
| L40S VM (Xeon 6548Y+, load 10–17) | 6,6–7,1 (p95 35) | 19,9 | 114 |

Cíl < 10 ms na klip platí pro SFX. Čas roste lineárně s délkou, asi 1,3 ms na sekundu zvuku na M2. První verze (pyloudnorm a převzorkování po špičkách) měla na VM 33 ms na 3s klip.

**Záchyt vad na test části** (podíl klipů s verdiktem ≠ pass, mild / strong). Podíl se správným kódem důvodu je stejný, kromě mírných vad v ambientu a hudbě, kde je o 4–14 p. b. nižší:

| vada (mild / strong) | SFX (n = 36) | ambient (n = 21) | hudba ACE (n = 12) |
|---|---|---|---|
| rumble | 1,00 / 1,00 | 1,00 / 1,00 | 0,25 / 1,00 |
| clipping | 1,00 / 1,00 | 1,00 / 1,00 | 1,00 / 1,00 |
| hiss | 1,00 / 1,00 | 0,57 / 1,00 | 0,17 / 0,58 |
| mlha > 8 kHz | 1,00 / 1,00 | 0,86 / 1,00 | 0,25 / 1,00 |
| kliky | 1,00 / 1,00 | 0,76 / 0,86 | 1,00 / 1,00 |
| DC offset | 1,00 / 1,00 | 1,00 / 1,00 | 1,00 / 1,00 |
| ticho na začátku | 0,92 / 1,00 | 1,00 / 1,00 | 1,00 / 1,00 |
| **falešný poplach na čistých** | **0,14** | **0,14** | **0,17** |

**Kde ruční prahy narážejí:**
- **Vrzání.** Syntetické vrzání (sled impulzů) je samo o sobě "klikavé", proto falešné poplachy u SFX.
- **Oheň.** Praskání ohně v ambientu vypadá jako kliky.
- **Šum v hudbě.** Mírný šum v hutné hudbě se signálově neodliší od obsahu.
- **Rumble v hudbě.** ACE-Step chiptune má 49–72 % energie pod 250 Hz, takže mírný rumble zapadne. Proto má hudba volný práh 0,80, a mírný rumble v hudbě proto chytí jen ve 25 %.

## Krok 2 — soudci
GPU mělo volných 0,8 GB, proto vše běželo na **CPU** VM: 4 vlákna, nice 19, souběžně s UC11, UC15 a UC25. Propustnost na L40S GPU **neměřena**.

| soudce | klipů/h (CPU) | sada |
|---|---|---|
| PANNs Cnn14 | 9 594 | celá (2 070) |
| CLAP clap-htsat-unfused | 4 650 | celá |
| CLAP larger_clap_general | 2 277 | test + čisté dev (1 104) |
| Audiobox Aesthetics | 1 257 | test seed 3 / ACE 4 + čisté dev (460) |

**Vady, AUC čistý vs. vadný (test).** Detekce při prahu z dev (p90 čistých) je uvedena za lomítkem.

| | SFX rumble | SFX hiss | SFX kliky | SFX clipping | ambient hiss | hudba rumble |
|---|---|---|---|---|---|---|
| sfxgate (verdikt) | 0,95 / 1,00 | 0,93 / 1,00 | 0,96 / 1,00 | 0,98 / 1,00 | 0,78 / 0,79 | 0,77 / 0,63 |
| Audiobox PQ | 0,95 / 0,88 | 0,96 / 0,79 | 0,64 / 0,17 | 0,62 / 0,17 | 0,57 / 0,07 | 0,81 / 0,38 |
| CLAP-unfused sonda "rumble" | **1,00 / 0,99** | 0,80 | 0,52 | 0,20 | — | 0,89 / 0,83 |
| CLAP-general sonda "hiss" | 0,83 | **0,99 / 0,99** | 0,91 | 0,61 | 0,65 / 0,10 | — |
| PANNs suma šumových tříd | 0,68 | 0,82 | 0,53 | 0,48 | 0,59 | 0,65 |

CLAP sondy jsou zero-shot texty: "a recording with loud background hiss…", "…low frequency rumble…" atd., od skóre se odečítá "a clean high quality recording".

**Co z toho plyne:**
- Na SFX umí PQ a CLAP sondy najít rumble a hiss skoro stejně dobře jako signál.
- Kliky a clipping vidí jen signál.
- Na ambientu selhávají všichni neuronoví soudci (AUC 0,5–0,7). Je to stejná slepá skvrna jako u ručních prahů.
- Osa Audiobox PC (complexity) jde proti vadám (AUC 0,05–0,47), jako soudce kvality je nepoužitelná.

**Shoda s popisem** (vlastní popis vs. cizí popis, test):

| | AUC SFX | přesnost SFX | AUC ambient | přesnost ambient |
|---|---|---|---|---|
| CLAP clap-htsat-unfused | 0,83 | 0,79 | 0,84 | 0,81 |
| CLAP larger_clap_general | 0,81 | 0,78 | 0,83 | 0,74 |
| PANNs třídy z promptu (Qwen3-8B mapování) | 0,69 | 0,51 | 0,70 | 0,62 |

Procedurální zvuky jsou pro CLAP těžké, např. syntetický "štěkot" se psovi moc nepodobá. Na skutečných generovaných klipech čekám vyšší čísla, ale **neměřil jsem je**.

**Qwen2-Audio-7B (JSON soudce):** skript je hotový a ověřený na CPU (procesor a chat šablona). Na GPU neběžel, protože volné VRAM bylo 0,8 GB. Je připravený ve frontě, viz níže.

**Mapování prompt → AudioSet třídy přes Qwen3-8B** (`prompt_classes.py`): 18 z 20 popisů dostalo platné třídy. U "fast whoosh" model nevrátil přesný název třídy (AudioSet má "Whoosh, swoosh, swish") a mince dostaly jen "Sound effect".

## κ-gate (čeká na Olina)
- `rating_sheet.py` vyrobí blind list: 150 klipů stratifikovaných podle verdiktu, přejmenovaných na r001…, s `ratings.csv`, přehrávačem `index.html` a klíčem `key.json`. Na hodnocení jsou 3 otázky: nepříjemné? odpovídá? jaká vada?
- `kappa.py` počítá κ stejně jako `FinetuneGallery/server/kappa.go`: Cohen κ, asymptotická SE, rozhoduje dolní mez 95 % intervalu. Konstanty jsou n ≥ 40, ≥ 8 lidských "down", κ ≥ 0,60 a dolní mez ≥ 0,40. Test drží tabulky z `kappa_test.go`.
- Demo list ze syntetiky leží v `/cold/uc24/rating_demo`. Pro skutečnou kalibraci musí být v listu generované klipy (`run_gen_real.sh … judges` je vyrobí).

## Krok 3 — učený gate
LightGBM nad 32 příznaky z kroku 1 (+ CLAP embedding) → p(nepříjemné). Konformní práh je zvolen tak, aby recall na kalibrační části byl ≥ 0,95. Rozdělení: train = seed 0–1, kalibrace = seed 2, test = seed 3–5. Štítky jsou syntetické; "nepříjemné" = rumble, clipping, hiss, mlha nebo kliky.

| | recall (test) | falešné poplachy na čistých | AUC | |
|---|---|---|---|---|
| ruční prahy (krok 1) | 0,923 | 0,145 | — | |
| GBM jen signál → `sfxgate_signal.onnx` (327 kB, ~0,2 ms) | **0,958** | **0,116** | 0,990 | běží jen s onnxruntime → Spark/AiStack |
| GBM signál + CLAP-unfused (544 příznaků) → `sfxgate_signal_clap.onnx` | **0,968** | **0,101** | 0,997 | potřebuje CLAP audio věž (~4 650 klipů/h na CPU) |

GBM odstraní hlavní falešný poplach ručních prahů, vrzání (na čistém vrzání dává p = 0,0003). Mírnou mlhu v dešti ale pustí: p = 0,989 pod prahem 0,9967. **Model se naučil moje syntetické vady, ne to, co vadí posluchači.** Před nasazením ho je nutné přetrénovat na Olinových hodnoceních a pseudo-štítcích ze shody ≥ 2 soudců.

## Krok 4 — feedback smyčka, slovník, OpenEvolve
Smyčka (Python harness `feedback.py` i Go `internal/sfxgate.Loop`) běží takto: generuj → brána → (oprava | přepis LLM) → znovu, nejvýš 3 iterace. Loguje se celá cesta. Systémový prompt přepisovače je v Pythonu i v Go stejný.

Sémantická kontrola v bráně se vždy dělá proti **původnímu** zadání (oprava v commitu `281339a`). Přepis tak nemůže odvést zvuk jinam.

Úlohy: celý katalog Storytelleru, 173 SFX (`rag/audio/catalog.json`). Generátor je **simulovaný** (`UC24_GEN=sim`): procedurální zvuk a vady podle rizikových slov (deep/thunderous → rumble, loud → clipping, vintage/distant → hiss …). Ochranná slova (clean, dry, no rumble …) šanci vady snižují. Tahle čísla měří smyčku a přepisovač, ne kvalitu MOSS ani Stable Audio.

| sim, 173 úloh | přijato napoprvé | + oprava | přijato do 3 iterací | generací | přepisů |
|---|---|---|---|---|---|
| jen nový seed (kontrola) | 0,538 | 0,717 | 0,763 | 266 | 0 |
| přepis promptu (qwen3-8b, profil S) | 0,538 | 0,717 | **0,879** | 247 | 74 |

- Přepisy:
  - obsah zadání zachovalo 72 ze 74 (≥ 70 % obsahových slov),
  - 60 přidalo ochranná slova,
  - 52 odstranilo rizikové slovo nebo žádné nemělo,
  - žádný finální prompt nepřesáhl 60 slov (průměr 19,0 → 21,2),
  - medián latence 15,4 s: vLLM sdílený s gridem UC11, 48 paralelních požadavků.
- Všech 21 zbylých zamítnutí s přepisem má v simu stejnou příčinu: falešný poplach brány na výchozím zvuku (vrzání, klepání), na kterém prompt nic nezmění. Smyčka je správně nepustí a vrátí nejlepší pokus s `accepted=false`. V kontrole jen s novým seedem zůstalo 41 zamítnutí a u 8 z nich byla vada ze slova v promptu.
- **Slovník** (`gen_bulk.py` + `toxicity.py`): faktoriální návrh, katalogové zadání + 0–3 z 30 modifikátorů. L2 logistická regrese s kontrolou zadání a bootstrap 95 % CI, 3 000 sim generací.
  - Nalezlo všech 11 skutečných spouštěčů: booming, loud, huge, vintage, deep, heavy, lo-fi, dark, powerful, thunderous, distant.
  - Navíc 2 falešně toxická slova se slabým efektem (subtle 0,44, studio recording 0,37). Přesnost je 0,85; při |koef.| > 0,7 je 1,0.
  - Ochranná slova: clean, dry (+ magical, což je artefakt simu).
  - Metoda funguje. Skutečný seznam pro MOSS a Stable Audio vyžaduje 5 000 skutečných generací.
- **Statický sken katalogu:** 30 ze 186 promptů obsahuje slova, kterým se pravidla přepisovače vyhýbají. Příklady: "a bear grumbling **deep**", "a dark curse falling, **low rumble**", "**distant** thunder rolling", 10× "**heavy**". Je to hypotéza do slovníkové analýzy, ne měření (`bench/results/UC24_catalog_risk_words.json`).
- **OpenEvolve 0.4** nad šablonou promptu (`evolve/`), 50 fixních katalogových zadání, skóre = projde napoprvé ∧ zachová obsah:
  - Harness funguje. Ruční šablona "clean, dry, no low-end rumble, …" dá v simu 0,54 → 0,66.
  - 8 iterací s qwen3-8b pod zátěží UC11 žádné skutečné zlepšení nenašlo, jen triviální změny (0,54 → 0,58, a to byl šum simu ještě před opravou RNG).
  - Pro skutečný běh potřebuje skutečný generátor a ideálně qwen3-32b.
  - Nález: OpenEvolve neumí poslat `chat_template_kwargs` a `/no_think` v system promptu vLLM s `--reasoning-parser qwen3` nerespektuje (content = None). Řešením je `evolve/nothink_proxy.py`.

## Krok 5 — oprava místo zahození
Řetězec:
1. DC pryč,
2. ffmpeg HP 200 Hz (SFX) / 120 Hz (ambient, hudba) 4. řádu,
3. ořez ticha,
4. noisereduce, jen když je důvod "hiss" (profil z nejtiššího úseku 0,5 s),
5. loudnorm −16 LUFS / −1,5 dBTP a alimiter.

Když po první opravě zůstane opravitelný důvod, proběhne 2. průchod. Příklad: DC "zaplní" ticho na začátku a to se ukáže až po odstranění DC. Klip se přijme jen tehdy, když projde bránou a CLAP (larger_clap_general) neklesne o více než 0,05.

Měřeno na 300 test klipech s verdiktem repair, CPU, s CLAP. Medián je 1,3–1,8 s na klip, z toho většinu tvoří CLAP na CPU.

| vada (mild / strong) | projde bránou po opravě | CLAP neklesl o > 0,05 | přijato |
|---|---|---|---|
| DC offset | 0,98 / 0,96 | 0,86 / 0,93 | 0,84 / 0,89 |
| ticho na začátku | 1,00 / 0,98 | 0,73 / 0,75 | 0,73 / 0,73 |
| rumble | 0,94 / 1,00 | 0,79 / 0,70 | 0,73 / 0,70 |
| hiss | 0,67 / 0,58 | 0,82 / 0,42 | 0,59 / 0,16 |
| mlha > 8 kHz | 0,41 / 0,37 | 0,74 / 0,59 | 0,30 / 0,07 |
| **celkem** | | | **0,61** |

- **Signálové vady oprava spolehlivě odstraní.** DC, ticho a rumble projdou bránou v 94–100 % případů.
- **Noisereduce na stacionární šum pod obsahem nestačí** (37–67 %).
- **Práh CLAP 0,05 je pod šumem CLAPu.** Odříznutí ticha na začátku obsah nemění, a přesto CLAP klesl o víc než 0,05 u 25–27 % klipů. Práh je nutné nastavit na skutečných klipech, ne na procedurálních.

## Integrace
- **Storyteller**, `w2/storyteller`, větev `l40s-w2/uc24` (lokálně, nepushnuto):
  - `cc7ce7a`: `internal/sfxgate` (klient sidecaru, přepisovač s OpenAI-kompatibilním API a `enable_thinking: false`, `Loop` s trace). V `render-audio` přibyly flagy `-gate`/`STORYTELLER_SFXGATE_URL`, `-llm`, `-llm-model`, `-gate-iters`. Trace se ukládá do `<id>.gate.json`. Bez flagu se render nemění.
  - `281339a`: brána porovnává s původním promptem.
  - `18cf654`: `cmd/gate-smoke`, ověřeno proti skutečnému Python sidecaru. Čistá sova projde, sova s rumble je zamítnuta se třemi důvody, sova s tichem na začátku dostane "repair" a po opravě projde.
  - `0c8abbe`: výchozí port 8096.
  - `go test ./internal/... ./gateway/...` prochází.
- **AiStack**, `w2/AiStack`, větev `l40s-w2/uc24`, `97e688f`: `services/audio` posílá při zapnutém `AUDIO_QUALITY_URL` každou hotovou SFX nebo hudební variantu na `/v1/gate`. Výsledek přidá do `outputs[].quality` jako facet a nic nezahazuje. Výpadek brány job neshodí. Brána záměrně hodnotí soubor až po post-procesu: zesílení na −18 LUFS s limiterem je přesně to, co z transientního SFX udělá "prskání". Ověřeno proti skutečnému sidecaru na OGG. Prošlo všech 225 testů (3 nové).
- Port 8095 je obsazený AiStack `audio-sfx`, proto sidecar běží na 8096.

## Queue-ready (spouští Director)
| co | příkaz (na VM, `~/tmobile-sandbox/uc24`) | GPU / profil | odhad |
|---|---|---|---|
| Qwen2-Audio soudce, 400 test klipů | `bash run_qwen2audio.sh 400` | ≥ 21 GB volné VRAM (bf16 16,5 GB + batch 8); bez vLLM s velkým modelem | asi 10–20 min (neměřeno; 400 krátkých klipů, 96 tokenů výstupu) |
| hudba ACE-Step, bulk + soudci + blind list | `UC24_GEN=comfy-ace KIND=music N=2000 bash run_gen_real.sh bulk judges` | profil D (ComfyUI :8188), peak ~27 GB (UC5) | odhad 2 000 × ~3 s ≈ 1,7 h (UC5: 5,3 s na 60s klip) + soudci na GPU |
| smyčka nad katalogem (hudba 26 úloh) | `UC24_GEN=comfy-ace KIND=music bash run_gen_real.sh loop` | ComfyUI a vLLM qwen3-8b (≤ 0,35 GPU util) současně | ~78 generací ≈ 5 min + přepisy |
| SFX (MOSS) bulk 5 000 + smyčka 173 | `UC24_GEN=aistack UC24_AUDIO_URL=http://127.0.0.1:8093 bash run_gen_real.sh bulk loop judges` | AiStack `audio` + `audio-sfx` na VM (image tu není, nutno sestavit) | neměřeno (MOSS na L40S jsem netestoval) |
| OpenEvolve nad skutečnými zvuky | `UC24_GEN=aistack bash run_evolve.sh 40` | generátor + vLLM (ideálně qwen3-32b) | 40 iterací × 50 klipů |

Wh na klip zatím není. Měří ho `run_gen_real.sh`: `bench/gpu_log.sh` a `summarize.py --extra` se zapnou u každé fáze.

## Čeká na uživatele
1. **150 blind hodnocení na telefonu.** List vyrobí `run_gen_real.sh … judges`, κ pak spočítá `kappa.py`. Bez nich není žádný soudce "pass".
2. **`phone_ir.wav`**: sweep ze 3 telefonů, nahraný mikrofonem Macu Mini. Simulace se pak přepne z modelu na IR (`SFXGATE_PHONE_IR`).
3. **Stable Audio Open 1.0** je gated repo s licencí Stability Community (do 1 M USD obratu). Nutné odsouhlasit na HF, pak stáhnout checkpoint a t5_base do ComfyUI. AiStack už jako primární SFX zvolil MOSS (Apache-2.0).
4. Blind A/B poslech 3 lidí před a po (výstup plánu). Zatím nic není, protože chybí skutečné klipy.

## Nálezy a rizika
- **Syntetické vady jsou snadné.** Detekce kolem 100 % neznamená, že brána chytí skutečné artefakty difuzního modelu. Ty je nutné ověřit na generovaných klipech a Olinových hodnoceních.
- **Ambient je pro signálovou bránu nejtěžší.** Déšť a oheň jsou legitimně šum a kliky, takže ruční prahy jsou tam slabé (mírný hiss 57 %). Tohle mají řešit CLAP a učený gate.
- **Model telefonu je odhad.** `phone_dist_db` odliší silný rumble, ale ne mírný. Dokud není IR, jde spíš o pomocný příznak.
- **VM CPU je sdílené.** Load byl 10–17 (UC11, UC15, UC25 a UC24 současně), latence brány na VM má proto dlouhý ocas (p95 35 ms na 3 s). Přepis promptu přes qwen3-8b trval v mediánu 15–17 s, protože vLLM paralelně obsluhoval grid UC11.
- **Moje dávky na sdíleném vLLM.** Smyčky a OpenEvolve poslaly na Directorovo vLLM asi 200 krátkých požadavků (bez GPU jobů). Mohly mírně ovlivnit měření propustnosti UC11.
