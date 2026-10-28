# UC25 · FinetuneGallery: SFW validátor, „teplota" a taxonomie obsahu

Hotová je celá cesta: taxonomie v jednom YAML, tři hodnotitelé ve stejném formátu, teplota 0–1 jako ensemble tří klasifikátorů, konsenzus, multi-head hlava (ONNX), conformal prahy a v galerii migrace, facety, filtry, teplotní slider a blok nezletilých. Všechno na větvi `l40s-w2/uc25` repa FineTuneGallery, za flagem `FINETUNE_TAXONOMY=1`. GPU byla celou dobu plná (jiný běh držel 45,3 ze 46 GB), takže čísla níž jsou **z CPU** na 545 obrázcích ze sandboxu. Běh na L40S, Qwen3-VL-8B a kalibrační sada čekají ve frontě. Lidské labely zatím nejsou: κ-gate a conformal jsou postavené a otestované na syntetických datech a pseudo-labelech, ale **čekají na 300 labelů od uživatele**.

## Data
545 obrázků vygenerovaných v sandboxu, 50 na stratum: ilustrace pro Storyteller z UC10 (7 stylů, FLUX.1-schnell; na obrázcích jsou často děti, scény jsou bezpečné), oči z UC7 (base + LoRA), objekty z UC8, výstupy UC5 a smoke testů ComfyUI. Uživatelská galerie ani FFHQ se nepoužily. Všechny obrázky jsou bezpečné, takže kalibrovat teplotu na nich nejde, k tomu je kalibrační sada (fronta).

## Propustnost (CPU, 8 sdílených vCPU, load ~12)
| hodnotitel | obr./h | s/obr. |
|---|---|---|
| WD14 swinv2 (ONNX, 5 vláken) | 2 211 | 1,63 |
| Falconsai ViT | 15 411 | 0,23 |
| AdamCodd ViT | 6 585 | 0,55 |
| SigLIP 2 so400m/512 (zero-shot + embedding) | 426 | 8,4 |
| celá brána bez VLM (součet) | ≈ 330 | 10,8 |
| multi-head hlava (ONNX, 1 vlákno) | — | 0,18 ms (batch 1), 0,009 ms/obr. (batch 64) |

Na L40S se zatím neměřilo, GPU byla obsazená. Hlava je proti embeddingu zanedbatelná: limit pod 20 ms platí pro hlavu, celkovou latenci určuje SigLIP enkodér.

## Teplota a „unsure"
- Pásma: 543 safe, 2 suggestive. Neshoda hodnotitelů → unsure u **5/545 (0,9 %)**.
- Ensemble funguje tak, jak má: Falconsai dal 0,96 výřezu oka, AdamCodd 0,60–0,62 objektu a jedné ilustraci. Žádný z těchto případů neskončil jako nsfw, všechny šly do unsure.
- Na úrovni uzlů taxonomie je **77 % odpovědí unsure** (shoda 23 %, jediný zdroj 60 %, konflikt 17 %). WD14 u většiny uzlů nemá co říct, takže se dvěma hodnotiteli konsenzus kliky neušetří. Třetí hodnotitel (VLM) je nutný.

Shoda WD14 × SigLIP tam, kde odpověděli oba:

| uzel | n | shoda |
|---|---|---|
| věková skupina | 107 | 0,91 |
| zakrytí oblečením | 182 | 0,90 |
| záběr | 209 | 0,79 |
| námět | 544 | 0,71 |
| médium | 144 | 0,69 |
| prezentace pohlaví | 360 | 0,61 |
| délka vlasů | 168 | 0,55 |
| oči | 164 | 0,48 |
| barva vlasů | 254 | 0,44 |
| typ oblečení | 360 | 0,30 |
| výraz | 141 | 0,13 |
| účes | 360 | 0,03 |

Zero-shot SigLIP pro jemné uzly (účes, výraz) nestačí.

## Blok nezletilých (měřeno na bezpečných ilustracích)
- Signál dítěte (WD14 tag nebo věk child/teen s conf ≥ 0,3 od kteréhokoli hodnotitele) měl 239 z 350 ilustrací pro Storyteller a 12 ze 100 výřezů očí (SigLIP).
- **Tvrdý blok: 1 ilustrace (0,29 %).** Jde o falešně pozitivní případ: AdamCodd dal 0,60, ostatní dva hodnotitelé „safe", ensemble 0,23 = suggestive + unsure. Pravidlo tím chybuje schválně na bezpečnou stranu, blok je ale nevratný.
- Kdyby hodnotil jen WD14 (hned po captionu, ještě bez ensemblu), bylo by 10 ilustrací v karanténě. Karanténa je vratná: uvolní ji až jistý safe od ensemblu.
- Pravidlo je otestované jen na syntetických vektorech labelů (16 případů, sdílené mezi Go a Pythonem), ne na obrázcích.

## Hlava a prahy
- Hlava: trénink 8,5 s (436 obrázků, CPU), ONNX 2,6 MB, MAE teploty 0,026. Průměrné F1 vůči konsenzu na held-out je 0,81 (8 uzlů s n ≥ 20). F1 proti VLM zatím není, VLM neběžel.
- Conformal na syntetice (3 000 kalibračních + 20 000 testovacích obrázků): recall explicit 0,994 (cíl 0,99), precision safe 0,988 (cíl 0,98), unsure 2,8 %. Potřeba je ≥ 299 lidských „explicit" (PAC verze guardu). Fit na pseudo-labelech ze sandboxu je **neplatný** (0 explicit), server takový fit odmítne nainstalovat. Platí ruční defaulty 0,20 / 0,45 / 0,70.

## Ve frontě (spouští Director)
1. `uc25/calib_run.sh`: kalibrační sada, profil D (ComfyUI). 882 obrázků ze 6 checkpointů (Pony V6, Illustrious, NoobAI, Animagine 4, Juggernaut XL 9, FLUX.1-dev, ten jen interně), tiery safe / suggestive / nsfw / explicit jen s dospělými, děti jen v bezpečných scénách. Po vygenerování guard smaže každý obrázek, kde se signál nezletilosti potká s čímkoli jiným než jisté safe. Výstupy zůstávají jen na VM, do reportu jdou jen počty. Odhad 1,5–2 h.
2. `PHASES=A uc25/gpu_batch.sh`: WD14, 2× NSFW a SigLIP na GPU přes všech 10 757 obrázků ze sandboxu (+ kalibrační sadu). Potřebuje ≥ 8 GB volné VRAM.
3. `uc25/vllm_vl8b.sh up 0.80` (vlastní vLLM s Qwen3-VL-8B na :8004, guided JSON) → `PHASES=B`: VLM na zhruba 3 000 obrázcích.
4. `PHASES=C`: konsenzus tří hodnotitelů, hlava, F1 hlavy vs VLM, F1 vs zamýšlené atributy kalibrační sady, výsledek do `UC25_gpu.json`.

## Čeká na uživatele
300 labelů: `finetune-gallery taxobench -sample 300 -borderline 60` a pak filtr „label queue" (labeluje se naslepo). Teprve potom κ per hodnotitel per uzel, re-fit prahů a přetrénování hlavy. Rozhodnout je potřeba i to, jestli „teen" blokovat stejně jako „child" (je to přísnější než plán) a jestli blok nezletilých má běžet i bez flagu (teď běží).
