# Demo 3 – Vytěžování dokumentů (L40S sandbox)

*25. 10. 2026 · T‑Mobile GPUaaS sandbox, 1× NVIDIA L40S 48 GB · vLLM 0.32.0*

## Shrnutí
Jeden L40S s modelem **Qwen3‑VL‑32B‑Instruct‑FP8** vytěží z českých faktur a smluv strukturovaný JSON
s **100 % přesností polí** (150 dpi) při **944 stranách za hodinu**, nebo **1 529 stran/h** s přesností **99,75 %** (110 dpi).
To odpovídá ~23–37 tisícům stran denně na jedné GPU. Skeny (natočení, šum, JPEG) přesnost nesnižují.

## Sada a metoda
- **50 syntetických dokumentů** (50 stran): 35 faktur ve 4 rozvrženích, 15 smluv (o dílo, servisní); 24 z nich jako „sken“
  (natočení ±2,5°, rozmazání, šum, JPEG 35–65, odstíny šedi). Generátor: `demos/03_documents/generate.py`.
- **Pole:** faktura 16 (číslo, VS, 3 data, dodavatel/odběratel název+IČO+DIČ, základ, sazba, DPH, celkem, účet, počet položek),
  smlouva 14 (typ, číslo, strany, datum uzavření, účinnost, doba určitá/konec, cena, výpovědní lhůta, smluvní pokuta). Celkem 770 polí.
- **Pipeline:** PDF → PNG (pymupdf) → Qwen3‑VL přes OpenAI API → výstup vynucený JSON schématem (`response_format: json_schema`) → porovnání s ground truth.
- Jeden dotaz na dokument (typ dokumentu i pole najednou), `temperature 0`.

## Výsledky

| Prompt | DPI | Souběh | Přesnost polí | Dokumenty bez chyby | Stran / h | Latence medián / p95 | Tokeny vstup / strana |
|---|---|---|---|---|---|---|---|
| v1 | 150 | 8 | 99,9 % | 98 % | 799 | 33 s / 41 s | 2 349 |
| v2 | 150 | 8 | **100 %** | **100 %** | 794 | 33 s / 37 s | 2 647 |
| **v2** | **150** | **16** | **100 %** | **100 %** | **944** | 49 s / 79 s | 2 647 |
| v2 | 150 | 32 | 100 % | 100 % | 924 | 94 s / 134 s | 2 647 |
| v2 | 110 | 16 | 99,9 % | 98 % | 1 399 | 34 s / 43 s | 1 622 |
| **v2** | **110** | **32** | **99,75 %** | **96 %** | **1 529** | 56 s / 91 s | 1 622 |

**Doporučená nastavení:** *přesnost* = 150 dpi × souběh 16 (944 stran/h, 100 %); *propustnost* = 110 dpi × souběh 32 (1 529 stran/h, 99,75 %).

### Zbývající chyby (110 dpi)
| Dokument | Pole | Správně | Model |
|---|---|---|---|
| ctr_045 (sken) | strana2_nazev | Jelínek Trade k.s. | Jelinek Trade k.s. |
| inv_001 | duzp | 2026‑01‑20 | 2026‑02‑20 |

## Co jsme se naučili
1. **Prompt v2** (explicitní pravidla, jak poznat dodavatele vs. odběratele) odstranil poslední chybu ve 150 dpi; stojí +300 tokenů na stranu.
2. **Souběh:** při 150 dpi je strop ~16 souběžných požadavků; dál se jen prodlužuje fronta (KV cache 30k tokenů ≈ 11 stran naráz).
3. **Rozlišení je hlavní páka výkonu:** 110 dpi = −39 % vstupních tokenů, +48–62 % propustnosti, za cenu ojedinělé ztráty diakritiky.
4. První běh (95,3 %) odhalil chybu generátoru – minimalistická faktura neměla dodavatele vůbec; model v tom případě správně vracel `null`.

## Omezení
- Data jsou syntetická a jednostránková; reálné faktury (víc stran, tabulky přes stránky, ručně psané poznámky) čekají na vzorek.
- **Cena:** 58 Kč / 1000 stran (150 dpi × 16) a 36 Kč / 1000 stran (110 dpi × 32) při plném vytížení. Počítáme s předpokládaným nákladem asi 55 Kč za hodinu za celou VM (GPU, 8 vCPU, 23 GB RAM, disky). S větší RAM nebo diskem cena roste.
- vLLM `--gpu-memory-utilization 0.93`; zvýšení na 0.95 by dalo ~+30 % KV cache a nejspíš vyšší strop souběhu.

## Další kroky
- Vzorek reálných (anonymizovaných) dokumentů od T‑Mobile; vícestránkové faktury.
- Ladění vLLM (KV cache, `--max-num-seqs`), případně Qwen3‑VL‑30B‑A3B (MoE) pro vyšší propustnost.

*Reprodukce:* `~/tmobile-sandbox/demos/03_documents/run.sh 16` (profil C). Data a reporty: `/srv/work/demo3/`.
