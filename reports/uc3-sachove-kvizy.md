# UC3 – Šachové kvízy pro děti: lc0 ohodnotil celou Lichess puzzle databázi (L40S)

*26. 10. 2026 · T‑Mobile GPUaaS sandbox, 1× NVIDIA L40S 48 GB · lc0 v0.32.1 (CUDA fp16) · síť BT4‑1024x15x32h · Qwen3‑32B pro texty*

## Shrnutí
Šachová neuronová síť lc0 (BT4, transformer) na jedné L40S ohodnotila **všech 6 157 341 pozic** Lichess puzzle databáze
za **21 minut**, tedy **17,6 milionu pozic za hodinu**. Plán počítal s 1 milionem za hodinu. GPU jela v průměru na 94 %,
spotřebovala 110 Wh a náklad VM vyšel na ≈ 19 Kč. Z ohodnocených pozic jsme vybrali **1 235 kvízů** pro aplikaci
Chess4Princesses: řešení jedním tahem, právě jeden správný tah, obtížnost 1–5. Ke každému kvízu Qwen3‑32B napsal český příběh a nápovědu.
Mat jedním tahem je hotový ve třech lekcích po 400 kvízech. Vidličky, vazby a braní nechráněné figury tímhle filtrem skoro
neprošly (0, 2 a 33 kvízů). Na ně je potřeba jiný zdroj.

## Postup
1. `uc3/prepare.py`: Lichess puzzle CSV → 6 157 341 pozic. U každé je zahrán soupeřův úvodní tah, takže pozice je přesně ta, kterou řeší hráč.
2. `uc3/analyze.py` v kontejneru `tms/lc0`: přes Python bindings lc0 vyhodnotí síť dávky po 1 024 pozicích. Pro každou
   pozici ukládá Q (očekávaný výsledek), pravděpodobnost remízy, odhad zbývajících tahů, pravděpodobnost řešení podle policy,
   pořadí řešení mezi tahy, počet kandidátů s policy nad 5 % a entropii policy.
3. `uc3/select_quizzes.py`: kandidáti s řešením na jeden tah a nejvýš 16 figurami. Jednoznačnost: u matu jedním tahem
   python-chess spočítá, že matuje právě jeden tah. U ostatních motivů Stockfish v režimu multipv 2 ověří náskok
   aspoň 2 pěšců. Obtížnost 1–5 jsou kvintily skóre z intuice sítě (p_sol), počtu kandidátů a počtu figur.
4. `uc3/texts.py`: Qwen3‑32B (profil A) napíše dvě české věty, příběh a nápovědu, v jazyce aplikace (Princezna = dáma,
   Věžka na kolečkách = věž, drakův král). Nápověda nesmí obsahovat pole ani tah, kontroluje to regulární výraz.

## Benchmark sítí

| Síť | Pozic | Čas | GPU util | Příkon | VRAM |
|---|---|---|---|---|---|
| T1‑256x10 (2 procesy) | 200 000 | 17 s | 32 % | 204 W | 3,5 GB |
| T1‑512x15 (2 procesy) | 200 000 | 28 s | 34 % | 234 W | 3,7 GB |
| BT4‑1024x15, 1 proces | 150 000 | 43 s | 61 % | 240 W | 3,4 GB |
| BT4, 2 procesy | 150 000 | 41 s | 71 % | 259 W | 6,4 GB |
| BT4, 3 procesy | 150 000 | 42 s | 69 % | 259 W | 9,4 GB |
| **BT4, 2 procesy, celá DB** | **6 157 341** | **1 260 s** | **94 %** | **314 W** | 6,4 GB |

Krátké běhy zahrnují start kontejneru a načtení sítě, proto u nich GPU nejede naplno. V ostrém běhu GPU nečinně
nestála ani 1 % času. Dva paralelní procesy stačí: třetí už nepřidá, protože úzkým hrdlem je CPU (příprava vstupů, 8 vCPU).

## Ostrý běh

| Metrika | Hodnota |
|---|---|
| Pozic | 6 157 341 |
| Doba | 21 min (1 260 s) |
| Propustnost | **17,6 M pozic/h** (4 890/s) |
| GPU | průměr 94 %, p95 100 %, 314 W, max 59 °C |
| Energie | 110 Wh |
| Náklad VM | ≈ 19 Kč (předpoklad 55 Kč/h) |

## Co síť říká o puzzle databázi
Tah s nejvyšší policy byl řešením úlohy v **95 %** pozic, bez jakéhokoli prohledávání. Shoda klesá s ratingem úlohy:

| Rating úlohy | Pozic | Top tah = řešení | Průměrná p(řešení) |
|---|---|---|---|
| 250–750 | 504 546 | 98,8 % | 0,92 |
| 1000 | 1 061 352 | 98,8 % | 0,89 |
| 1500 | 872 961 | 96,2 % | 0,82 |
| 2000 | 571 387 | 88,6 % | 0,69 |
| 2500 | 182 814 | 78,6 % | 0,59 |
| 3000+ | 1 868 | 69,1 % | 0,53 |

Pro dětské kvízy je to užitečné i opačně: úlohy, kde síť řešení „vidí“ okamžitě, jsou ty s jasným vzorem.

## Kvízy

| Lekce | Kandidátů | Vybráno |
|---|---|---|
| Mat na poslední řadě | 6 514 | 400 |
| Mat Princeznou (dámou) | 31 886 | 400 |
| Mat v 1 – mix | 67 260 | 400 |
| Sněz nechráněnou figuru | 34 | 33 |
| Vazba | 2 | 2 |
| Vidlička | 0 | 0 |

Každá matová lekce má 80 kvízů na stupeň obtížnosti. Ukázka (`quizzes_cs.json`):

- **Mat na poslední řadě, obtížnost 1:** 8 figur, řešení Rf1#.
  Příběh: „Pomoz Věžce na kolečkách ukázat drakovu králi, že už nemá kam utéct!“
  Nápověda: „Pomoz Věžce najít cestu, která drakovu králi zablokuje všechny úkryty.“
- **Mat Princeznou, obtížnost 3:** 13 figur, řešení Qxg7#.
  Příběh: „Princezna chce ukázat drakovi, že nejkrásnější věc je královský mat!“

## Co nefunguje
- **Vidličky a vazby:** v Lichess DB jsou to skoro vždy úlohy na víc tahů, takže filtr na jeden tah je vyřadí.
  Pro dětské varianty („napadni dvě figury najednou“) je potřeba kvízy generovat z vlastních mini‑pozic, jak plán počítal.
  Tohle jsme zatím neudělali.
- **Obtížnost:** kvintily z intuice sítě skoro nesouvisejí s ratingem Lichess (medián ratingu na stupních 1–5 je 620–765),
  protože u matu jedním tahem je síť skoro vždy jistá. Stupeň zatím určuje hlavně počet figur. Kalibrace potřebuje
  data z hraní dětí.
- **Texty:** 1 085 unikátních příběhů z 1 235, ale 309 začíná „Pomoz Věžce na…“. Občas sklouznou ke špatné šachové
  logice („jedním tichým tahem“ u šachu matem, „královu věž zaskočit“). Před nasazením je potřeba je projít, nebo je generovat
  s kontrolou motivu.
- V aplikaci Chess4Princesses zatím chybí typ cíle „mat jedním tahem“. Datový soubor je připravený, PR do aplikace ne.

## Ekonomika
Analýza celé databáze stála ≈ 19 Kč a 21 minut. Pro šachovou aplikaci to znamená, že se dá kdykoli přepočítat
po aktualizaci sítě nebo databáze (Lichess přidává úlohy průběžně). Stejné GPU by zvládlo hodnotit ~5 000 pozic za sekundu
pro živé hry, tedy tisíce souběžných hráčů s nápovědou „nejlepší tah podle sítě“.

*Reprodukce:* `uc3/prepare.py` → `uc3/run_lc0.sh` → `uc3/select_quizzes.py` → `uc3/texts.py`.
Data: `/srv/work/uc3/` (`positions.jsonl`, `runs/full-bt4/analysis.s*.jsonl`, `quizzes.json`, `quizzes_cs.json`).
Veřejná data Lichess (CC0), žádná data T‑Mobile.
