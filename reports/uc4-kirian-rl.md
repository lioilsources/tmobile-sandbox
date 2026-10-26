# UC4 – Kirian: RL agent jako automatický testér obtížnosti levelů

*26. 10. 2026 · T‑Mobile GPUaaS sandbox · Flutter 3.44 + Flame (hra Kirian/Tyrian Mobile, headless) · PPO (CleanRL styl, PyTorch na CPU)*

## Shrnutí
Hru Kirian jsme spustili bez obrazovky uvnitř `flutter test` a přes TCP ji ovládal agent trénovaný metodou PPO.
Za **1,5 hodiny a 12 milionů kroků** (≈ 2 200 kroků/s, 8 instancí hry, ≈ 150× rychleji než reálný čas)
se naučil přežít a **spolehlivě dokončit 10 z 18 sektorů** (90–100 %). Náhodný hráč dokončí v průměru 15 % epizod.
V **8 sektorech** agent nedokončí nic. Našel místo v rohu hrací plochy, kde ho nic nezasáhne, a zůstane tam se 100 % HP až do
limitu 150 s. Pro vývojáře hry je to nejcennější výsledek: automaticky nalezené „bezpečné místo“ a sektory,
kde se dá vydržet bez boje. Trénink běžel na CPU. GPU mezitím sloužila jiným úlohám (UC9, UC3), takže UC4 L40S nijak nevytížil.

## Jak to funguje
- **Headless hra:** `test/rl/rl_server_test.dart` (větev `rl-headless`, není pushnutá) spustí `TyrianGame` bez vykreslování
  a poslouchá na TCP portu. Příkazy `reset{sector}`, `step{a:[dx,dy,fire], k}`, `close`. Do hry přibyl jen háček `botInput`,
  který nahradí klávesnici.
- **Pozorování (145 čísel):** loď (pozice, HP, štít), 12 nejbližších nepřátel, 16 nejbližších střel, sběratelné předměty, čas.
- **Akce (18):** 9 směrů × střílet/nestřílet, jeden krok = 4 snímky hry.
- **Odměna:** +Δskóre/100, −poškození/20, +0,002 za krok, −5 za smrt, +5 za dokončení sektoru.
- **PPO:** 8 paralelních her, rollout 256, malá MLP. Na CPU byla 4× rychlejší než na sdílené GPU (2 100 vs 546 kroků/s),
  protože síť je malá a GPU zrovna jela naplno s vLLM.

## Průběh tréninku (po desetinách)

| Krok | Dokončeno | Smrt | Vypršel čas | Skóre | HP na konci |
|---|---|---|---|---|---|
| 1,2 M | 41 % | 17 % | 42 % | 154 | 0,77 |
| 5,9 M | 57 % | 2 % | 41 % | 643 | 0,96 |
| 12,0 M | 54 % | 1 % | 45 % | 769 | 0,98 |

Smrtí ubylo 15×, skóre vzrostlo 5×. Podíl epizod, které vyprší časem, se ale nezměnil, protože jsou to pořád tytéž sektory.

## Evaluace: agent vs náhodný hráč (30 epizod na sektor)

| Sektor | Agent dokončí | Náhodný dokončí | Agent skóre | Poznámka |
|---|---|---|---|---|
| 0 | 100 % | 100 % | 2 518 | lehký úvod |
| 1 | 100 % | 37 % | 893 | |
| 3 | 100 % | 63 % | 464 | |
| 6 | 100 % | 33 % | 0 | dá se dokončit bez výstřelu |
| 7 | 97 % | 0 % | 433 | |
| 8 | 100 % | 7 % | 540 | |
| 9 | 90 % | 0 % | 552 | jediný sektor, kde agent ještě umírá |
| 14 | 97 % | 3 % | 3 193 | |
| 15 | 97 % | 0 % | 0 | dá se dokončit bez výstřelu |
| 16 | 100 % | 17 % | 303 | |
| 2, 4, 5, 11, 12 | 0 % | 0–7 % | 323–1 368 | agent přežije 150 s v rohu |
| 10, 13, 17 | 0 % | 0 % | 0 | agent v rohu, ani jeden zásah |

Náhodný hráč umírá v těžkých sektorech za 17–43 s. Agent v nich vydrží celých 150 s s plným HP,
40–90 % času na jednom místě v rohu (souřadnice ≈ 42, 31). Sektor skončí, až všechny formace odletí nebo jsou zničeny.
V těchto sektorech formace zůstávají na obrazovce, dokud je někdo nesestřelí, takže sektor nikdy neskončí.

## Co to říká o hře
1. **Bezpečné místo v rohu:** v 8 sektorech se dá stát v rohu a nic hráče nezasáhne. Hráč tam nevyhraje, ale ani neprohraje.
   Pro hru s „vydrž a přežij“ je to slabina designu, kterou automat našel bez lidského testera.
2. **Sektory 6 a 15 se dají dokončit bez jediného výstřelu.** Formace prostě proletí.
3. **Křivka obtížnosti není monotónní:** náhodný hráč dokončí sektor 3 v 63 % a sektor 16 v 17 %, ale sektor 2 jen v 7 %
   a sektory 7–15 skoro nikdy. Sektor 9 je jediný, kde agent po tréninku ještě umírá (10 % epizod).

## Co je potřeba udělat dál
- **Odměna:** přidat malou penalizaci za čas a odměnu za zničení formace, aby se nevyplácelo kempovat.
  Případně curriculum: začít v těžkých sektorech s posílenou zbraní.
- **Ověřit na zařízení:** projít bezpečný roh ručně v emulátoru. Souřadnice jsou z pozorování agenta, vizualizace v GIFu je schematická.
- **Měřítko:** 8 her na 8 vCPU je strop. Na větším CPU by šlo desítky her paralelně a GPU by se uplatnila u větší sítě
  (například pozorování z pixelů), kterou jsme zatím nezkoušeli.

## Využití GPU
UC4 v této podobě L40S nepotřebuje. Simulace hry je čistě CPU a malá síť je na CPU rychlejší. Hodnota pro GPUaaS je v tom,
že trénink může běžet souběžně s GPU úlohami na stejném stroji, tady s UC9 a UC3.

*Reprodukce:* Kirian větev `rl-headless` (`test/rl/rl_server_test.dart`, háček `botInput`) → `uc4/ppo.py --envs 8 --steps 12000000`
→ `uc4/evaluate.py --episodes 30` → `uc4/replay_gif.py`. Data: `/srv/work/uc4/runs/ppo1/` (`agent.pt`, `episodes.jsonl`,
`train.jsonl`, `eval/eval.json`, `eval/replays/`, `eval/gif/`). Vlastní hra, žádná data T‑Mobile.
