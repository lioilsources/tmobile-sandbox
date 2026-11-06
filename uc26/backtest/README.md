# UC26 backtest — pravidla, která se nesmí porušit

Převzato z plánu (`Prompts/07-BESTBET_VALIDATION_PLAN.md`, sekce 3). U každého pravidla je uvedeno, kde ho kód vynucuje a který test ho hlídá.

1. **Walk-forward:** trénuj na sezónách 1..k, testuj na k+1, posuň. Žádné náhodné dělení zápasů.
   → `walkforward.run`: s0 jen trénink, s1 validace, s2…sK test s rozšiřujícím se oknem. Test: `test_walkforward_seasons_and_locked_threshold`.
2. **Point-in-time features:** test v CI, který ověří, že žádná feature nečte data s `available_at >= bet_time`.
   → do DB přistupuje jen `features/pit.py` (filtr `available_at < bet_time` v SQL a kontrola výstupu), ostatní moduly ve `features/` nesmí SQL vůbec spouštět. Testy: `tests/test_pit_leakage.py` (hranice `==` se nepočítá, smazaná budoucnost a otrávená budoucnost nesmí features změnit, statická kontrola importů).
3. **Kurz v čase sázky**, ne závěrečný, pro výpočet zisku. Závěrečný jen pro CLV.
   → `dataset.bet_time_odds` bere poslední snapshot sázkové kanceláře před `bet_time` (taky přes `pit.py`). `closing_odds` se používá jen pro CLV a Baseline A. Test: `test_dataset_excludes_non_final_and_uses_bet_time_odds`.
4. **Marže a limity:** zisk počítat z reálného kurzu kanceláře, u které bys sázel; okrajové ligy mají nízké limity, modelovat max. sázku (např. 500 Kč) a vliv na kurz.
   → `--bet-book`, `Config.max_stake` (výchozí 5 j. ≈ 500 Kč při 1 j. = 100 Kč), `Config.slippage` (zhoršení kurzu vlastní sázkou). Test: `test_flat_and_kelly_cap`, `test_slippage_reduces_profit`.
5. **Staking:** flat 1 jednotka a frakční Kelly (1/4). Reportovat obojí. Pokud edge existuje jen s Kellym, neexistuje.
   → `staking.flat`, `staking.kelly`; verdikt se počítá jen z flat (`verdict.decide`).
6. **Bootstrap:** 10 000 převzorkování sázek → interval ROI.
   → `metrics.bootstrap_roi_ci(n_boot=10_000)`, 95% percentilový interval. Test: `test_bootstrap_ci`.
7. **Žádné ladění na testovací sadě.** Jedna validační sezóna na threshold a hyperparametry, pak zámek.
   → threshold a C se vybírají jen na s1 a pak jsou zamčené; zamíchání výsledků testovacích sezón je nesmí změnit (`test_walkforward_seasons_and_locked_threshold`).
8. **Report** `reports/backtest_<date>.md`: počet sázek, ROI flat, ROI Kelly, CLV, interval, drawdown, výsledky po ligách a po trzích (1X2, handicap, total). Pokud edge žije jen v jedné lize s 80 sázkami, je to šum.
   → `report.render`: verdikt, souhrn, modely (log-loss, Brier), tabulky po ligách, trzích a sezónách; skupiny pod 200 sázek jsou označené „málo“ a report varuje, když zisk stojí na jedné lize.

## Verdikt (sekce 0, natvrdo v `verdict.py`)

GO jen tehdy, když platí všechno: průměrné CLV > +2 %, CLV kladné u > 55 % sázek, ROI flat > +3 % na ≥ 1 000 sázkách walk-forward, 95% bootstrap CI ROI nad nulou. Prahy se nemění bez změny plánu.

## Spuštění

```
cd uc26
.venv/bin/python -m backtest.run --db data/<db>.sqlite --bet-book <kancelář> --close-book <ostrá kancelář>
.venv/bin/python -m backtest.run --help
```
