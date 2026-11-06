# UC26 — BestBet: má „vlna“ edge, nebo ne?

Validace podle `Prompts/07-BESTBET_VALIDATION_PLAN.md`. Cíl: **jedno číslo**, které rozhodne, jestli model „vlny“ (forma, absence, brankář, únava, momentum) na okrajových hokejových ligách poráží trh. Žádný produkt ani UI, dokud to číslo neexistuje.

Stav: kostra fází 2–3 (point-in-time store, features, modely, backtest s verdiktem) a syntetická data se známým edge. Ingest (fáze 1) jsou zatím jen stuby. Papírové sázení (fáze 4, `paper/`) ještě neexistuje.

## Rozhodovací kritéria (sekce 0 plánu, doslovně)

> Model má edge, pokud **všechny tři** platí:
>
> | Kritérium | Práh | Proč |
> |---|---|---|
> | **CLV (closing line value)** | průměrně > +2 % na sázku, kladné u > 55 % sázek | Jediná metrika, která se nedá "vyhrát" štěstím. Pokud sázíš lepší kurz, než je závěrečný, dlouhodobě vyhráváš. |
> | **ROI v backtestu** | > +3 % na min. 1 000 sázek, walk-forward | Zisk po marži. Pod 1 000 sázek je šum. |
> | **Bootstrap** | 95% interval ROI nezahrnuje nulu | Statistická významnost, ne pocit. |
>
> Pokud kterékoli selže → projekt se **zavře s číslem** a zapíše se, proč. Žádné "ještě jeden feature".
>
> Papírové sázení (fáze 4) musí pak potvrdit kladné CLV naživo na ≥ 300 sázkách. Teprve potom se řeší produkt nebo peníze.

Prahy jsou natvrdo v `backtest/verdict.py`. Pravidla backtestu jsou v [`backtest/README.md`](backtest/README.md).

## Co do repa nepatří

Repo je **veřejné**. Žádné stažené stránky, kurzy, ceny, jména ze scrapingu, IP, hostnames ani klíče. `data/`, `*.sqlite` a vygenerované `reports/*` jsou v `uc26/.gitignore`. Syntetická data jsou vymyšlená (ligy `SYN1…`, kanceláře `softbook` a `sharpbook`).

## Kde co běží

- **Mac:** ingest (scraping a snapshoty kurzů), stavba SQLite, vývoj, testy. Na T-Mobile VM se nestahuje nic.
- **L40S VM:** jen dávkové výpočty nad hotovou DB (`run_l40s.sh`, výstupy do `/cold/uc26/`), např. GBM nebo bootstrap sweep přes ligy a sezóny. Zatím placeholder, nespouštět.
- **Kurzy:** nejdřív zdarma (OddsPortal a vlastní snapshoty Tipsportu a Fortuny), o placených zdrojích se rozhodne později.

## Struktura

```
uc26/
  schema.sql        games, results, lineups, player_games, odds_snapshots, closing_odds; každý řádek má available_at
  db.py             připojení, formát času (UTC 'YYYY-MM-DD HH:MM:SS')
  synth.py          syntetické ligy, sestavy a kurzy se zadaným edge (výchozí 0)
  ingest/           rozhraní Source/Row, RateLimiter, upsert; stuby eliteprospects, flashscore, OddsPortal, Tipsport/Fortuna
  features/pit.py   JEDINÝ přístup features do DB: vše filtruje available_at < bet_time
  features/build.py features vlny (forma, absence, brankář, odpočinek, momentum)
  models/           baseline_novig.py (A), baseline_elo.py (B), wave_logreg.py
  backtest/         dataset, walk-forward, staking, metriky, verdikt, report, CLI (run.py)
  tests/            leakage, modely, backtest, ingest bez sítě, smoke test na syntetice
  reports/          backtest_<date>.md (gitignored)
  run_l40s.sh       placeholder pro dávku na L40S
```

Odchylka od plánu 1.3: výsledek zápasu je ve vlastní tabulce `results`. Rozpis je známý týdny předem, výsledek až po zápase, takže každý řádek má jediné `available_at`. Status `games.status` se ve features nepoužívá (mění se zpětně).

## Spuštění

```
cd uc26
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests               # ~1.5 min, včetně smoke testu GO/NO-GO

# syntetika + backtest
.venv/bin/python synth.py --out data/synth.sqlite --edge 0      # edge 0..1 = podíl informace ze sestav, kterou kurz při sázce nezná
.venv/bin/python -m backtest.run --db data/synth.sqlite --bet-book softbook --close-book sharpbook
#  → reports/backtest_<date>.md + JSON se souhrnem na stdout
```

Volby backtestu: `--league` (opakovatelně), `--bet-offset-min` (výchozí 60 = sázka v T-1h, sestavy se potvrzují těsně před zápasem), `--max-stake`, `--slippage`, `--n-boot`.

## Syntetická data a smoke test

`synth.py` generuje hokejové zápasy (střely ~ Poisson, góly ~ Binomial podle save% chytajícího brankáře), zranění hráčů (Markov), sestavy potvrzené v T-75 min, box score a kurzy:

- `softbook` (tady se „sází“): snapshoty T-24h a T-2h, marže 5 %, nezná podíl `edge` informace ze sestav (absence, kdo chytá),
- `sharpbook`: závěrečný kurz při začátku zápasu, zná všechno, marže 2,5 %,
- ~1 % zápasů je odložených. Mají kurzy, ale nemají výsledek a backtest je počítá jako vyřazené.

Smoke test (`tests/test_smoke_synth.py`, 3 ligy × 5 sezón × 14 týmů): `edge=0` musí dát **NO-GO**, `edge=1` musí dát **GO**. Poslední běh: edge 0 → NO-GO (1 331 sázek, ROI flat +3,2 % jen štěstím, CI [−7,6 %, +14,1 %], CLV −4,5 %, kladné jen u 8 %); edge 1 → GO (1 149 sázek, ROI flat +19,8 %, CI [+11,0 %, +29,0 %], CLV +17,4 %, kladné u 70 %). Edge 0 ukazuje, proč je CLV v kritériích: kladné ROI ze štěstí neprojde.

## Další kroky

1. Fáze 1: implementovat ingest (nejdřív robots.txt a podmínky webu, rate limit, cache v `data/raw/`). Hned rozjet denní snapshoty kurzů CZ kanceláří.
2. Spustit backtest na reálných datech, rozhodnutí zapsat do `reports/` a pak buď `paper/` (fáze 4), nebo `POSTMORTEM.md`.
