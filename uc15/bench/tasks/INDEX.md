# UC15 — 25 benchmarkových úloh z veřejných repozitářů `lioilsources`

Každá úloha = adresář `<id>/` s:
- `task.md` — zadání pro agenta (anglicky, ve stylu GitHub issue; řešení neprozrazuje, ale jmenuje API, které testy vyžadují),
- `task.json` — metadata pro harness (`base_commit`, `gold_commit`, `workdir`, `test_files`, `setup_cmd`, `test_cmd`, `fail_to_pass`, `pass_to_pass`, `timeout_s`, `difficulty`),
- `base.patch` — jen u úloh `kind: removal` (aplikuje se na `base_commit`).

Agent dostane **jen `task.md`** a pracovní kopii repa ve stavu `base_commit` (+ `base.patch`), **bez** testových souborů z gold commitu. `task.json` (obsahuje `prod_files` a gold commit) se agentovi neukazuje.

## Druhy úloh

- **reverse-commit** (19×) — skutečný historický commit, který přidal feature/fix spolu s testy (metoda SWE-bench). Base = rodič commitu, skryté testy = testové soubory z gold commitu.
- **removal** (6×) — u repo, jejichž historie je squashnutá (Mutants, Kittens) nebo má jediný vhodný commit (Aukrofy): na HEAD se těla dobře otestovaných funkcí nahradí stubem (`throw UnimplementedError` / `panic`), doc komentáře zůstanou. Gold = HEAD.

## Jak se ověřuje (pass@1)

1. `git clone --shared` z `/srv/work/w2/uc15/repos/<repo>` → checkout `base_commit` → `git apply base.patch` (pokud je).
2. Agent pracuje (vidí jen `task.md`).
3. Harness **přepíše** soubory z `test_files` verzí z `gold_commit` (`git show <gold>:<path>`), aby agent nemohl testy obejít ani je nevědomky rozbít.
4. Ve `workdir` spustí `setup_cmd` (je-li) a `test_cmd` s prostředím z `env.sh`. **Exit 0 = úloha vyřešena.** `test_cmd` spouští celý dotčený balíček / všechny testy v souboru, takže pokrývá `fail_to_pass` i `pass_to_pass` (regrese).

## Úlohy

| id | repo | jazyk | úloha | ověření (`workdir`: `test_cmd`) | obtížnost |
|---|---|---|---|---|---|
| aukrofy-01 | AukrofySimulation | Go | Penny aukce: prodloužení odpočtu jen v kritické zóně | `.`: `go test ./internal/auction` | easy |
| aukrofy-02 | AukrofySimulation | Go | (removal) implementovat Vickrey aukci (druhá cena, reserve) | `.`: `go test ./internal/auction` | medium |
| aukrofy-03 | AukrofySimulation | Go | (removal) implementovat Dutch aukci (klesající cena) | `.`: `go test ./internal/auction` | medium |
| ftg-01 | FineTuneGallery | Go | Eval: `?source=human\|judge\|merged`, κ-gate, nové sloupce CSV | `server`: `go test .` | medium |
| ftg-02 | FineTuneGallery | Go | Migrace v5 `criteria_judgments` + Cohenovo κ s intervalem a gate | `server`: `go test .` | medium |
| ftg-03 | FineTuneGallery | Go | `GET /stylemaps/{path...}` — statické soubory bez traversal/dotfiles | `server`: `go test .` | easy |
| ftg-04 | FineTuneGallery | Go | Migrace v2 (style/lora strength/repose), filtry, `DELETE /api/sessions/{id}`, registr modelů | `server`: `go test .` | hard |
| statusc-01 | status-collector | Go | Teplota GPU z textfile collectoru node_exporteru (se stárnutím) | `.`: `go test ./internal/host` | easy |
| statusc-02 | status-collector | Go | Disky: každý FS jednou, `MissingDisks()`, skip FS < 1 GiB | `.`: `go test ./internal/host` | medium |
| chatbridge-01 | chatbridge | Go | Nový balíček `rules`: katalog akcí, parser, `HashUser`, limiter (cooldown, token bucket, entity cap) | `.`: `go test ./internal/rules ./internal/twitch` | medium |
| promoclown-01 | PromoClown | Go | `promo-tribute --dry-run` jen čte frontu (nic neclaimuje) | `.`: `go test ./cmd/promo-tribute` | medium |
| promoclown-02 | PromoClown | Go | `MinDaysBetween *int` — 0 je platná hodnota, nil = výchozí týden | `.`: `go test ./internal/core` | easy |
| genealogy-01 | genealogy-matriky | Go | Analytické SQL dotazy: velikost rodin vč. bezdětných, top rodiny, sňatky na osobu, znovusňatky | `genealogy`: `go test .` | medium |
| aistack-02 | AiStack | Go | Proxy portálu: odmítnout path traversal v `/cpe/workflow/save` | `services/gen-queue`: `go test ./internal/portal` | easy |
| mutants-01 | Mutants | Dart | Bonus za mutaci jen při přepsání slotu (`RulesConfig.v1` zachová staré) | `mutant_core`: `dart test` | easy |
| mutants-02 | Mutants | Dart | Časované rozdávání z banku, nové výchozí `RulesConfig` vs `v1` | `mutant_core`: `dart test` | hard |
| mutants-03 | Mutants | Dart | (removal) `evaluateRarity` podle doc komentáře | `mutant_core`: `dart test` | medium |
| mutants-04 | Mutants | Dart | (removal) `dominantElement`, `uniqueCreatureName`, `romanNumeral` | `mutant_core`: `dart test` | easy |
| kittens-01 | Kittens | Dart | (removal) pravidla: shoda karet, reakce Kradež/Štít/Ne!, doutnák, odměna | `fuse_core`: `dart test` | medium |
| kittens-02 | Kittens | Dart | (removal) replay: JSON serializace + znovupřehrání | `fuse_core`: `dart test` | easy |
| wlp-01 | WorldLibraryProject | Python | Tokenový rozpočet podle písma + `is_context_error` v LLMBatch | `rag`: `pytest tests/test_chapter_budget.py` | medium |
| wlp-02 | WorldLibraryProject | Python | Export bundlů: závorky, `split_gloss`, jeden pojem = jedna bublina | `rag`: `pytest tests/test_export_bundle.py` | medium |
| wlp-03 | WorldLibraryProject | Python | `parse_json` opraví holé hodnoty a chybějící čárky | `rag`: `pytest tests/test_search_logic.py` | easy |
| wlp-04 | WorldLibraryProject | Python | Listy z úseků (verše „1,1–12"), texty originálu, `EXPORT_VERSION pg-2` | `rag`: `pytest tests/test_export_bundle.py tests/test_kindlify_sync.py` | hard |
| aistack-01 | AiStack | Python | Audio vibe: 503 `MODEL_DOWN` když hudební model neběží, `BackendUnavailable` | `services/audio`: `pytest tests/test_vibe.py` | medium |

Mix: 14× Go, 6× Dart (čistý Dart, bez Flutteru), 5× Python, 0× TS (LiveCode má jen výukové lekce — vynecháno).
Obtížnost: 9× easy, 13× medium, 3× hard (hrubý odhad podle rozsahu diffu a počtu nových API).

Rezervy (validované, ale nezařazené; jen `task.json`, bez `task.md`): `_reserve/aistack-03` (TTS rytmus, nový modul ~490 ř., velmi těžké), `_reserve/chatbridge-02` (in-memory fronta událostí + `GET /events`), `_reserve/statusc-03` (teploty strojů), `_reserve/promoclown-03` (termální pauza).

## Jak bylo validováno

Na VM (`/srv/work/w2/uc15`), skript `_tools/validate.py <taskdir>`:
- **(a)** `base_commit` (+ `base.patch`) + testové soubory z `gold_commit` → `test_cmd` **musí selhat** (u všech 25: rc 1, u Pythonu i rc 2 = chyba importu, u aistack-01 rc 134),
- **(b)** `gold_commit` → `test_cmd` **musí projít** (rc 0, u všech 25).
- `fail_to_pass` u reverse-commit úloh = nové testy v gold diffu; u Dart removal úloh = testy, které na base skutečně padají (`dart test -r expanded`, `_tools/f2p.py`).
- Kandidáty vybral `_tools/mine.py` (commity, které mění produkční kód i testy, 15–700 změněných řádků produkčního kódu, ≤ 12 souborů); stuby pro removal úlohy generuje `_tools/stub.py`, specifikace `_tools/gen.py` / `_tools/genrm.py`.

## Prostředí (VM)

`. uc15/bench/tasks/env.sh` — Go `/usr/local/go/bin`, Dart SDK `$UC15_TOOLS/dart-sdk`, Python venv `$UC15_TOOLS/venv`, Go/pub cache pod `$UC15_TOOLS`.
Reinstalace toolchainů (≈ 3 GB celkem):

```bash
R=/srv/work/w2/uc15; mkdir -p $R/{repos,tools}
# Dart SDK (stable)
cd $R/tools && V=$(curl -s https://storage.googleapis.com/dart-archive/channels/stable/release/latest/VERSION | python3 -c 'import json,sys;print(json.load(sys.stdin)["version"])') \
  && curl -sLo dart.zip https://storage.googleapis.com/dart-archive/channels/stable/release/$V/sdk/dartsdk-linux-x64-release.zip \
  && python3 -c 'import zipfile;zipfile.ZipFile("dart.zip").extractall(".")' && rm dart.zip && chmod +x dart-sdk/bin/* dart-sdk/bin/utils/*
# Python venv (bez torch/CUDA)
python3 -m venv $R/tools/venv && $R/tools/venv/bin/pip install pytest fastapi pydantic openai pyyaml lxml httpx uvicorn numpy "psycopg[binary]" pymupdf requests librosa==0.11.0 soundfile==0.13.1 python-multipart
# Repa (plná historie)
cd $R/repos && for r in AukrofySimulation FineTuneGallery status-collector chatbridge PromoClown genealogy-matriky AiStack Mutants Kittens WorldLibraryProject; do git clone https://github.com/lioilsources/$r.git; done
```
