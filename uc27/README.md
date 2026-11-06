# UC27 — agenttune: eval a LoRA trénink lokálního coding agenta

Projekt [agenttune](https://github.com/lioilsources/agenttune) je smyčka **eval → data → LoRA SFT + DPO → promotion gate**. Učí lokální open-weights coding model dodržovat konvence vlastních repozitářů. Smyčku řídí samostatná session (DevConventions/agenttune). L40S pro ni slouží jako GPU pro serving i trénink. Kód smyčky je v repu agenttune, tady je jen to, co se týká L40S.

## Co běží na L40S

| Krok | Profil / úloha | Stav |
|---|---|---|
| 1. Baseline eval | profil W: `qwen3-coder` (Qwen3-Coder-30B-A3B-Instruct-FP8), :8003, kontext 64k, `--tool-call-parser qwen3_coder` | 6. 11. okno `agenttune-baseline` (qctl, 3 h) |
| 2. Trénink | LoRA/QLoRA SFT, pak DPO, výplň v qctl (`filler`, SIGTERM → checkpoint → resume) | čeká na zadání a dataset |
| 3. Eval po tréninku | stejný harness jako krok 1, model s LoRA | — |

## Eval

- Na Macu běží `evalctl` s OpenCode jako agentem. Úlohy edituje v dočasných klonech na Macu, z L40S používá jen OpenAI-kompatibilní endpoint přes ssh tunel (`-L 8003:localhost:8003`).
- Jeden stream požadavků, žádný paralelismus. Sady: 1 zkušební úloha, dev split (20), holdout (5).
- Okno otevírá dispečer `queue/qctl`: marker `AGENTTUNE_READY` se zapíše, až odpovídá `/v1/models`. Konec okna je `AGENTTUNE_DONE` nebo vypršená pojistka.

## Trénink

- Venv `/cold/venvs/agenttune`, výstupy a checkpointy `/cold/agenttune/`. `/srv` je plný, HF cache zůstává v `/srv/hf`.
- Ověřená cesta z UC1: Unsloth QLoRA. `max_seq_length` nastavit i na modelu, jinak se tiše zkrátí na 2048. Kopie `unsloth/*` po tréninku smazat.
- Dataset: jen vyčištěné `sft.jsonl` a `dpo.jsonl` z Macu, do `/cold/agenttune/data/`. Surové přepisy sessions se nekopírují a do tohoto (veřejného) repa nepatří ani dataset.

## Výsledky

Čísla z běhů (tok/s, TTFT, trvání úloh, chybné tool cally, skóre evalu před a po tréninku) jdou do `MODEL_NOTES.md` a do `reports/uc27-agenttune.md`.
