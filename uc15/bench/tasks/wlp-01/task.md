# Chapter enrichment: token budget by script; context-length errors are input errors

In `rag/` (WorldLibraryProject):

`enrich_chapters.py` sizes chapter input by characters (60 000), which overflows the model context for
non-Latin scripts. Replace it with an estimated token budget:
- `est_tokens(text) -> int`: conservative upper estimate without a tokenizer — ASCII ≈ 0.3 token/char,
  Latin with diacritics ≈ 1 token/char, other scripts (Greek, Han, Devanagari, …) 1.5; at least 1.
- `fit(text, budget) -> str`: longest prefix within the budget; `fit_list(items, budget)`: longest prefix
  of the list within the budget (count a small separator cost per item);
  `windows(chunks, budget) -> list[str]`: pack consecutive chunks into windows within the budget,
  truncating a single oversized chunk instead of dropping it.
- Use these in `main()` (new `--max-model-len`, default 32768) for one-shot vs map-reduce.

`llm_batch.py`: add `is_context_error(exc) -> bool` (status 400/413/None and a message like "maximum context
length", "context length exceeded", "max_model_len", "prompt is too long", …). In `LLMBatch`, such an
error fails only that item (count it as `too_long` in `BatchStats`), no retries and no outage back-off.

Verify: `cd rag && python -m pytest tests/test_chapter_budget.py`.
