# LLM JSON parser: repair bare string values and missing commas

`rag/llm_batch.py` `parse_json(text)` extracts the first `{...}` block and tolerates only a trailing comma.
Qwen3 sometimes returns `"gloss_cs": Úryvek se zabývá …` (unquoted value) and omits the comma before the
next key, losing ~4 % of chunks.

Make `parse_json` try progressively more tolerant candidates: as-is → without trailing commas → repaired
(quote bare values that run to end of line, escaping `\` and `"`; insert missing commas between a value
and the next `"key"` line; drop trailing commas). Valid JSON must never be altered by the repair step.

Verify: `cd rag && python -m pytest tests/test_search_logic.py`.
