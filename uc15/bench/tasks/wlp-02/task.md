# Bundle export: keep parentheses, one concept = one term bubble

`rag/export_bundle.py` builds term clouds for the Kindlify reader. Bugs from the first live export:
- `norm_term` strips `)` so "Tao (Cesta)" becomes "Tao (Cesta". Stop stripping parentheses (only a term
  wrapped entirely in parentheses is unwrapped).
- Add `split_gloss(term) -> (head, orig | None)`: "Wu wei (無為)" → ("Wu wei", "無為"),
  "Tao (Cesta)" → ("Tao", None) — the gloss is returned as `orig` only when it is not Latin script
  (helper `is_latin`).
- One concept = one term: in `chunk_terms`, merge the same term case-insensitively across keywords and
  entities; the kind is chosen by rank `orig` < `entity` < `word` (`KIND_RANK`).
- Across the whole work, add `canonical_terms(chunk_rows)` choosing the most frequent spelling per
  casefolded term, and use it in `own_counts(rows, canon)` / `build_bundle`; boosts in `score_terms`
  compare casefolded too.

Verify: `cd rag && python -m pytest tests/test_export_bundle.py`.
