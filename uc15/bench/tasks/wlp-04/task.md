# Bundles: verse labels, leaf sections under chapters, original text

`rag/export_bundle.py` (+ `kindlify_sync.py`, new `verse_refs.py`) — extend the Kindlify bundle export:
- Leaves: under every chapter with more than one chunk, add one leaf per chunk (`kind: "paragraph"`,
  id `leaf_id(seq)` = `k%05d`, label from `leaf_label(row)`): CTS refs `ref_start`/`ref_end` like "1.1"–"1.12"
  render as "1,1–12", across chapters "1,44–2,3", a single ref as "1,12", no refs → "Úsek <n>".
  Leaves get at most `LEAF_TOP_TERMS = 5` terms.
- Texts: optional `texts` key `{nodeId: original text}` — a leaf carries its chunk without the overlap
  with the previous chunk (`strip_overlap(prev, cur)`), a chapter without leaves its whole text; when
  there are no texts the content digest stays the same as before (`content_digest(..., texts=None)`).
- Bump `EXPORT_VERSION` to `pg-2`; chapter summaries use the `long` detail.
- `kindlify_sync.build_index` keeps working with the new bundle shape.

Verify: `cd rag && python -m pytest tests/test_export_bundle.py tests/test_kindlify_sync.py`.
