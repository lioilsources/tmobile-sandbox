# Implement creature naming helpers

In `mutant_core/lib/src/rules/naming.dart` three functions are stubs. Implement them per their doc comments:
- `ElementType? dominantElement(List<CardDef> partsInSlotOrder, {ElementType? sealElement})` — the seal
  wins; otherwise the most common element among parts, ties broken by the earliest slot; null if none.
- `String uniqueCreatureName(String base, Set<String> taken)` — returns `base` if free, otherwise appends
  ` II`, ` III`, … (first free).
- `String romanNumeral(int n)` — standard Roman numerals; `ArgumentError` for n ≤ 0.

Verify: `cd mutant_core && dart test`.
