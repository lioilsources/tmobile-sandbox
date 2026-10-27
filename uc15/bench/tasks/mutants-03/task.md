# Implement rarity evaluation (`evaluateRarity`)

In `mutant_core/lib/src/rules/rarity.dart` the function `RarityResult evaluateRarity(RarityInput input)`
is a stub. Implement it according to its doc comment (PLAN §4 Rarita 0–5):

- chaos: +1 per `input.chaosOriginsPerPoint` distinct origins among parts (null origin ignored), **or**
  harmony +2 when all parts share one element and there are no stumps — whichever is higher (harmony
  wins ties), never both; reasons `chaos` / `harmony`;
- +1 any fusion (`fusion`), +1 any mutation (`mutation`), +1 per met hazard (`hazard`), +1 for ≥ 3 synchro
  throws (`synchro`), −1 premature hatch (`premature`), −1 per 2 stumps (`stumps`);
- `raw` = unclamped sum of reasons, `rarity` = `Rarity.values[raw.clamp(0, Rarity.values.length - 1)]`.

Verify: `cd mutant_core && dart test`.
