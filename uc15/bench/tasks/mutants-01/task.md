# Rarity: mutation bonus only when a mutation overwrites an occupied slot

Package `mutant_core` (pure Dart). The rarity `+1 mutation` bonus currently lands on most creatures
because any mutation counts. New default rule: the bonus is earned only if a mutation **replaced an
occupied slot**; `RulesConfig.v1` keeps the old "any mutation" behaviour.

- `RulesConfig`: new `final bool mutationBonusNeedsOverwrite` (default `true`, `v1` → `false`),
  supported by `copyWith` and `toString`.
- `Silhouette` (game_state.dart): new counter `mutationOverwrites` (also copied wherever the other
  counters are), incremented by the engine when a mutation replaces a part.
- `buildCreatureRecord` passes `mutationOverwrites` instead of `mutations` to the rarity input when the
  rule is on. Several overwrites still give +1 once.
- Optional: `bin/sim.dart` flags `--chaos N` and `--mutation-overwrite true|false`.

Verify: `cd mutant_core && dart test`.
