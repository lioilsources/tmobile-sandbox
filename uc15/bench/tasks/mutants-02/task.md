# Pace the build with timed pot dealing (RulesConfig defaults vs v1)

Package `mutant_core` (pure Dart). A creature is built far too fast because hands are refilled after
every throw. Introduce timed dealing from the pot and new default rules, keeping the old numbers as
`RulesConfig.v1`:

- `RulesConfig` gains `handSize` (default 3), `refillOnPlay` (default false), `dealIntervalMs`
  (default 6000, 0 = off), `maxHandSize` (default 5), `drainPerSec` default 0.85 (+`drainPerHatch` 0.2),
  `chaosOriginsPerPoint` default 5; `static const v1` = the original plan (hand 5, refill on, no timed
  dealing, chaos 3, original drain). All fields in `copyWith`.
- `GameState.nextDealAt` (null when dealing is off). Every `dealIntervalMs` the pot deals one card to the
  player with the fewest cards. If that player's hand is full (`maxHandSize`), the oldest card that
  does not fit the creature is swapped back to the pot first; a full hand of fitting cards is left alone.
- No refill after a throw/pass when `refillOnPlay` is false. Hazards still surface while the pot deals;
  the deal rhythm continues across a hatch. Emit a dedicated event for a deal.
- Rarity chaos uses `chaosOriginsPerPoint` from the rules.
- Optional: `bin/sim.dart` gets rule flags and `--sweep KEY=a,b,c`.

Verify: `cd mutant_core && dart test` (dealing_test.dart, rarity_test.dart and the rest).
