# Implement the Kittens rule helpers (match, reactions, fuse, reward)

Package `fuse_core` (pure Dart). The rule helpers in `lib/src/rules/` are stubs; implement them per their
doc comments and the engine that calls them (`lib/src/engine/`, `lib/src/model/rules_config.dart`):

- `match_rule.dart`: `matchesTarget(card, target, mode)` — only number cards; `MatchMode.normal` = same
  colour or same number, `MatchMode.twist` = number differs by exactly 1. `isValidThrow(state, card)` —
  number cards must match the target; twist/blind always valid; defuse/detonate only when a bomb is live;
  bomb/nope/shield/steal are never valid throws.
- `reactions.dart`: `stealAllowedAt` (hesitation strictly longer than `rules.stealThresholdMs`),
  `shieldableLoss` (earliest-deadline pending loss of the player still inside the shield window and
  before its deadline), `nopeTargetOpen` (last action exists and is open), `ghostNopeAvailable`
  (ghost, once per round).
- `bomb_fuse.dart`: `tickFuse(bomb)` — decrement; at ≤ 0 the bomb explodes (returns `bomb: null, exploded: true`).
- `turn_reward.dart`: `rewardDraw(rules, throwMs)` — `drawFast` when `throwMs <= fastThrowMs`, else `drawNormal`.

Verify: `cd fuse_core && dart test`.
