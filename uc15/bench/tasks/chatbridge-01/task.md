# Phase 2: chat command parser, action catalog and rate limiter

Create package `internal/rules` (module `github.com/lioilsources/chatbridge`), using the existing
`internal/config` types (`config.Action` with `Name`, `Type`, `Aliases`, `TTL`; `config.Limits` with
`UserCooldownSec`, `GlobalActionsPerMin`, `MaxLiveEntities`):

- `NewCatalog(actions []config.Action) *Catalog`; `(*Catalog).Lookup(cmd string) (*config.Action, bool)`
  (aliases matched case-insensitively, trimmed); `(*Catalog).Parse(message string) (*config.Action, bool)`
  using the first whitespace token; empty/unknown → false.
- `HashUser(nick string) string`: stable opaque key (e.g. hex of a SHA-256 prefix of the lowercased,
  trimmed nick) so raw nicks are never stored.
- `Clock` interface (`Now() time.Time`), `SystemClock() Clock`.
- `Reason` (`ReasonOK`, `ReasonUserCooldown`, `ReasonGlobalBudget`, `ReasonEntityCap`) with `String()`
  (`ok`, `user_cooldown`, `global_budget`, `entity_cap`).
- `NewLimiter(cfg config.Limits, clock Clock) *Limiter`; `(*Limiter).Allow(userHash string, isSpawn bool, ttl time.Duration) (bool, Reason)`
  enforcing, in this order, per-user cooldown, a global token bucket (capacity = actions/min, starts
  full, refills continuously) and a cap on live spawned entities (only spawns with ttl > 0 are tracked;
  they expire after ttl). Consumption is recorded only when allowed. `(*Limiter).LiveCount() int`.
- Wire it in `cmd/chatbridge/main.go` (parse → limiter → log accepted actions).

Verify: `go test ./internal/rules ./internal/twitch`.
