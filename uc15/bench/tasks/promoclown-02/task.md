# Project config: min_days_between = 0 must mean 0, not the default week

`model.Project.MinDaysBetween` is an `int`, so `min_days_between: 0` in projects.yaml is
indistinguishable from a missing field and `Service.UpsertProject` (internal/core) turns it into the
default 7 days. Some projects legitimately need no gap.

Make the field a pointer (`*int`, same json/yaml tags): nil → default (7 days), 0 → 0, negative → validation
problem "min_days_between cannot be negative". Adjust conversions to/from the DB row accordingly.

Verify: `go test ./internal/core` (TestMinDaysBetweenZeroSurvivesUpsert + existing tests).
