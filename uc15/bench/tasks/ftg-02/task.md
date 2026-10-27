# Judge calibration: migration v5 and per-criterion Cohen's κ gate

FineTuneGallery (`server/`, package main) needs groundwork for a VL judge that rates images
next to humans.

1. DB migration **v5** (append to the `migrations` list in `server/db.go`): new table
   `criteria_judgments(image_id → images(id) ON DELETE CASCADE, criterion, judge, verdict ∈ {-1,0,1},
   reason TEXT default '', ref_key TEXT NOT NULL, created_at default now, PRIMARY KEY(image_id, criterion, judge))`
   plus an index on `(judge, criterion)`. Existing v4 rows must survive the migration.
2. New file `server/kappa.go` with an `agreement` type: the 2×2 table human×judge
   (`N, UpUp, UpDown, DownUp, DownDown` — first word = human), method `add(human, judge int)`
   (only ±1 pairs count), and `finish()` computing `Po`, `Pe`, `Kappa`, 95% interval `Lower/Upper`
   (asymptotic SE `sqrt(po(1-po)/n)/(1-pe)`, clamped to [-1,1]), `DownAgree` (share of human downs
   the judge also called down) — all `*float64`, nil when undefined (κ undefined when pe = 1) — and `Gate`.
3. Gate constants: `gateMinPairs = 40`, `gateMinHumanDown = 8`, `gateMinKappa = 0.60`,
   `gateMinKappaLower = 0.40`; states `gateUncalibrated = "uncalibrated"`, `gateFail = "fail"`,
   `gatePass = "pass"`. Too few pairs / too few human downs / undefined κ → uncalibrated;
   κ or its lower bound below threshold → fail; otherwise pass.

Verify: `cd server && go test .` (TestKappa*, TestMigrationV5KeepsV4Rows).
