# Eval endpoint: choose label source human | judge | merged

FineTuneGallery (`server/`, package main) has an eval matrix (`handleEval` in `server/eval.go`)
that counts criterion labels from `image_criteria` (human labels). There is also a VL judge whose
verdicts live in `criteria_judgments` and a per-criterion κ-gate (`judgeStats`, `passedCriteria`,
gate states `pass|fail|uncalibrated`).

Add a query parameter `source` to the eval endpoint:
- `human` (default, also when missing): exactly today's behaviour — no new JSON keys, no new CSV columns.
- `judge`: count only judge verdicts of the current judge version (`s.judge.Version()`), only on
  criteria whose gate is open; every other criterion cell is n/a with reason `"gate: <state>"`.
- `merged`: the human label wins where it exists; the judge fills images no human rated (again only gated criteria).
- Abstentions (verdict 0) never count. A gate does not carry over to a new judge version.
- Unknown value → 400; judge disabled → 503 (`ErrJudgeDisabled`).
- Cells gain `ratedHuman`, `ratedJudge` (omitempty) and `na` (omitempty); for non-human sources the
  response also carries `source` and a `judge` block. CSV export for non-human sources adds per
  criterion `<c>_rated_human`, `<c>_rated_judge`, `<c>_na` columns.

Verify: `cd server && go test .` (TestEval* incl. TestEvalSourceErrors, TestEvalCSVCarriesSourceColumns).
