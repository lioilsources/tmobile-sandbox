# promo-tribute: make --dry-run read-only

`cmd/promo-tribute` with `--dry-run` currently claims every queued tribute and then gives it back
with an error, so the queue marks them failed (and the agent token cannot requeue them).

Change the dry run so it never claims or finishes anything:
- list the queue via the API client (`Tributes(ctx, "queued", limit)`), sort oldest first (by ID —
  the order a real run claims in), honour `--max`, and print one line per tribute
  `#<id> <credit> → <style ids>` (styles picked like a real run);
- print `tributes previewed: N (dry run, queue untouched)` to stderr and exit 0;
- the real (non-dry) run path is unchanged.

Verify: `go test ./cmd/promo-tribute` (TestDryRunDoesNotTouchQueue, TestDryRunRespectsMax, TestRunClaimsAndRecordsFailure).
