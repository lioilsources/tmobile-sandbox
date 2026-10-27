# Penny auction: countdown extension must only fire in the trigger zone

In the penny auction engine (`internal/auction/penny.go`, `runPenny`), the viral rule
`ViralCountdownExtension` currently extends the timer by `Viral.CountdownExtendS` on **every**
bid. It should only extend when the bid lands in the critical zone, i.e. when the time that was
remaining before the bid reset the timer is `<= Viral.CountdownTriggerS`.
`CountdownTriggerS <= 0` keeps the old behaviour (extend on every bid).

Acceptance:
- A bid placed while plenty of time remains resets the timer but does not add the extension and
  does not emit `countdown_extended`.
- A bid placed with `remaining <= CountdownTriggerS` resets the timer **and** adds `CountdownExtendS`
  (event `countdown_extended` with `by_s`).
- Existing auction tests (`go test ./internal/auction`) keep passing.
