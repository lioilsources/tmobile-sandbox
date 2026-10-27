# Implement the Vickrey (sealed second-price) auction format

`internal/auction/vickrey.go` contains only a stub of `func (e *Engine) runVickrey(ctx context.Context) Result`.
Implement it so the Vickrey format works like the other formats in the package (see `dutch.go`,
`penny.go`, `engine.go` for the helpers: `snapshot`, `Parts.Decide`, `tryBuyNow`, `recordBid`,
`recordNonBid`, `Parts.Leave`, `Parts.Debit`, `emit`).

Rules:
- Start price is `Params.StartPrice`; emit `submission_open` with `window_s` (= `Params.SubmissionWindow`) and `reserve`.
- One decision round: every bidder submits once (sealed). `ActionBuyNow` may end the auction
  immediately via `tryBuyNow`; `ActionLeave` removes the bidder; `ActionBid` with amount > 0 is a
  sealed bid (record it, emit `sealed_bid_received`); everything else is a non-bid.
- No bids → `no_sale` (reason), empty `Result`. Highest bid under `Params.ReservePrice` → `no_sale`.
- The highest bidder wins and pays the **second highest** bid, at least the reserve price and never
  more than their own bid. Debit the winner, set the leader, return `Result{WinnerID, FinalPrice}`.

Verify: `go test ./internal/auction` (TestVickreySecondPrice, TestVickreyReserveNoSale and all others).
