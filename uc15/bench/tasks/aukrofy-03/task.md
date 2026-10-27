# Implement the Dutch (descending price) auction format

`internal/auction/dutch.go` contains only a stub of `func (e *Engine) runDutch(ctx context.Context) Result`.
Implement the Dutch auction in the style of the other formats in the package (`vickrey.go`,
`penny.go`, helpers in `engine.go` / `types.go`).

Rules:
- The price starts at `Params.StartPrice` and drops by `Params.Decrement` every tick; ticks are
  separated by `Clock.Sleep(Params.TickInterval)`; at most `Params.MaxTicks` ticks (default 1000 when <= 0),
  then `no_sale`. Respect `ctx` cancellation.
- `no_sale` when the price falls below `Params.ReservePrice` or no active bidders remain (`Parts.ActiveIDs()`).
- Each tick: emit `price_updated` (`price`, `next_price`), let bidders decide on a fresh
  `snapshot(tick, 0, next)`; pick the first winning bid (there is a helper `pickWinningBid`).
  Buy-now goes through `tryBuyNow`. A winner who cannot afford the current price (`Parts.CanAfford`)
  is recorded as a non-bid and the auction continues.
- The winner pays the **current** price: debit, `recordBid`, set the leader, emit `bid_placed`
  (`bidder_id`, `amount`, `reasoning`), return `Result{WinnerID, FinalPrice}`.

Verify: `go test ./internal/auction` (TestDutchFirstBidWins and all others).
