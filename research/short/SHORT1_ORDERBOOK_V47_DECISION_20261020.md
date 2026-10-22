# SHORT1 Order-Book Intelligence – V47 Decision (2026-10-20)

## Scope
Research/shadow only. No executor, risk, order or lifecycle behavior changed.

## Historical reconstruction
- Source: Binance Vision futures `bookDepth` + existing V37 entry microstructure.
- V45B rows: 1,103 C600 entries.
- Usable ±1.0% :30 liquidity snapshot: 902 entries (~81.8%).
- Usable ±0.20% :30 liquidity snapshot: 688 entries (~62.4%).
- Snapshot timing error: median ~1s at :30, p90 ~3s, max 5s by construction.
- Historical `bookDepth` is a coarse ~30s percentage-band snapshot, not equivalent to live V44 depth20/L1/microprice.

## Main OOS result
Chronological 4-fold adverse-first prediction on actionable 20s entries:
- micro logistic mean AUC: ~0.547
- micro + ±1% book logistic: ~0.534 (worse)
- ±1% book-only logistic: ~0.473
- ±0.20% subset micro logistic: ~0.530
- ±0.20% subset micro+book logistic: ~0.532 (negligible)
- ±0.20% shallow-tree micro: ~0.510
- ±0.20% shallow-tree micro+book: ~0.527, but folds unstable.

Conclusion: coarse historical bookDepth does NOT justify a hard gate or standalone model.

## Context interactions
Some small-sample interactions remain research candidates:
- BUY_FLOW_PRICE_DOWN + bullish ±0.20% book shift showed higher adverse-first rate than bearish shift in both halves, but effect is modest.
- Existing extreme-sell + acceleration + 30s rebound group combined with bullish book shift showed high adverse-first in very small samples.
These are hypotheses only, not policy.

## Live exact V44
V44 uses one public combined WebSocket (`trade + depth20@100ms + bookTicker`) and preserves the V43 flow/price hazard contract exactly.
Order-book data is TELEMETRY_ONLY.

Live sample gate:
- actionable <20 or <5 distinct days: COLLECT_MORE
- 20–39 or <10 distinct days: EXPLORATORY
- >=40 and >=10 distinct days: REVIEW_CANDIDATE
This is a research sufficiency gate only; never routes or blocks orders.

## Next feature
Endpoint snapshots are insufficient. Add live order-book persistence over 0–20s and 20–30s:
- L1/depth5/depth20/microprice mean/min/max
- positive/negative fraction
- sign flips
- bullish persistence
- bid/ask depth persistence
No threshold or routing effect until OOS/live sample gate is satisfied.

## Live persistence V44 extension
Runtime patch validated on 2026-10-20 after controlled restart.
The collector now keeps an in-memory depth trace from the existing combined WebSocket; no new API or socket was added.
For 0–20s and 20–30s it emits:
- L1/depth5/depth20/microprice mean/min/max/first/last/delta
- positive/negative persistence fractions
- sign-flip counts
- spread and bid5/ask5 trace statistics
- aggregate bullish/bearish persistence
These fields are TELEMETRY_ONLY and are not passed to classify_20s/classify_30s.

First live trace (18:00 UTC candidate) validated:
- model gate READY, policy NORMAL
- trace 0–20s n=169, bullish persistence ~0.340
- trace 20–30s n=88, bullish persistence ~0.426
- depth5 positive fraction 20–30s ~0.023
- depth20 positive fraction ~0.591
- L1/microprice positive fraction ~0.545 each
- L1 and microprice sign flips = 7; depth5 sign flips = 2
This demonstrates why endpoint snapshots alone can be misleading; persistence distinguishes transient L1/microprice flips from deeper liquidity support.
