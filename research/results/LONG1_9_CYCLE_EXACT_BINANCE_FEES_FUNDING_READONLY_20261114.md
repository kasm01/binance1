# LONG1 NEARUSDT — 9 Historical Closed Cycles and Exact Exchange Fee/Funding Audit
14 Nov 2026 UTC. Binance USD-M signed REST GET only. No production Redis writes, orders, cancels, service or .env changes.

## Bounded exchange sources (NOT full lifetime history)
- GET /fapi/v1/allOrders?symbol=NEARUSDT&limit=100: 18 FILLED, precisely nine BUY non-reduce-only LONG entries alternating with nine reduceOnly SELL LONG exits.
- GET /fapi/v1/userTrades?symbol=NEARUSDT&limit=100: 24 exact order-matched fills from 2026-11-11 12:17:18 UTC to 2026-11-14 03:36:23 UTC.
- GET /fapi/v1/income?incomeType=COMMISSION&startTime..&endTime..&limit=100: 24 rows, all NEARUSDT, all BNB, exactly 24/24 joined via tradeId. Each commission income equals negative fill BNB commission; total +0.00034955 BNB fee paid.
- GET /fapi/v1/income?symbol=NEARUSDT&incomeType=FUNDING_FEE&startTime..&endTime..&limit=100: one income -0.00280688 USDT on 13 Nov 16:00 UTC, unambiguously falls within cycle 8 (13 Nov 14:51–19:28 UTC) in the queried subset.
- No raw order IDs, client IDs, trade IDs, keys or account secrets persisted. Report JSON includes timestamps, quantities, aggregate costs and numbered cycles only.

## Historic cycle gross realized PnL before BNB commission
1. Nov 11 12:17–13:14 UTC: +0.03200000 USDT (8 NEAR), BNB fee 0.00003967.
2. Nov 12 00:06–00:23 UTC: +0.11200000 USDT (8), BNB fee 0.00003799.
3. Nov 12 01:12–02:06 UTC: +0.12800000 USDT (8), BNB fee 0.00003819.
4. Nov 12 14:13–14:14 UTC: +0.14400000 USDT (9), BNB fee 0.00004165.
5. Nov 12 14:45–14:51 UTC: +0.12000000 USDT (8), BNB fee 0.00003746.
6. Nov 12 17:05–18:33 UTC: +0.12000000 USDT (8), BNB fee 0.00003742.
7. Nov 13 02:42–03:52 UTC: -0.87200000 USDT (8), BNB fee 0.00003788.
8. Nov 13 14:51–19:28 UTC: -0.12000000 USDT (8), funding -0.00280688 USDT, BNB fee 0.00003995.
9. Nov 14 03:34–03:36 UTC: +0.14400000 USDT (8), BNB fee 0.00003934.
- Seven positive gross closes, two negative gross closes. Gross realized PnL = -0.19200000 USDT.
- Gross after the known USDT funding = -0.19480688 USDT. This is an UPPER BOUND on net PnL assuming nonnegative BNB/USDT conversion and no omitted costs. Thus this historical sample is net losing even before valuing BNB fees. It does NOT establish all-time trading performance.
- Total BNB commission = 0.00034955 BNB. No exact historical BNB/USDT fee conversion verified, so final USDT net PnL and per-cycle net-win count are UNKNOWN, not estimated.

## Implemented
- orchestration/risk/historical_position_cycle_accounting_v1.py:
  - exact ordered, quantity-matched entry/exit LONG full cycles, strict positionSide/reduceOnly,
    unique trade IDs, 1:1 income tradeId/asset/value/time match,
    funding attributed once to one open cycle, fail-closed on absent/duplicate/fuzzy sources.
  - full sample at API limit 100 is rejected as potentially truncated; these 18/24/24/1 bounded results are not near that limit but still NOT proven exhaustive beyond sampled time range.
  - no flat generation/stop/position ownership proof, no authenticated pre-submit intent or safety authorization.
- research/tests/test_historical_position_cycle_accounting_v1.py: PASS 18 synthetic adversarial assertions.
- research/results/LONG1_HISTORICAL_ACCOUNTING_READONLY_20261114.json: exact numerical results (no order/client/trade IDs) of the live read-only sample.

## Current production blockers / next work
1. BNB commission conversion requires actual fill-time BNB/USDT rate with grounded time/price source and correct multi-fill valuations; do NOT silently use current prices or equate BNB units to USDT.
2. Long1 historical orders do not have verified shared V3 reservation/intent generations or opened_risk_mode. Cannot count any old trade as recovery win, cannot COMMIT/RELEASE.
3. This module only supports discrete fully matched BUY/SELL pairs with no overlapping positions, partial remaining exposure, manual intervention or Binance stop child orders. Such cases must be classified UNKNOWN until reconciled.
4. Funding outside the limited requested time window, broader account fees, and external BNB inventory valuation are not independently proven complete. Multiple non-atomic Binance REST reads can race; no production clearance from sample.
5. SHORT1 remains DRY_RUN=1 ARMED=0; LONG1 live supervisor/native stop, central dry risk active; no live changes. No collector or permanent service added (VM disk ~91% used).

## Financial observation
- Gross win rate 7/9 (~77.8%) hides net negative aggregate. The -0.872 USDT loss in cycle 7 dominates several small gross profits. Adjusting a stop or profit-lock parameter solely from this 9-trade sample would be statistically premature; larger forward and replay evidence, fee-converted net PnL and market-context entry analysis are required.
