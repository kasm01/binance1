# SHORT1 INJUSDT: Fresh MODEL PENDING revalidation + live shadow audit
Date 13 Nov 2026 UTC. This is a research/staged-code result, not a production strategy profit validation.

## Code changes
- core/trade_executor_short.py: old PENDING scores never re-open at next 1m Sniper window. With SHORT_PENDING_REQUIRE_NEW_MODEL_PROOF=1 old pending remains up to 15m awaiting a distinct NEW upstream hard-READY intent, then expires.
- New _resolve_short_pending_by_fresh_model() validates origin model event ID and intent ID against independent new signal event ID and new intent ID; requires same provider, strictly newer completed 3m source close timestamp, <=5min freshness, model score >= new authentic upstream threshold, passed score gate. Valid new intent POPS old PENDING, then executes NORMAL Sniper->Entry Confirmation->Execution Risk->Final Safety path; old cached score is never granted authority to order.
- core/short_pending_freshness_contract.py validates all required causal and score fields including event IDs.
- Signal provenance from orchestration/aggregator/aggregator.py -> orchestration/executor/master_executor.py -> orchestration/executor/intent_bridge.py already staged in previous step; mocked complete pipeline passed event ID and source-close-time fields.
- Existing NEW short direct readiness and API stop guards retained. SHORT1 remains DRY_RUN=1, ARMED=0. SHORT main not started; no Binance order submitted. Existing LONG1 kept running.

## Historical INJ native intent evidence
- Redis exec_events_stream_short1: 205 INJ SHORT execution intents in 25–31 October period: 163 READY and 42 PENDING_CONTEXT.
- Legacy historical records did not carry the explicit completed source-close and signal-ID proof; cannot genuinely retroactively assert that a pending was later authorized or filled.
- 5m INJUSDT public Binance Futures history Oct25–Nov1 collected with exactly TWO bounded read-only paginated GET requests, 2304 contiguous completed 5m candles. Source: research/short/data/INJUSDT_5m_20261025_20261102_PENDING_REPLAY.csv (~1day? actual 8 days).
- Exploratory pairing: 42 historical PENDING events; 26 candidate PENDING->later READY pairings within 15m, but these share 12 distinct later READY intents! Corrected to **12 unique READY** by associating each READY to its most recent earlier PENDING. No independent source-event-ID proof exists in these old records.
- 12 unique-event sample, SHORT direction 5m close-markout from most recent FULLY CLOSED candle at event time:
  15m original PENDING mean -0.3973% underlying price vs later READY +0.0683%; 7/12 later positive;
  30m original -0.3510% vs later +0.2139%; 7/12 later positive;
  60m original -0.2989% vs later +0.0198%; 5/12 later positive.
- These are historical *price-direction proxy markouts*, NOT order fills, fee-net PnL, exact model revalidation, native Sniper window success, take-profit or full portfolio backtest. 12 independent READY events are too few to infer edge.
- Among 12 new READY proxy events, only 2 meet additional 5m bearish and both 15m/60m taker flow negative; making those additional features a HARD veto could reject 10/12 tentative setups and some favorable opportunities. Keep 5m+flow as shadow advisory for now, do not hard-code a new threshold.
- Replay CSVs:
  research/results/SHORT1_INJ_PENDING_TO_FRESH_READY_5M_PROXY_20261113.csv (26 correlated pairs; do NOT interpret 26 as independent trades)
  research/results/SHORT1_INJ_PENDING_TO_FRESH_READY_ONE_TO_ONE_20261113.csv (12 one-to-one unique READY)
  research/results/SHORT1_INJ_NEW_READY_CLOSED_5M_CONTEXT_20261113.csv (12 completed 5m contexts)

## Live incoming model telemetry / missing normal routing
- SHORT economic model shadow V5 and V6 both active and have current Redis telemetry, 480 completed INJUSDT 3m bars EACH within last 24h.
- V5 target short-entry quality fresh logistic, reference threshold ~0.521; V6 target short 4% ROI within60m with max2% adverse price, threshold ~0.808. These are different target semantics. Do not combine scores as if equivalent probabilities.
- Of 480 aligned completed 3m bars, V5 passed score threshold 233, V6 56, both passed 56; only 6 had V5+V6 threshold passes together with both Entry Confirmation states CONFIRM. This is an observational funnel, NOT validated native orders or future profits.
- Both V5 & V6 live shadow outputs have routing OFF. Latest signals_stream_short1 was ~263.5h old, exec_events_stream_short1 ~315.8h old, whereas V5/V6 telemetry active NOW. SHORT main/executor not running. This means there is NO currently operating end-to-end SHORT native signal path for the new fresh-ID gate.
- Do NOT auto-route shadow model events to live execution: requires an explicit dedicated SHORT1 dry-run signal bus adapter with model version selection, feature hash / timestamp / completed 3m proof, de-duplication, model + Sniper + Entry Confirmation + final risk validation.
- Research agreement CSV: research/results/SHORT1_INJ_V5_V6_24H_NONROUTING_AGREEMENT_20261113.csv.

## Required next steps
1. Choose candidate SHORT model (V5 and V6 have different targets); construct NONROUTING or explicitly DRY_RUN signal adapter and feed an isolated replay stream (never the current live trade exec stream), then test new intent correlation and time-window rules with real incoming events.
2. Re-evaluate economic edge on enough UNIQUE matched signals, with actual entry slippage and fee-net USDT exit logic; test scenario sensitivities and out-of-sample holdout rather than optimizing on these 12.
3. Complete SHORT1 native stop after filled exchange price and separate boot-resume reconciliation service once production-approved.
4. Global 8-bot Risk Master caps across distinct coins and exactly-once reservation/accounting, plus SHORT1 dedicated PID/log/Redis/service isolation before live promotion.

## Safety checks
- DRY_RUN=1, ARMED=0, SYMBOL=INJUSDT, SHORT_PENDING_REQUIRE_NEW_MODEL_PROOF=1.
- No SHORT1 main service restarted, no Binance order, no actual Algo stop created for a flat INJ short; LONG1 live service remains active.
- Python compile and unit tests passed (full test suite described in final response).
- VM disk ~91% used; fetched only fixed 5m historical range, no repeated pollers launched.
