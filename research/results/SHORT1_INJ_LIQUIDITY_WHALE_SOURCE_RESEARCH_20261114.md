# SHORT1 INJUSDT — V44 liquidity and WhaleDetector source research
14 Nov 2026 UTC. Research only. No exchange order, Redis write, collector deployment, environment/model promotion or service restart.

## V44 fresh book and notional capacity
- Read-only V44 latest 2400 finalized outcome events; 2397 valid source-time verified 20-second book observations.
- Spread median 1.34671 bps, p10 1.24293, p90 1.40066 bps. Quote age at observed time median 119ms.
- SHORT market SELL consumes bids. Median best-bid L1 notional ~678.12 USDT; median aggregated best-five bid quantity times best-bid price ~20235.86 USDT, but this aggregated quantity does NOT encode individual limit prices or executable VWAP.
- L1 best bid can cover 30 USDT in 2319/2397 (96.75%); 100 USDT in 2240/2397 (93.45%); 500 USDT in 1559/2397 (65.04%). Historical and not a guarantee of fill/real slippage.
- Depth-5 cover 100% for the three notional sizes; these levels are not guaranteed at best-bid price.
- Future 1m/3m/5m adverse markout splits by 100 USDT L1-cover flag did NOT yield validated causal execution slip. Real Binance fill, latency and fee records still required.
- Files: research/results/SHORT1_INJ_V44_20S_LIQUIDITY_CAPACITY_20261114.csv and SHORT1_INJ_V44_20S_LIQUIDITY_CAPACITY_SUMMARY_20261114.json.

## Existing WhaleDetector cannot become Master whale oracle
- core/whale_detector.py already has price-and-volume OHLCV pattern-derived direction/score, not sourced actual large trader identities.
- orchestration/scanners/model_worker.py optionally writes whale shadow, but current INJ V5/V6 shadow services run different scripts. Existing whale_shadow stream stale around 11 days.
- Read-only source-completed-3m-bar as-of WhaleDetector evaluation matched 459 of 480 recent V6 events within available archived candles: 242 short-direction patterns, 213 long-direction, 4 none, median score 0.2037, 355 ignore vote. This is a correlated OHLCV proxy, NOT independent whale validation.
- Files: research/results/SHORT1_INJ_WHALE_OHLCV_PROXY_480_20261114.csv and SHORT1_INJ_WHALE_OHLCV_PROXY_480_SUMMARY_20261114.json.

## New research modules and safety
- orchestration/context/short1_liquidity_evidence.py calculates as-of bid-side L1/L5 observed coverage only. Returns liq_score=None and can_authorize=False, never fabricates VWAP or real execution slippage.
- orchestration/context/short1_asof_market_join.py previously added, prevents 20s V44 data being used retroactively at 6s V6 model event.
- Test groups: test_short1_liquidity_evidence_v1.py PASS 10; test_short1_asof_market_join_v1.py PASS 10; test_short1_strict_master_bridge_v1.py PASS 13.
- SHORT1 DRY_RUN=1 ARMED=0. Production V7 promotion gate OFF. LONG1 live unchanged. SHORT1 Master remains fail closed due missing canonical independent whale and liquidity score plus model unapproved.

## Next
1. Do NOT turn spread/depth into arbitrary canonical liq_score. Test actual live fill/slippage and fee telemetry first, with size, liquidity, volatility and delay buckets.
2. Keep OHLCV WhaleDetector shadow as heuristic only. Independent whale source remains unresolved, not neutral.
3. Test secure Master source origin and single-coin Risk Master atomic reservations/stop lifecycle in isolated dry-run before production.
4. No additional collectors unless needed after cost and source verification; prioritize LONG1+SHORT1 only.
