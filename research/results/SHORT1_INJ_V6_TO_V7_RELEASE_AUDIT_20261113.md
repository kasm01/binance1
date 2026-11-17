# SHORT1 INJUSDT — V6-to-Master audit and V7 TP/SL model research
Date: 13 Nov 2026 UTC. Research only. LONG1 unchanged, SHORT1 DRY_RUN=1 ARMED=0. No live SHORT order.

## Existing V6/SHORT1 semantic mismatch
- MasterExecutor's MASTER_MIN_TRADE_SCORE=0.555 applies to a historical model score.
- V6 predicts short_arm4roi_within60m_mae2pct, uses original model threshold ~0.808288.
- Current SHORT1 exit target is +1.25% gross margin ROI with +2.99% underlying adverse price native SL, leverage=3x.
- V6's 4% ROI target is no longer aligned, and its score must NOT be gated using old Master .555 threshold.

## Original V6 model and Master data audit
- SHORT live V6 shadow event contains verified provider, source event ID, completed 3m close, probability and source threshold, flow15/flow60, return15, OI and entry time confirmation.
- It lacks a complete currently fresh Master price/spread/ATR/liquidity/whale context. The old whale shadow and main SHORT signals are ~11 days stale; do not fill missing fields with zero/neutral values.
- 480 recent V6 source events: 424 rejected below their authentic 0.808 threshold; 56 passed score, but blocked because Master inputs are missing. ZERO executable intents authorized.
- Actual SHORT executor model/market stage separately rejects all 480 non-orchestrated shadow events, as expected.
- Files: orchestration/shadow/short1_v6_master_contract_v1.py; research/tests/test_short1_v6_master_contract_v1.py; research/tests/short1_v6_master_480_event_readonly_audit_v1.py.
- CSV: research/results/SHORT1_INJ_V6_MASTER_CONTRACT_480_20261113.csv.

## Fresh V7 TP1.25/SL2.99 research training
- Offline-only script research/short/train_inj_tp125_sl299_v7_research.py trains a NEW separate LogisticRegression model, no updates to V5/V6, no new collector or Binance request.
- Data: existing 220708 feature samples, source INJUSDT_3m_460d.csv covering 10 Jul 2025 through 13 Oct 2026.
- Next 3m open after model signal: TP if SHORT price falls enough for gross +1.25% ROI at 3x; stop if price rises +2.99%; horizon 20 3m bars (60m) with time exit after horizon. Same-bar TP+SL means STOP FIRST, conservative.
- Assumed 0.08% notional round-trip fee plus 0.06% notional round-trip slip = ~0.42% ROI cost at 3x, no funding.
- Chronological train 70% / validation 15% / holdout test 15%; 20-bar embargo. Validation AUC 0.6410, holdout test AUC 0.6235.
- Best among validation tested top-2/5/10/15/20/30% threshold choices was top 5%, score cut ~0.7124; even validation estimated net ROI was NEGATIVE -0.2285% mean.
- Holdout 2344 selected overlapping 3m candidate windows: ~80.89% winners but assumed net mean ROI -0.5664%. These are NOT independent 2344 trades.
- At most one 60m event per 20 bars: 185 non-overlapping selected scenarios: win 81.62%, mean assumed net ROI -0.4349%. Simplified two-outcome breakeven threshold 91.88% with TP1.25 gross, SL8.97 gross, 0.42% ROI costs.
- Model marked research-only, not promoted. Model file research/short/models/short_model_INJUSDT_3m_tp125_sl299_v7_research_only.joblib and result research/short/results/SHORT1_INJ_V7_TP125_SL299_OFFLINE_RESEARCH.json.

## Limits
- Not actual executor replay: 3m instead of sniper 1m fill, no actual Binance fee tier, funding, spread, live latency or dynamic stop trigger.
- Synthetic 60m timeout used for labels, while currently live SHORT1 lifecycle has no corresponding enforced 60m exit. Further lifecycle matching required.
- Labels highly overlap; 185 nonoverlap events better than 2344 overlapping events, still only research.
- Latest Oct13–Nov13 interval is not in model training data; require fresh external holdout before any promotion.
- Different stop/TP ratios should be compared on training/validation first; do not tune on final holdout.

## Next steps
1. Obtain fresh timestamp-consistent INJ market context using existing sources where possible, no neutral/default fake values.
2. Improve SHORT risk/reward via fee-adjusted OHLC worst-case replay with realistic profit locks; avoid trading merely to raise signal count.
3. Only when validation and independent holdout are positive, connect model with faithful target/threshold to isolated DRY_RUN Master->Intent Bridge->SHORT executor.
4. Finish two-bot global live Risk Master, SHORT native stop service and post-fill lifecycle before a separately authorized live SHORT1.
