# SHORT1 INJUSDT — V7 frozen score + real V6 1m snapshot + PENDING + same-coin replay
14 November 2026. ONLY RESEARCH / SHADOW. No SHORT1 live entry, no V7 promotion.

## Context and sources
- Last full frozen out-of-sample V7 (13 Oct–14 Nov) is negative: among 83 60m nonoverlapping research candidates mean net margin ROI -0.3493%; October -0.6339%, November +0.1806%. It cannot be promoted on 8 cherry-picked events.
- Research model V7 was NOT running at original V6 signal time; V7 scores computed retrospectively from already completed 3m bars, same closed bar/time matched to actual V6 and V5 model events.
- Source V6 events contain real emission timestamp ~6s after closed 3m candle, last updated live partial 1m OHLC summary, source model quality and confirmation, completed bar ID, target-specific model probability threshold, and unique event ID. V6 provider is a DIFFERENT target (+4% gross ROI) from frozen V7 (+1.25% ROI /2.99% price SL).
- The last 1,900 V5+V6 matched historical events were read-only replayed by ACTUAL orchestration/shadow/short1_inj_shadow_signal_bus_v1.py process(), with now_ms=max(actual source V5/V6 event ts). No artificial bar-close +5000ms event time, no Redis writes.
- Event states across 1,900 source bars: 1,807 SHADOW_REJECT, 59 SHADOW_PENDING, 23 SHADOW_WAIT, 3 SHADOW_READY_DIRECT, 8 SHADOW_READY_FRESH_MODEL. 11 total shadow READY, 10 also have frozen V7 score >0.712398 from preselected validation cutoff.
- The pure FSM's FRESH_MODEL transition uses actual check_independent_model_revalidation on new completed bar ID, new event ID, model provenance; it is not a broker order.

## Real executor confirmation audit
- For those 10 overlapping V6+V7 candidate events, archived V6 partial 1m snapshot was reconstructed from as-of curr_open, curr_close, curr_range_pct and curr_close_loc. High and low reconstruction was checked to contain current open/close. No finalized current 1m OHLC, so no future partial candle lookahead.
- EXACT core/trade_executor_short.py _evaluate_short_entry_confirmation_stage was called in an isolated mock candle service with existing SHORT 1m thresholds; no Binance order.
- 8 READY (entry_confirmation_ready), 2 PENDING (entry_confirmation_wait). These 2 were previously erroneously counted as shadow READY; real executor has stricter retrace-quality conditions.
- 8 event IDs and reasons: research/results/SHORT1_INJ_V7_V6_REAL_EXECUTOR_CONFIRM_SNAPSHOT_20261114.csv.
- Only these 8 advance to simulated single-coin price path; NONE has passed original Master upstream score gate, live Market Context, Central Risk Master or Final Safety. The V7 model is still RESEARCH ONLY.

## Eight candidates’ 1m price-path scenario
- Actual V6 model emission timestamp -> next FULL 1m opening price (typically 50+sec delayed from original model emission). This does NOT reproduce actual executor at +6sec/2sec polling.
- SL short adverse price 2.99%, gross TP ROI 1.25% at 3x; high/low stop-first on same bar. Assumed 0.08% roundtrip notional fees + 0.06% notional slippage = 0.42% margin ROI.
- In cached 10–14 November 1m data: 8/8 touched the TP within 18 minutes, 0 stop touches before TP. Every candidate could be selected without overlapping its previous 1m close-candle bound exit in the toy single-coin replay.
- Threshold-touch fill assumption yields theoretical +0.83% after assumed cost per candidate; eight ROI percentages SUM +6.64 percentage points (NOT account equity return, no real fills). Do not assume stop/TP exact trigger price can be executed on Binance.
- A different hypothesis (wait until a FULL 1m CLOSE <= TP, exit NEXT 1m OPEN) produces mean +1.18455% assumed ROI. This is NOT conservative guaranteed fill and cannot estimate real app's 2sec polling; favorable next-minute price might not have been available at original touch.
- Source 1m data gives no 2-second mark-price or order-book path. SHORT1 actual lifecycle loop configured POSITION_LIFECYCLE_INTERVAL_SEC=2, price derived from _resolve_order_price, TP giveback=0; stop native uses MARK_PRICE. Without signed mark price/trades and real fill latency, an OHLC 1m replay cannot prove the actual exit ROI. Synthetic 60m time-out is absent in real position lifecycle; 8 cases TP before 30m so time-out is not deciding their sample outcomes.

## Architecture and safety gates
1. MODEL: frozen V7 score analyzed retrospectively, not fresh orchestrated model event.
2. SIGNAL NORMALIZATION/ MARKET CONTEXT: current V6 shadow has 15m/60m flow/price/OI, but fresh whale/spread/liquidity and canonical Master context incomplete.
3. MASTER SCORE GATE: old score threshold 0.555 has different semantics; no valid V7 live score contract or upstream Master authorization.
4. SNIPER: V6 source progress within source window, but actual executor's live timing/risk gate needs separate end-to-end proof.
5. ENTRY CONFIRMATION: actual function isolated and tested against recorded partial 1m snapshot (8 READY, 2 PENDING).
6. PENDING REVALIDATION: pure FSM independent-new-model source evidence (8 fresh READY among 1,900), but no canonical production intent or 8-to-2 actual broker transitions.
7. EXECUTION RISK/FINAL SAFETY: SHORT1 still DRY_RUN=1, ARMED=0, LIVE production risk master not authorized.
8. ORDER/POSITION: no SHORT1 live order, no persistent SHORT1 native Algo reconciler service; requires atomic shared Risk Master and full stop protection/close reconciliation before separately authorizing SHORT1.

## Files
- research/short/audit_inj_v7_v6_confirmation_overlap_v1.py
- research/short/audit_inj_v7_shadow_pending_causal_replay_v1.py
- research/short/audit_inj_v7_v6_executor_entry_confirm_v1.py
- research/short/audit_inj_eight_ready_single_coin_path_v1.py
- research/short/audit_inj_live_v6_tp_touch_vs_close_v1.py
- research/results/SHORT1_INJ_V7_V6_REAL_CONFIRMATION_OVERLAP_SUMMARY_20261114.json
- research/results/SHORT1_INJ_V7_SHADOW_PENDING_CAUSAL_REPLAY_SUMMARY_20261114.json
- research/results/SHORT1_INJ_V7_V6_REAL_EXECUTOR_CONFIRM_SNAPSHOT_SUMMARY_20261114.json
- research/results/SHORT1_INJ_8_EXECUTOR_READY_SINGLE_COIN_SUMMARY_20261114.json
- research/results/SHORT1_INJ_10_REAL_SHADOW_TP_TOUCH_VS_1M_CLOSE_SUMMARY_20261114.json

## Next action
Do not promote V7 to production. Build a canonical shadow-only admission/deny trace that runs actual Market Context and Master score contract without spoofing missing fields, then a full 1m or tick/mark-price lifecycle replay on an independent later period with real Binance fees. Confirm stop service/reconciler and atomic Risk Master before live SHORT.
