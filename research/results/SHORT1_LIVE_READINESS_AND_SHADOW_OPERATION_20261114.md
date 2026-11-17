# SHORT1 INJUSDT — Running Shadow / Live Readiness Audit
14 Nov 2026 UTC, read-only audit and existing deterministic Master gate replay; no live or production config changes.

## User priorities
- LONG1 is deliberately unchanged: no new entry loosening, stop, profit-lock or position lifecycle modifications.
- SHORT1 must be inspected first, with a later joint LONG1+SHORT1 detailed audit.
- Do not treat shadow READY as executable, and do not bypass market context, native stop, or central risk.

## Actual observed SHORT1 operations
- systemd --user binance1-short1-inj-shadow-signal-bus.service ACTIVE (shadow-only).
- INJ model V5/V6 Redis telemetry and shadow bus received records within about 30 seconds at read time.
- Redis stream counts at audit: V5 7048 events, V6 5426 events, shadow bus 346 events.
- Last 345 shadow-bus decisions: 324 SHADOW_REJECT, 9 SHADOW_WAIT, 11 SHADOW_PENDING, 1 SHADOW_READY_FRESH_MODEL. These are observation-only events, not live entry approval.
- Stateless recent 480 closed-bar replay: 409 REJECT, 49 PENDING, 14 WAIT, 1 READY_DIRECT, 7 READY_FRESH_MODEL; 7 distinct fresh READY intents. Pure replay does not submit orders.
- Whale shadow last update about 11.7 days ago; telemetry:market_context_shadow stream empty when checked. New independent canonical Market Context (price, spread_pct, atr_pct, liq_score, whale_dir, whale_score) absent.
- Signed Binance GET read-only INJUSDT: zero nonzero positions, zero open regular orders, zero open Algo orders at time of check.
- Env: DRY_RUN=1 ARMED=0 LIVE_KILL_SWITCH=1, SHORT_NATIVE_STOP_ENABLE=1, SHORT_MARGIN_TYPE=ISOLATED, SHORT_REQUIRE_HEDGE_MODE=1, MAX_OPEN_POSITIONS=1, MAX_LEVERAGE=3, SHORT_NORMAL_POSITION_EQUITY_PCT=.46 and SHORT_REDUCED_POSITION_EQUITY_PCT=.23.
- SHORT native-stop flag in config does not establish presence or functionality of a live standalone native-stop reconciler; no matching SHORT native-stop systemd unit found.
- LONG1 supervisor and native stop services remained active, not restarted or edited.

## Actual entry-stage trace
- Re-ran research/short/audit_short1_master_fail_closed_trace_v1.py against eight recorded SHORT1 events that had passed retrospective V7 research score and isolated 1m entry confirmation.
- strict context contract BLOCK_CONTEXT 8/8; missing required fields: atr_pct, liq_score, price, spread_pct, whale_dir, whale_score.
- actual executor score gate REJECT 8/8, reason upstream_model_score_gate_required.
- 0 execution-risk stages reached; 0 final safety stages passed; 0 exchange orders authorized.
- Source data remained non-orchestrated. Did not inject fake model score, market context, pass flags or broker proof.
- Analysis outputs research/results/SHORT1_INJ_8_REAL_READY_FAIL_CLOSED_MASTER_TRACE_20261114.csv and _SUMMARY_20261114.json.

## Model validation hard stop
- V6 short_arm_quality_logistic_v6 predicts a different higher-ROI target (4% gross) from current SHORT1 exit plan (+1.25% gross ROI, adverse SL 2.99% price); cannot use Master historical 0.555 gate on V6 model score.
- V7 research-only, NOT PROMOTED. Frozen 13 Oct–14 Nov independent period among 83 nonoverlapping theoretical trades had mean estimated net ROI -0.3493% (October -0.6339%, November +0.1806%). Fee, stop, latency and timeout assumptions differ from production. This is not an approved model.
- Eight attractive retrospective 1m price-path candidates are selected conditional on historical scores; cannot supersede negative broader validation. User priority daily return means fees and the severe win/loss asymmetry matter more than raw signal count.

## Decision / safe next implementation sequence
1. Keep active INJ shadow bus running. Leave DRY_RUN=1 ARMED=0 LIVE_KILL_SWITCH=1. No live SHORT order.
2. Repair canonical real-time price/spread/ATR/liquidity/whale as-of context from independent existing sources, with freshness and completeness checks; do NOT substitute stale whale data or zero-valued default fields.
3. Develop and freeze a SHORT1 model whose label matches actual 1m entry, 3x, 1.25% ROI TP/giveback, 2.99% price SL, Binance fee+slippage, and real time exit policy. Validate walk-forward OOS before promotion.
4. Dry-run actual Master Score Gate and final safety with genuine current V7 or newly trained model event; strict source provenance, frozen threshold and no fake READY.
5. Verify isolated margin/hedge-mode compatibility and native MARK_PRICE Algo STOP attach, cancel/retry, crash/ambiguous fill, generation-bound stop ownership and robust flat release with actual central risk. Production Risk Master V3 not authorized.
6. Only after all gates pass, plan separately authorized SHORT1 live pilot with conservative notional sized to current REDUCED central mode and strict one-coin, one-position limit. No layering.
7. Inspect LONG1 entry loosening only after SHORT1 readiness, with shadow/replay evidence; do not weaken existing contra-trend guard on a live bot without proof.

## Resource constraints
- Existing Google Cloud VM disk ~91% full; no always-on new collectors/services, no heavy duplicate downloads.
- Signed Binance GET only for market account state; research replay read-only; no Redis live writes except routine already-running services.
