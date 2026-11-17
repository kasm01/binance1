# SHORT1 INJ — existing V30/V44, OI, V5/V6 time-accurate context audit
14 Nov 2026. Shadow-only; no order, Redis writes, new collector, service restart, LIVE model or parameter change.

## Source evidence
- Active V5/V6 model shadow has source ID, 3m close, actual emission timestamp (~6s into next minute), partial 1m close, ATR14 of 3m features, 15m price return, 15m/60m money flow and OI cached prior timestamp.
- Active V44 public order-book collector emits best bid/ask spread and bid/ask 5-level depth at minute ~20s and ~30s. Active V30 contains paper lifecycle/risk heartbeats, NOT reliable wholesale whale direction or source-specific whale score.
- 480 recent V6 events: causal price, ATR, flows, 15m return and OI 480/480. Current V44 orderbook at model emission ~6s: 0/480 pass <=120s age. Prior published V44 book ~166s old, too stale for execution. Whale direction/score and canonical liquidity score are not present: 0/480.
- At 20s, 431/480 (89.79%) have fresh order-book snapshot published AFTER V6 emission, median availability ~13.746s after model and 120ms quote observation age at publish. Of 68 V6 score passed, 64 have fresh 20s V44.
- No looking into future: published time, orderbook observed_at and original source timestamp all bounded before evaluation time. Never backfill 20s data into 6s earlier signal.
- Actual SHORT1 executor Entry Confirmation against 20s partial 1m snapshots for 68 model-passed: 7 READY, 43 PENDING, 14 REJECT, 4 NOT_EVALUATED. Among seven 20s entry-ready, original 6s V6 confirmations: three CONFIRM and four NOT_DOWN.
- 20s entry-ready is NOT new independent completed 3m model event (required by original PENDING revalidation); no production Master, Central Risk or exchange order authorized. Market score target V6 still old +4% ROI, and V7 target current TP1.25 failed full OOS.
- No whale score inferred from V44 depth imbalance, no liq_score invented. Strict Master intentionally blocks 480/480.

## Tests
- research/tests/test_short1_asof_market_join_v1.py: PASS 10 data-age, future publication, wrong stage/minute, source timestamp, no-order assertions.
- research/tests/test_short1_strict_master_bridge_v1.py: PASS 13.
- research/tests/test_short1_inj_shadow_signal_bus_v1.py: PASS 9.
- Analysis scripts and pure module compile. Existing production Master, SHORT1 live routes and services are unchanged in this turn.

## Next
1. Investigate an actual fresh and sourced SHORT1 whale indicator; do not substitute none or inferred book imbalance.
2. Define and OOS-validate canonical liquidity score using depth and spread, accounting for different order sizes, market liquidity, volatility and execution slippage.
3. Treat V44 20s book as optional additional evidence AFTER source event; PENDING -> READY still needs independent fresh model completed 3m evidence and full stage sequence.
4. Authenticate real Master origin instead of trusting flags. Complete shared production Risk Master, SHORT1 persistent native stop reconciler and dry-run end-to-end before LIVE.
5. Focus LONG1+SHORT1 only; retain old six candidates/settings.

## Artifacts
- orchestration/context/short1_asof_market_join.py
- research/short/audit_short1_asof_market_coverage_v1.py
- research/short/audit_short1_v44_pending20_coverage_v1.py
- research/short/audit_short1_v44_20s_entry_confirm_v1.py
- research/results/SHORT1_INJ_ASOF_MARKET_480_SUMMARY_20261114.json
- research/results/SHORT1_INJ_V44_PENDING20_COVERAGE_20261114.json
- research/results/SHORT1_INJ_V44_20S_ENTRY_CONFIRM_SUMMARY_20261114.json
- research/results/SHORT1_INJ_V44_20S_ENTRY_CONFIRM_EVENTS_20261114.csv
