# LONG1 + SHORT1 — Exact Binance Order/Trade Read-Only Connector
2026-11-14 UTC. Research only: signed REST GET, no order placement/cancel, no Redis mutation, no service restart.

## New files
- orchestration/risk/binance_exact_readonly_adapter_v1.py:
  - restricted to NEARUSDT/INJUSDT, strict bounded client/order IDs;
  - GET /fapi/v1/order?symbol&origClientOrderId for exact normal order;
  - GET /fapi/v1/userTrades?symbol&orderId&limit=100 for fills, fail closed on >=100 potential truncation;
  - GET positionRisk, openOrders, openAlgoOrders by symbol;
  - GET /fapi/v1/algoOrder?clientAlgoId for exact Algo identity, with symbol/identity check.
- research/tests/test_binance_exact_readonly_adapter_v1.py: PASS 12 allowed GET/strict source and missing/invalid/truncated tests.
- Existing orchestration/risk/bound_exchange_reconcile_v1.py and test (23 cases PASS) used without modification to compose the real GETs.

## Real signed Binance historical smoke test
- NEARUSDT last 100 order history included 9 FILLED BUY entries, and additional FILLED SELL exits.
- First inspected sample was a SELL exit; strict LONG1 entry contract correctly returned UNKNOWN_FREEZE/strategy_symbol_side_mismatch. A SELL exit is NOT a new LONG entry.
- Selected a genuine historical FILLED BUY entry with origQty=8 NEAR. Exact client ID order GET returned the same Binance order ID, one trade fill of 8 NEAR was returned by exact orderId userTrades.
- For read-only integration proof, injected a clearly SYNTHETIC test reservation and synthetic UNKNOWN intent using that real historical entry clientOrderId. The live-read binder correctly classified the condition RECOVERED_FILL_PENDING_JOURNAL, broker_verified_qty=8, can_submit=false, can_commit=false, can_release=false.
- This does NOT establish that the synthetic reservation was owned by the historical LONG1 process. No authoritative historical generation/intent ledger exists for it; simulated ownership is only proof of connector and classifier wiring.
- No current Binance position or stop was created to reproduce the scenario, no live Redis state was changed.
- SHORT1 live mode remains OFF.

## Real production blockers and recommended sequence
1. Need a durable authenticated pre-submit intent+clientOrderId written by ACTUAL future executor, atomic with central reserve generation, not a synthetic test pair.
2. Broker read-only GET calls are individually signed API requests, NOT a signed cross-endpoint atomic snapshot; order status, fills, positions and open Algo may race. Unknown state must freeze and demand a later bounded recheck.
3. The connector currently refuses >=100 fill rows rather than paginating: safety-correct but incomplete for many-fill orders. Add bounded pagination keyed by trade IDs and strict monotonic source timestamps if operationally needed.
4. Exact Algo history status and current stop need cross-check by clientAlgoId; triggered child normal order and fill must be proved separately. EXPIRED alone never demonstrates a stop executed.
5. Central Risk V2 is still isolated test only, existing running central dry service remains unmodified and uses the old reservation protocol. No authenticated broker COMMIT or RELEASE writer. Production account snapshot and drawdown REDUCED state must gate any capital allocation.
6. SHORT1 model promotion, independent whale/liquidity source and production native stop lifecycle are unfinished; do not live-enable.
7. No Redis service, account settings, or collectors started/changed. Protect running LONG1 supervisor/native stop.
