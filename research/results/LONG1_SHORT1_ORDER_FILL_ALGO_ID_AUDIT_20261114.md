# LONG1 + SHORT1 — Historical order, fill and Algo reconciliation
14 Nov 2026 UTC. Read-only signed Binance GET only, no order or Redis writes, no production Risk Master change.

## Bounded sources
- GET /fapi/v1/allOrders, /fapi/v1/userTrades, /fapi/v1/allAlgoOrders by NEARUSDT/INJUSDT, limit=100.
- Existing signed core/binance_futures_v3.py; no bulk download or new collector.
- Sample bounded by Binance endpoint historical retention; not proof of complete lifetime trading events.
- Research output uses hashes of exchange order IDs and client IDs, no raw IDs or API credentials persisted.

## Actual LONG1 NEARUSDT
- 18 historical orders, all FILLED; 24 matched trade fills.
- Per-order sum of matched fill qty exactly matched executedQty and origQty for all 18 orders; zero orphan trade order IDs in this queried sample.
- Two historical conditional STOP_MARKET SELL/LONG Algo orders marked EXPIRED. Algo actualOrderId absent/unlinked; neither status proves stop-triggered exit.
- Each expired Algo update timestamp was within 30s of a FILLED normal SELL. Time correlation does not prove causality or protection.
- Commissions are denominated in BNB and must not be deducted numerically as USDT without per-fill historical conversion.
- This audit does NOT certify broker-flat or durable order lifecycle across time.

## SHORT1 INJUSDT
- Queried recent order/trade/Algo lists all empty (0/0/0). SHORT1 DRY_RUN=1 and ARMED=0.

## Safety
- New orchestration/risk/binance_order_fill_evidence_v1.py classifies matched fills, terminal orders and incomplete fills.
- Read-only history result is ORDER_FILL_MATCHED_SAMPLE for NEARUSDT; safe_to_auto_release=False and risk_master_can_commit=False.
- INJUSDT result INCOMPLETE_OR_EMPTY_HISTORY; empty history does not prove release or live authorization.
- A POST timeout may leave an accepted order unknown. Risk must be held until exact client/order ID status and position/normal order/Algo state reconciliation.
- Expired Algo Stop cannot authorize close; triggered child status requires actual matched normal order and fills.
- V2 Redis research Lua currently uses a caller-supplied flat proof text. This is not an authenticated broker reconciliation token and MUST NOT be deployed.
- Existing LONG1 and SHORT1 services unchanged. The central dry service remains production_authorized=False.

## Test and report files
- research/tests/test_binance_order_fill_evidence_v1.py: PASS 12 edge cases.
- research/tests/test_exchange_flat_readonly_reconcile_v1.py: PASS 15 cases from prior stage.
- research/short/audit_long1_short1_order_fill_readonly_v1.py
- research/results/LONG1_SHORT1_HISTORICAL_FILL_LEDGER_READONLY_20261114.json

## Next
1. Exact clientId/orderId query with GET /fapi/v1/order or /fapi/v1/algoOrder using actual local-owned ID, read-only.
2. Durable intent-order-fill ledger binding event ID, reservation generation, client order ID, Binance order ID, position ID and stop ID.
3. Simulate ambiguous submit, partial fill, duplicate ack, stop expiration and Redis restart while exchange has active position; UNKNOWN => freeze.
4. Service-side authenticated or ACL-bound flat/commit evidence, no caller-supplied flags. Isolated dry-run only until all gates validate.

## Exact order ID signed GET smoke check, completed
- From one actual recent NEARUSDT FILLED normal order retrieved independently, signed GET /fapi/v1/order by origClientOrderId returned same orderId and FILLED; by orderId returned the same. IDs were not logged in this report.
- From one actual historical NEARUSDT EXPIRED conditional Algo record, signed GET /fapi/v1/algoOrder by clientAlgoId returned matching algoId and EXPIRED; by algoId returned the same.
- This confirms exact normal and conditional Algo lookup endpoints work for this account. It does not make an unknown original POST response safe or prove that expired Algo triggered.
- The production intent-to-clientOrderId persistent mapping and uncertain-outcome recovery/ownership are not implemented; release stays forbidden.
