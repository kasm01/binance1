# LONG1 + SHORT1 — Bound Exchange Intent, Fills and Native Stop Reconciliation
14 Nov 2026 UTC — Research-only. No changes to running LONG1 or SHORT1, central dry risk, production Redis or Binance orders.

## Existing actual architecture
- LONG1 NEARUSDT active supervisor and native stop; SHORT1 INJUSDT DRY_RUN=1, ARMED=0; central risk dry service running.
- Previous independent signed Binance account/positions/orders/algo/filled trade queries confirmed exact client ID lookups and historical LONG1 fills. Current stage does NOT reissue actual Binance orders or alter any production state.
- Existing experimental atomic Lua RESERVE+PREPARE binds request_id, generation token, strategy, symbol, side, client ID in a single isolated Redis script with AOF and run_id fencing.

## New evidence binding
- orchestration/risk/bound_exchange_reconcile_v1.py:
  - validates reservation request_id, generation, strategy, symbol, owner and immutable ledger client_order_id.
  - composes exact order status lookup by client ID, per-order ID fill evidence and current broker position, open normal orders and Algo orders as one bounded read-only observation.
  - missing/malformed, mismatched, timeout or incomplete order/fills/broker source => UNKNOWN_FREEZE; no resubmission.
  - fully filled broker order while ledger SUBMITTING/UNKNOWN => RECOVERED_FILL_PENDING_JOURNAL; require authenticated ledger repair, NEVER repeat order.
  - fully matched fill and broker position, correct-side owned MARK_PRICE STOP_MARKET Algo => PROTECTED_EVIDENCE_ONLY.
  - missing/foreign stop => FREEZE_UNPROTECTED.
  - flat broker position with open normal order/Algo or missing matched exit identity => HOLD_RECONCILIATION.
  - fully consistent flat/order history => FLAT_EVIDENCE_ONLY, but no auto release. Source data merely evidence; no authenticated production proof.
  - all results can_submit=false, can_commit=false, can_release=false, independent of success.
- research/tests/test_bound_exchange_reconcile_v1.py: PASS 23 adversarial cases (order timeout, wrong order/client identity, wrong reservation generation, partial fill, open other orders, non-matching positions, missing/foreign stop, multiple unavailable broker sources, zero fill terminal and full filled, no automatic release).
- research/tests/test_bound_reconcile_atomic_reservation_isolated_v1.py: PASS single isolated Redis atomic reservation and PREPARED ledger plus mocked broker order/trades/position/algo. Simulated UNKNOWN and confirmed full fill, correct stop, missing stop, flat without exit. Redis reservation remains held and ledger unchanged; no production write.
- Independent current snapshot check from previous stage: Binance/central mtm equity 62.66814695 USDT, no positions, central risk mode REDUCED with drawdown approximately -7.96%. These values are historical point-in-time observations and not a current trading authorization.

## Security limitations (production blockers)
1. Test read_order/read_fills/read_positions/read_open_orders/read_open_algo callbacks were all MOCKS. This is not yet one authenticated Binance reconciler that signs and verifies the expected per-intent snapshots and emits a verifiable authorization.
2. Read-only multiple REST calls are not atomic. An UNKNOWN POST outcome or exchange lag must keep the central entry gate frozen until order history, fills, active positions and all normal/algo orders agree across a trusted reconciliation epoch.
3. A fill seen at Binance while ledger UNKNOWN cannot be automatically written to Redis by arbitrary inputs. Requires authorized reconciliation writer, durable transaction generation, and exact clientOrderId + reservation ownership.
4. Native stop must be bound to position generation and exposure quantity. Algo expiry, stop cancellation, triggered child order and manual/outside-bot position close need a separate audited state transition.
5. Current experimental Lua RESERVE+PREPARE still takes caller-supplied equity and policy/test reconciliation strings; it is NOT production source authentication nor real shared MTM risk authority. Existing central production dry service is NOT replaced.
6. Partial-fill entry status OPEN_OR_PARTIAL is currently categorized HOLD_RECONCILIATION by this stricter cross-source evidence binder. It does not confirm exact partial fill history; never assign PROTECTED based only on partial order status.
7. No code or services created for LONG2-4/SHORT2-4. LONG1 live code unchanged, SHORT1 model remains unpromoted and LIVE disabled.

## Next safe step
- Create an independently restricted read-only broker evidence worker and isolated state journal writer, not caller-supplied proof flags, with signed Binance REST exact order+trade/Algo and central snapshot binding by generation and causal timestamps.
- Test ambiguous submit, partial then full fill, native stop failure/expired stops, repeat ACK, all orders empty but unseen in-flight client ID, REDUCED mode sizing and risk freeze across Redis/worker restart.
- Only afterward consider pilot dry-run integration. Never enable SHORT1 LIVE or migrate LONG1 live risk until all safety components including native stop and model provenance validate.
