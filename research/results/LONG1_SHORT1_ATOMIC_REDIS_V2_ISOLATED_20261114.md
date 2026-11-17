# LONG1 + SHORT1 central risk V2 isolated Redis atomicity test
14 Nov 2026 UTC — RESEARCH ONLY. No service restart, no production Redis mutation, no live orders.

## Scope and test isolation
- New research module: orchestration/risk/central_reservation_redis_v2_research.py
- New test: research/tests/test_central_risk_redis_v2_isolated_v1.py
- Spawned temporary Redis 7.0.15 child process with port=0, unixsocket only under TemporaryDirectory, restricted permissions 0700, no RDB, no AOF, maxmemory 24 MB. Process and socket were torn down at end.
- Real central Redis host/port 6379 was NOT used by V2 test. LONG1 and SHORT1 production services were NOT restarted or modified.
- Atomic Lua EVAL implements reserve ownership check, idempotent request-id response, exposure math and reservation insert in one server transaction. COMMIT and RELEASE require exact request-id/reservation-id with position-id for committed states.

## Confirmed PASS cases on isolated Redis
- New or restarted Redis gate starts CLOSED, rejects RESERVE until explicit isolated test reconciliation marker.
- Example equity=100 USDT: normal LONG1 NEARUSDT 46 USDT notional plus SHORT1 INJUSDT 46 USDT notional = 92 total under 100 combined gross cap (research budget only).
- 60 concurrent SHORT1 requests with distinct IDs across 20 Python worker threads -> exactly ONE READY, 59 denied by occupied slot. All calls land on Redis Lua atomic operation.
- Same request-id + same payload repeated -> exactly same response. Same ID with changed owner or token -> idempotency conflict.
- Wrong owner RELEASE denied. Unverified/flat-unknown RELEASE denied, both pending and committed.
- COMMIT moved matching pending reservation to committed position record; repeat with same ID/position returned already_committed, wrong owner denied.
- Old RELEASE from A cannot delete B reservation after A has been released and B reserved.
- Expired unconfirmed reservation stays blocked. No automatic expiry/timeout clearing position risk.
- Cannot open gate while unresolved reservations exist.
- Isolated Redis server shutdown => client receives Redis connection error (no implicit READY/BYPASS), and after isolated restart new Redis has empty hashes but CLOSED gate until reconciliation.

## IMPORTANT limitations
- This is NOT the running central risk server: /home/kasm920/binance1-pro/scripts/central_risk_enforce_dry_service.py remains unmodified, ENFORCE_DRY_RUN with production_authorized=False.
- The test version accepts caller-supplied strings ISOLATED_TEST_RECONCILED and EXCHANGE_FLAT_AND_NO_ORDERS to simulate broker verification. They are NOT authentic evidence, NOT safe for production, and MUST be replaced with a trusted exchange reconciliation authority before deployment.
- A hostile producer with access to the test Redis keys could mutate gate or forge synthetic checks. No authenticated service identity, write ACL, cryptographic intent signature or independent reconciliation in this prototype.
- The test Redis uses NO persistence. Restart loses reservations, but closes the gate. Production needs a durable journal/fencing epoch, recovery from Binance positions and ALL order classes including Algo; unknown exposure freezes entries.
- COMMIT currently accepts caller-supplied position_id, not a signed exchange fill. Proper broker fill verification and requested vs actual filled notional remain future work.
- Equity is a test argument. Production must derive authoritative shared marked equity and reserved margin from signed account snapshot, not trust caller input.
- No MTM portfolio DD updates, margin availability, funding, fee reserves, exchange instrument filters, single-flight process leases or multi-VM partitions tested. 60-thread one-Redis atomicity != complete production fault tolerance.
- LONG1 live is not yet coordinated by a production shared Risk Master. SHORT1 DRY_RUN=1 and ARMED=0.

## Next safe implementation gates
1. Create trusted signed Binance account/positions/open orders and Algo orders reconciler for LONG1+SHORT1 in read-only mode, with UNKNOWN=>freeze.
2. Define service authentication plus restricted Redis ACLs and model/intent source verification, no caller-provided privileged OPEN_GATE/flat flags.
3. Durable reservation WAL, generation/epoch fencing and request-id tombstones surviving Redis restart.
4. Broker-real fill COMMIT, partial-fill and uncertain order outcome tests; exact marked-notional and margin/fees reconciliation.
5. Separate isolated risk service and load testing against a second isolated Redis with forced process crashes/clock skew; only then evaluate deployment to running dry-run service.
6. Do not change LONG1 live or enable SHORT1 until accepted by complete risk, model score, Market Context, Final Safety and native stop controls.

## Verification
- research/tests/test_central_risk_redis_v2_isolated_v1.py: PASS.
- research/tests/test_long1_short1_central_reservation_protocol_v2.py: previously PASS 17.
- Last check LONG1 supervisor, LONG1 native stop, SHORT1 shadow bus and current central dry gate services ACTIVE.
- SHORT1 .env DRY_RUN=1 ARMED=0 SYMBOL=INJUSDT.
