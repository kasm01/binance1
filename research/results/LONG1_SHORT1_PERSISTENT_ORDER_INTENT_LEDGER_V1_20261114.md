# LONG1 + SHORT1 — Durable Intent/Order/Fill Ledger V1
14 Nov 2026 UTC. Research-only. Live risk service and order execution unchanged.

## New isolated components
- orchestration/risk/order_intent_ledger_v1_research.py: atomic Redis Lua intent ledger with immutable owner/reservation_id/clientOrderId binding, uniqueness and staged PREPARE -> SUBMITTING -> UNKNOWN or ACKED -> PARTIAL/FILLED -> FLAT_RECONCILED.
- research/tests/test_order_intent_ledger_persistent_redis_v1.py: temporary isolated Redis 7 Unix socket port=0, appendonly=yes and appendfsync=always, no RDB, removed at end. AOF retains UNKNOWN intent and original identity after Redis restart.
- Redis Lua compares persisted OPEN state with current Redis server run_id read from INFO inside the Lua script, so a persisted old OPEN does NOT authorize a new server process after restart.
- Same intent PREPARE is idempotent; 60 concurrent calls yield exactly one new prepare and 59 idempotent prior results. Client IDs are unique; conflicting owner, Binance order ID or duplicate fill rejected. Partial and overfill validated. Blind resubmission after SUBMITTING or UNKNOWN is always blocked.
- research/tests/test_order_intent_ledger_persistent_redis_v1.py PASS.
- orchestration/risk/unknown_order_recovery_evidence_v1.py and research/tests/test_unknown_order_recovery_evidence_v1.py: 12 checks PASS. Exact client order ID, side/symbol, original quantity, exchange order ID and distinct fill IDs must match; failure, missing order or incomplete history remains UNKNOWN and never permits resubmission.
- None of these components calls Binance or changes production Redis.

## IMPORTANT unresolved integration
- The ledger is NOT atomic with Central Risk V2 RESERVE yet. Production needs one transactional reserve+intent operation with authenticated source owner and proper durable generation.
- Test-only proof strings are NOT signed Binance evidence. An untrusted caller can fabricate them; research module returns no actual broker permission. Isolated Redis ACL is not production ownership/security.
- The Redis restart run_id fence prevents stale OPEN from AOF restart; it does not provide resilience to malicious writes, shared Redis, changed ACL, full process/fault injection, replica failover, disk-full or partial AOF corruption.
- ClientOrderId must be generated and persisted BEFORE actual Binance submission, then exact REST order lookup performed after ambiguous response. No actual order submissions or POSTs in this test.
- Actual COMMIT must verify true exchange fill, partial quantities and position state; RELEASE must independently verify flat, normal open orders, Algo orders and in-flight orders. UNKNOWN => freeze.
- Missing whale/liquidity canonical source, V7 promotion and SHORT native stop production authorization still prevent SHORT1 live.
- Current central risk dry service and LONG1 live engine are unchanged. SHORT1 DRY_RUN=1 ARMED=0.

## Next
Build atomic reserve+intent transaction and authenticated read-only broker recovery proof. Then run additional faults in isolated Redis before any dry-run integration.
