# LONG1+SHORT1 — Atomic Risk-Mode RESERVE+PREPARE V2 AOF
2026-11-14 UTC. Isolated research only. No production Redis/long/short service changes; no orders.

## Scope
- New file orchestration/risk/atomic_reserve_prepare_mode_v2_research.py
  derives a new isolated Lua contract from the V1 RESERVE+PREPARE module. V1 and all production modules remain unchanged.
- Uses a local Redis risk snapshot under a dedicated key and checks snapshot status, allowed mode, finite equity, available and peak, snapshot freshness (<=120 sec), drawdown consistency, unknown-inflight flag, and caller equity equal to snapshot.
- Mode multipliers match actual current central dry profile: NORMAL 1.00, RECOVERY .70, REDUCED .50; operational multiplier .92 and base per-bot request .50 equity. Illustrative entry budgets: NORMAL .46, RECOVERY .322, REDUCED .23 of MTM equity.
- Preserves atomic RESERVE+PREPARE, unique strategy/symbol/client ID, common gross/side caps, same-request idempotency, Redis server-run-id gate fence and AOF persistence. No broker order or commit/release authority in returned result.
- Test risk snapshot uses deliberately explicit ISOLATED_TEST_EQUITY_SNAPSHOT tag. Any caller with Redis write permission can forge it; thus NOT a production security boundary.

## Test results
- research/tests/test_atomic_reserve_mode_aof_v2.py PASS, isolated Redis Unix socket, TCP port 0, AOF enabled, temporary directory cleaned after test.
- Equity 100 hypothetical: requested 50 -> NORMAL 46, RECOVERY 32.2, REDUCED 23 USDT notional.
- Equity 62.66814695 USDT (earlier signed Binance account observation), hypothetical REDUCED: 14.4136737985 USDT notional; no actual new order.
- Two REDUCED strategies at equity 100: 23+23=46 USDT total notional.
- 60 concurrent SHORT1 requests: exactly one new reservation, no dangling intent journal records.
- Stale/missing/inconsistent/untrusted snapshot, unknown inflight state, caller equity mismatch, unreasonable mode and DD conflict all fail closed.
- Redis restart despite AOF retaining OPEN state => closed; pending reserve+intent retained; no reopening with unresolved state.
- Existing research tests for Redis alert outbox OOM, hazard consumer, central sticky severity remain separate.

## Critical limitations
1. This is NOT yet wired to real signed Binance+Redis authoritative snapshot in one trusted service; caller-controlled Redis snapshot has NO ACL and can be forged. Do not use for live.
2. Mode/peak/DD persistence cannot be proven from a single fresh broker snapshot; the trusted central profile must carry historical REDUCED state and recovery wins. A source suddenly reporting NORMAL while currently above -12% DD may still be an unsafe state regression if prior history was REDUCED. A durable mode transition journal is needed.
3. The V2 isolated combined prototype does not yet merge all live exchange positions, open normal/Algo orders, margin/leverage, fee and native stop ownership, or live Central Risk service's full Governor decisions. A caller-supplied equity is checked against an injected snapshot, not a real broker identity.
4. Test has not simulated actual disk-full/AOF fsync failure, partial file corruption or multi-node Redis split-brain. Redis Lua atomicity does not guarantee transactional rollback on any runtime error.
5. Real COMMIT and RELEASE require source-authenticated exact order/trade/stop reconciliation. The research mode module cannot issue Binance orders and never returns an execution authorization.
6. No actual deploy of V2. LONG1 active native stop, SHORT1 DRY_RUN=1 ARMED=0; running Central Risk dry service unchanged.

## Next gate
- Persist verified CENTRAL MODE (NORMAL/RECOVERY/REDUCED) as generation-bound state in isolated Redis and test that mode cannot be elevated by one stale or conflicting snapshot.
- Integrate trusted signed account+order+position broker evidence via a restricted worker and service identities in dry-run test only.
- Fault-inject AOF/disk write errors and crash after reservations without touching VM root disk, and verify corruption/UNKNOWN is always frozen.
