# LONG1 + SHORT1 — Atomic RESERVE+PREPARE / durable Redis AOF experiment
2026-11-14 UTC. ISOLATED RESEARCH; no production Redis, Binance order, live service or .env modification.

## Implemented
- orchestration/risk/atomic_reserve_prepare_v1_research.py
- A single Redis EVAL transaction atomically checks LONG1/SHORT1 strategy+symbol+side, ownership, duplicate request/client ID, gross/side/notional caps, and then creates BOTH pending central reservation and immutable PREPARED intent, ID index, idempotency journal and audit stream event.
- Safety constraints in research:
  - scoped LONG1 NEARUSDT LONG and SHORT1 INJUSDT SHORT only.
  - example equity 100 and central notional per bot 46 each, total combined 92 and max gross 100, max per side 50.
  - duplicate identical request returns prior response; changed source-generation, size or client ID under same request rejected.
  - client order ID unique across intents.
  - error or insufficient equity/headroom cannot leave a dangling PREPARED intent.
  - Redis process run_id included in persisted gate, so AOF OPEN state never grants new Redis server risk authority after restart.
  - unresolved reservations/intent on restart block test reopening; new risk remains CLOSED.
  - Redis Lua type prevalidation before writes rejects malformed hashes or audit-stream key to mitigate nonrollback errors.
- The Lua reply explicitly includes order_submission_authorized=false. This is not a production broker admission system.

## Test execution
- research/tests/test_atomic_reserve_prepare_aof_isolated_v1.py
- Temporary isolated Redis 7 over owner-private Unix socket, --port 0, appendonly yes, appendfsync always, no RDB, 32 MB cap, noeviction; child processes cleaned after tests.
- PASS 60 concurrent competing SHORT1 reservation/intent requests => 1 accepted, no mismatched pending/intent/journal/client ID records.
- PASS two-bot shared 100 USDT example => 46+46=92 USDT *notional*, not wallet margin; no overlapping strategy slots.
- PASS idempotent duplicate and immutable payload conflict; no duplicate client ID.
- PASS persisted OPEN-gate Redis restart => CLOSED; AOF pending reservation and intent still exist; no reopening with unresolved state.
- PASS invalid strategy/symbol, zero equity and corrupt audit stream; corrupt key rejection leaves zero partial reservation/intent/journal.
- PASS simulated Redis outage returns transport error, never executable order permission.

## Unresolved, hard safety blockers
- This new combined Lua module and existing V2 reservation and intent ledger are SEPARATE research prototypes; full live combined state machine is not yet integrated with real exchange fills, native algo stop and centralized Risk Master service.
- Requested equity/notional and test OPEN_GATE proof are supplied by test caller: NOT authenticated Binance wallet/position, and not a production authority. Real shared portfolio MTM, existing LONG1 exposure and equity/margin must be injected by independently authenticated central snapshot.
- Lua operations are serialized at Redis, but Redis Lua is not a rollback transaction on runtime error and memory/disk-full faults remain untested; add Redis transactional durability/recovery and fail-closed reconciliation before deployment.
- Test did not verify all real stop/order classes, partial fills, stop failures, funding, cross-margin or exchange commission. Production RELEASE still requires signed/verifiable broker-flat and exact order ID + position reconciliation.
- Real service identity, per-app Redis ACL, generation fencing and reconciliation epochs must prevent forged client inputs; source provenance not authenticated.
- SHORT1 model V7 remains unpromoted; canonical independent whale/liquidity and native stop production reconciliation incomplete.
- IMPORTANT: do NOT point this module at Redis production port 6379, do NOT enable SHORT1 LIVE and do NOT transfer LONG1 live risk authority without full staged tests.

## Verification
- Existing binance1-long-supervisor.service, binance1-long1-native-stop.service, binance1-short1-inj-shadow-signal-bus.service and binance1-central-risk-enforce-dry.service remain ACTIVE.
- SHORT1 DRY_RUN=1 ARMED=0 SYMBOL=INJUSDT.
- No actual Binance buy/sell or order cancellation performed.
