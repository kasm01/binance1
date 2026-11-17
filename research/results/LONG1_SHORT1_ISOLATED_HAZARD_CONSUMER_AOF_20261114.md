# LONG1 + SHORT1 — Isolated hazard outbox consumer, recovery and local receipts
2026-11-14 UTC. Research only; no live bot, Binance order, production Redis, service unit or config modified.

## New component
- orchestration/risk/isolated_hazard_outbox_consumer_v1.py
- Redis Stream consumer group receives durable risk alerts (severity >=3) from the existing isolated reconciliation epoch outbox.
- XREADGROUP reads new stream messages. XAUTOCLAIM reclaims unacknowledged pending entries after worker crash or Redis restart.
- Single Lua script validates source stream message ID, intent/event/epoch/severity identity, validates destination types, checks group pending membership, writes exactly one LOCAL receipt and receipt event and XACKs.
- Receipt event ID is immutable/idempotent. Altered event fields are rejected. No order, commit, release, risk gate reopening or external notification.
- Redis group delivery is at-least-once. Local receipt + XACK is atomically tested. Real external exactly-once alert delivery cannot be guaranteed without consumer-specific delivery protocol and acknowledgment; this component does not call external services.
- Caveat: Redis Lua atomicity serializes client access but not automatic rollback of partial writes on arbitrary runtime error; hardened preflight type/pending checks are included. Full OOM/power-loss is not proven.

## Isolated test
- research/tests/test_isolated_hazard_outbox_delivery_aof_v1.py
- Temporary Redis 7 UNIX socket with port=0, private directory, AOF enabled and appendfsync always, no permanent service created.
- Two alerts produced from bound SHORT1 reservation and owned intent in test Redis.
- Simulated worker crash BEFORE ACK, then Redis AOF process restart: both stream events recovered from pending by XAUTOCLAIM, each locally received and ACKed once.
- Replaying already handled alert does not duplicate local receipt or event.
- Tampered event ID is refused; no receipt/ACK.
- A new third alert after restart is consumed and ACKed once.
- Fourth alert with intentionally wrong-type local receipt stream: rejected BEFORE ACK; pending count remains 1 and no receipt inserted. Restoring correct type permits a single ACK and receipt.
- PASS: four unique durable local receipt records, zero pending after recovery; Risk gate CLOSED and reservation intact.
- NO notifications, no e-mails, no Telegram, no live auto risk enforcement.

## Production blockers
1. Hard requirement: authenticated broker risk event source and Redis ACL/service identity. Mock-produced hazard streams cannot be treated as real Binance proof.
2. Deliver alerts to a trusted Central Control supervisor with at-least-once semantics, monotonic epoch and severity 5 sticky. Supervisor should read pending and source ledger before action.
3. External alert delivery cannot be exactly once with simple ACK: must decide and implement explicit receiver idempotency, retries, and acknowledgment semantics.
4. Test disk-full, AOF recovery after unclean crash, Redis memory exhaustion, replica split-brain and long-running backpressure in isolated environment before production dry-run rollout.
5. Existing central dry risk service is old protocol. LONG1 live independent, SHORT1 DRY_RUN=1 ARMED=0. No production authorization, COMMIT/RELEASE, model promotion or live order change.
6. VM disk free about 3.5 GB; do not start continuous worker or duplicate collectors without cleaning historical test data.

## Files
- orchestration/risk/isolated_hazard_outbox_consumer_v1.py
- research/tests/test_isolated_hazard_outbox_delivery_aof_v1.py
- research/results/LONG1_SHORT1_ISOLATED_HAZARD_CONSUMER_AOF_20261114.md
