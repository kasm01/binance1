# LONG1 + SHORT1 — Isolated Dry-Run Reconcile Worker / Durable Audit
14 Nov 2026 UTC. Research-only; NO live executor, systemd service, production Redis, Binance order or model/policy change.

## Delivered
- orchestration/risk/isolated_reconcile_worker_v1.py
  - Reads SAME durable research reservation and intent from isolated Redis.
  - Uses injected read-only broker adapter for exact order/client ID, order fills, positionRisk, open normal orders, open Algo.
  - Classifies results with bound_exchange_reconcile_v1.py: UNKNOWN_FREEZE, RECOVERED_FILL_PENDING_JOURNAL, PROTECTED_EVIDENCE_ONLY, FREEZE_UNPROTECTED, HOLD_RECONCILIATION, FLAT_EVIDENCE_ONLY.
  - Atomic Lua audit CAS checks immutable request_id/reservation generation/client_order_id; writes one evidence digest + audit stream event; sets isolated gate CLOSED in the same script.
  - Duplicate exact audit is idempotent. Different evidence under the same reconciliation epoch is rejected; it MUST create a new authenticated reconciliation epoch rather than overwrite the previous record.
  - All outputs force can_submit=false, can_commit=false, can_release=false.
  - No actual Binance execution, no real COMMIT/RELEASE, no live Redis endpoint configured.

## Tested
- research/tests/test_isolated_reconcile_worker_aof_v1.py:
  - Isolated Redis (Unix socket, TCP port 0), appendonly yes / appendfsync always, cleaned afterward.
  - Real Lua RESERVE+PREPARE creates one SHORT1 reservation and PREPARED intent; synthetic filled state for test mock and correct native stop produces PROTECTED_EVIDENCE_ONLY.
  - Atomic audit persisted; gate immediately CLOSED; original pending reserve still exists.
  - Same reconciliation replay and 30 concurrent identical workers produce no duplicate events.
  - Recheck with missing stop => FREEZE_UNPROTECTED, conflicts with existing positive result in the same epoch and is rejected from overwriting the existing audit. Gate remains CLOSED; production needs explicit urgent severity telemetry and new epoch.
  - Simulated broker outage => UNKNOWN_FREEZE, no authorization.
  - Separate fresh isolated Redis scenario: FIRST broker order lookup fails, UNKNOWN_FREEZE is durably audited and gate CLOSED; restart keeps both audit and pending reservation.
  - AOF restart preserves reservations/intent and audit, never reopens the gate automatically.
- Test PASS. Prior tests for exact Binance read-only adapter 12/12 and binder 23/23 remain separate components.

## IMPORTANT integrity and provenance caveats
1. Mock broker responses are NOT signed cross-endpoint Binance evidence. The worker has adapter injection but is NOT a production authorized signing/reconciliation service. Cryptographic service identity, source ACL, broker freshness and independent policy verification remain missing.
2. Test artificially updates the isolated intent FILLED via HSET (a simulated trusted journal writer). Production must never accept arbitrary HSET fills; trusted generation- and tradeId-bound writer is necessary.
3. Audit fingerprint alone is a hash of a classifier result; it does not prove the origin, freshness or validity of Binance data. The single-epoch conflict path rejects changes and leaves the first audit as-is, though gate CLOSED; an urgent risk/hazard stream and monotonic trusted rechecks are needed.
4. Redis Lua is atomic for scripts but not a rollback transaction on script errors. The audit uses key type validation after certain source reads but not every unexpected Redis failure. No full disk-full/power-loss/replica split-brain simulation yet.
5. The current module does NOT submit an order, commit positions, reserve live money or release actual exposure. Running LONG1 remains independent; SHORT1 DRY_RUN=1 ARMED=0.
6. This VM has limited free disk (~3.5GB), so no permanent collector, always-on test Redis, or duplicate historical market feed was created.
7. Existing central dry service in /home/kasm920/binance1-pro has older reservation protocol; V2 study has NOT been deployed to it.

## Next gates
- Authenticated service-to-service read-only broker evidence authority, strict account/equity freshness, same generation and source event signature or ACL-bound event provenance, reduced DD governor.
- Durable reconciliation epoch and severity append channel for changing broker evidence, retaining prior event audit, avoiding silent stale positivity.
- Full trusted order/fill COMMIT and native stop attach/verify, partial exits/Algo child order and confirmed zero position/no pending order RELEASE with multi-read race/recovery.
- Redis restart/disconnect, key corruption, disk full and writer failure injection before even isolated deployment as a production-shaped dry-run service.
- Keep LONG1 live unchanged; SHORT1 closed; do not scale LONG2-4 or SHORT2-4.
