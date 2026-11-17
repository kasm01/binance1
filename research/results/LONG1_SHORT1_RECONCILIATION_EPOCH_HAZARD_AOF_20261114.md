# LONG1 + SHORT1 — Reconciliation Epoch and Durable Risk Hazard Audit V2
14 Nov 2026 UTC. Research-only isolated Redis + MOCK Binance evidence, no production system or account changes.

## Motivation and implemented controls
Prior research audit V1 recorded one evidence fingerprint per intent and rejected later changes. It preserved a CLOSED gate but did not append newly observed hazards in an independently reviewable way. This V2 fixes the research audit history, not production authorization.
- New orchestration/risk/isolated_reconcile_epoch_v2.py.
- Every broker observation is issued an increasing Redis epoch BEFORE broker I/O. Redis Lua appends each distinct result to a durable event stream, guarded by immutable intent, reservation generation and client order ID.
- Results are ranked by severity: PROTECTED/FLAT evidence 1, unresolved fill/hold 3, UNKNOWN_FREEZE 4, FREEZE_UNPROTECTED 5.
- The newest observation by issued epoch is the latest pointer. A delayed older positive does NOT overwrite a newer negative. Late negative results are separately archived, including to hazard outbox.
- Highest observed hazard severity is sticky even when a newer observation appears positive; no automated reset or reauthorization.
- Every distinct severe observation (>=3) gets a separate durable hazard outbox stream record. Exact same event ID/payload replay is idempotent; event ID reuse with different evidence is rejected.
- The risk gate is set CLOSED during accepted audit, regardless of label. RESERVE+PREPARE study uses Redis run_id fencing, so persisted OPEN state cannot revive after Redis restart.
- Lua prechecks Redis key types to mitigate nonrollback partial writes on wrong-type errors.
- No production Redis, Binance POST, real stop placement, COMMIT or RELEASE; all authorization flags remain false.

## Tests
- research/tests/test_reconcile_epoch_aof_isolated_v2.py: PASS.
  - start epoch 1 positive then epoch 2 negative, complete negative first; late positive archived, latest still negative, severity 5.
  - New positive epoch 3 cannot clear severity 5.
  - Duplicate event returns idempotent replay without second audit/hazard.
  - Reused conflicting event ID and forged generation rejected.
  - Later out-of-order UNKNOWN archival retains separate hazard.
  - AOF restart preserves five audit entries, two hazard entries and held risk reservation; gate CLOSED.
- research/tests/test_reconcile_epoch_worker_integration_v2.py: PASS.
  - Isolated Redis RESERVE+PREPARE with staged mock SHORT1 fill and native stop.
  - Correct stop -> PROTECTED_EVIDENCE_ONLY; stop loss -> FREEZE_UNPROTECTED; simulated broker timeout -> UNKNOWN_FREEZE; later correct stop -> PROTECTED_EVIDENCE_ONLY.
  - Four events, two hazard entries, highest severity 5 persisted after restart. Reservation still held; COMMIT/RELEASE/order authority false.
- Previous tests for bound-exchange reconciliation (23 cases) and exact signed GET adapter (12 cases) remain independent and unchanged.

## Important production caveats and cost
1. Broker evidence in worker integration was SYNTHETIC; a positive label is not a cryptographic proof. Event hashes only detect changed input under the model, not source forgery; no service ACL/identity or independent trusted broker reconciler exists.
2. Epoch issuance order is a local Redis atomic counter. Across deleted/rebuilt Redis, multisite failover or multiple risk masters a durable fencing authority is still needed.
3. Sticky severity requires an independently authorized, auditable recover/unfreeze policy. Not implemented, so no automatic reopening after any incident.
4. Hazard outbox is stored durably on the isolated AOF, but delivery to a supervisor/operator and risk-master policy propagation are NOT implemented. Stream append alone does not constitute notification.
5. Redis scripts are atomic to concurrent clients but not automatic rollback transactions after runtime errors. Crash/disk-full, tampered journal, backpressure, Redis multi-node split-brain still need fault injection.
6. LONG1 live supervisor and native stop still independent; running central dry-risk service is old protocol. SHORT1 DRY_RUN=1 ARMED=0; model approval, independent whale and canonical liquidity provenance, exchange-authenticated COMMIT/RELEASE are missing.
7. No new continuous collector, paid feed or permanent Redis service; disk remains about 91% full.

## Next
- Create read-only hazard outbox consumer for isolated test, with monotonic cursor and idempotent acknowledgement but no production actions.
- Add fail-closed disk-full and Redis unavailable tests, and a trusted authenticated reconciliation envelope with strict source timestamps and per-client ID freshness. Never accept user-controlled proof strings for live release.
- Only then prototype dry-run Risk Master shared controller; no live migration and no scaling to LONG2-4/SHORT2-4 yet.
