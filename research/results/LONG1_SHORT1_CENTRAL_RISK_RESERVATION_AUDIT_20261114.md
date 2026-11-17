# LONG1 + SHORT1 — Central Risk reservation protocol audit
2026-11-14 UTC. No changes to running central risk service or trading execution; isolated research only.

## Verified system
- Running central gate comes from /home/kasm920/binance1-pro/scripts/central_risk_enforce_dry_service.py, not the short workspace.
- Current profile ENFORCE_DRY_RUN and production_authorized=False, validated LONG1 and SHORT1. Per-order baseline equity x 50% x 92% headroom = 46% equity notional subject to caps.
- Two-bot example equity=100, reserve LONG1 46 and SHORT1 46, combined 92 USDT notional. Central caps: gross 1x equity, long .5x, short .5x, absolute net .5x, symbol .5x, max two open.
- SHORT client sends only DRY_RUN EVALUATE_OPEN and RELEASE. LONG1 live executor remains independent and central risk is NOT the final live authority for both.
- Central dry hash keys reservations by strategy. Snapshot and live LONG1 Redis overlay are used for exposures. SHORT1 live overlay is not implemented.

## Critical isolated reproduction (NO live Redis write)
- RELEASE is processed before request_mode, validated strategy, or owner checks. A forged LIVE mode dry RELEASE with strategy LONG1 removes LONG1 dry reservation.
- Delayed RELEASE of older reservation A can erase newer reservation B for same strategy. Ownership fencing is absent.
- Replaying identical request_id after first reservation changes response READY to REJECT due strategy slot already taken, so idempotency is absent.
- No COMMIT handshake after exchange fill. Reservation may expire or be removed when position appears in snapshot without exchange-order reconciliation.
- hgetall/read/calculate/hset is not atomic across processes. One running worker serializes its own loop, but multiple worker concurrency is not protected.
- Client requests have strategy IDs rather than authenticated caller identity; cannot be used as source authentication.
- Reproduction script (in memory only): /home/kasm920/binance1-pro/research/tests/audit_central_risk_dry_protocol_isolated_v1.py

## Offline reference V2 (NOT DEPLOYED)
- /home/kasm920/binance1/orchestration/risk/central_reservation_protocol_v2_dry_reference.py
- Implements request_id payload binding, idempotent repeated reserve, single strategy/symbol slot, shared portfolio cap, owner-specific RELEASE, stale token rejection, COMMIT with exact position ID, flat-verified release, and unresolved expired reservation freeze.
- /home/kasm920/binance1/research/tests/test_long1_short1_central_reservation_protocol_v2.py
- PASS 17 isolation tests including same-source replay, stale release and 50 simultaneous contenders resulting in exactly one winner.
- Thread lock works only within one Python instance; production requires Redis Lua/DB atomic transaction, caller authentication, exchange/Redis reconciliation and broker-order fencing.
- This is a safety reference only, not production live risk middleware and not complete exchange PnL model.

## Next gates
1. Leave running LONG1 and central dry service unchanged while designing atomic Redis reserve/commit/release.
2. Add reservation generation and owner token plus strict compare-and-delete, durable response idempotency journal.
3. Pending or committed positions cannot expire without authoritative exchange flat and no in-flight orders.
4. Startup/restart reconcile both LONG1 and SHORT1 exchange positions, native Algo stop and open orders; uncertainty freezes all risk-increasing entries.
5. Test isolated Redis concurrency and outage recovery before deploying any new central service. No LONG2-4/SHORT2-4 scaling yet.
6. SHORT1 model Master provenance, whale/liquidity evidence and native stop remain incomplete; never bypass their gates to enable orders.

## Final system state
- LONG1 supervisor and protective stop active; SHORT1 DRY_RUN=1 ARMED=0; central risk dry gate active.
- No live order or bot config/service modifications in this session.
