# LONG1 + SHORT1 — Binance/Redis read-only reconciliation
14 Nov 2026 UTC. No Binance/Redis write, no risk release, no service restart or live configuration change.

## Read-only observations
- Signed Binance USD-M GET for NEARUSDT + INJUSDT: /fapi/v3/positionRisk, /fapi/v1/openOrders, /fapi/v1/openAlgoOrders.
- New isolated pure classifier orchestration/risk/exchange_flat_readonly_reconcile_v1.py.
- Any nonzero position, ordinary open order, or Algo open order => BLOCK_RELEASE.
- Unavailable, malformed, wrong symbol or ambiguous response => UNKNOWN.
- All three complete and no position/orders => FLAT_CANDIDATE only; never automatic RELEASE.
- Two independent repeated six-endpoint read-only sequences returned FLAT_CANDIDATE on both symbols; no ordinary orders or Algo, and no open positions at observation time.
- Redis read-only: neither coin has a local bot:positions state record, central:risk:dry:reservations:v1 empty, risk snapshot health OK, zero positions, snapshot age about 111 seconds.
- all_readonly_sources_agree_flat_candidate=true, safe_to_auto_release=false, risk_master_ready=false, authorization=NONE.
- Actual structured report: research/results/LONG1_SHORT1_BINANCE_REDIS_READONLY_RECONCILE_20261114.json.

## Tests and limitations
- research/tests/test_exchange_flat_readonly_reconcile_v1.py PASS 15 cases including absent/NaN data, ordinary-order/Algo-order blocking, nonzero positions and simulated API failure.
- Broker endpoints are not atomic across calls; two empty readings DO NOT prove an unknown in-flight order was rejected or filled. Cannot release reservation solely on empty open order lists.
- Production must reconcile specific order IDs/client order IDs, ambiguous POST/DELETE outcomes, actual fills/positions, reserved generation, matching service ownership and durable event journal.
- The current V2 research Redis Lua accepts synthetic reconciliation text values; these ARE NOT actual signed exchange proof.
- LIVE LONG1 is not centrally risk-authorized by V2. SHORT1 DRY_RUN=1 ARMED=0 and model promotion OFF. Full pipeline production-authorized status remains false.
- No new collector or paid API subscription created. Existing services preserved.

## New source paths
- orchestration/risk/exchange_flat_readonly_reconcile_v1.py
- research/tests/test_exchange_flat_readonly_reconcile_v1.py
- research/short/audit_long1_short1_exchange_flat_readonly_v1.py
