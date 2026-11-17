# LONG1 + SHORT1 — Central Account Snapshot and Partial Fill/Stop E2E
14 November 2026 UTC. Read-only Binance/Redis and isolated Redis with simulated broker events. No production changes or actual orders.

## Actual verified source observations
- Binance USD-M account: wallet 62.66814695 USDT; unrealized PnL 0; margin 62.66814695; available 62.66814695; initial margin 0; account positions 0.
- Central Redis snapshot mtm equity 62.66814695 and available same; age about 115s; zero reported positions; reservation hash empty.
- Risk Master portfolio state REDUCED, equity peak 68.08915253 USDT, drawdown about -7.96%. Normal notional cap 46% must NOT be applied directly in REDUCED mode.
- Source cross-check result CONSISTENT_READONLY, but authorization flags are ALL false: cannot reserve, commit or release based on observation alone.
- Snapshot read is not atomic across Binance REST and Redis. Real broker fills and all normal/Algo orders have to be checked for any actual release.

## Implemented research checks
- orchestration/risk/authoritative_snapshot_evidence_v1.py: finite wallet/PnL/margin, available, central snapshot equity, timing/freshness, DD mode and peak, position count consistency, account/Redis discrepancies => UNKNOWN.
- orchestration/risk/position_stop_lifecycle_evidence_v1.py: immutable reservation and intent binding; verified partial qty against broker position, matching side, native MARK_PRICE STOP_MARKET with clientAlgoId and closePosition true. If stop missing => FREEZE_UNPROTECTED; if flat but any order/Algo remains => BLOCK_RELEASE; if exit identity/qty match => FLAT_HISTORY_CONSISTENT, yet still no authorization to RELEASE.
- research/short/audit_long1_short1_authoritative_snapshot_readonly_v1.py: signed Binance GET and Redis read-only; result in research/results/LONG1_SHORT1_CENTRAL_RISK_ACCOUNT_SNAPSHOT_CROSSCHECK_20261114.json.
- research/tests/test_authoritative_snapshot_evidence_v1.py: PASS 12.
- research/tests/test_position_stop_lifecycle_evidence_v1.py: PASS 13.
- research/tests/test_atomic_reserve_to_stop_lifecycle_isolated_v1.py: PASS integrated atomic research RESERVE+PREPARE on isolated Redis, simulated partial fill/no stop, valid stop, flat with open Algo, matched flat history, reservation retained.
- Prior research/tests/test_atomic_reserve_prepare_aof_isolated_v1.py: PASS AOF and 60 concurrent requests.

## Production limitations
- In integrated test all fills/stops are MOCK evidence, not real Binance exchange events. No broker COMMIT, verified stop placement or live RELEASE was executed.
- Real exchange signed source evidence, transaction history, client ID and service authentication must be bound to a generation-fenced state machine before production.
- Current central dry service is not V2; LONG1 live and native stop code unchanged; SHORT1 DRY_RUN=1, ARMED=0, model unpromoted and whale/liquidity provenance incomplete.
- Caller-supplied proof strings in earlier isolated prototypes are not real signed exchange evidence and must never authorize live actions.
- VM disk ~91%; no permanent extra collector or service was started.

## Next
Build a trusted read-only reconciler binding immutable intent/reservation/client order ID to signed Binance order/fill/native Algo data, then test UNKNOWN, partial fill, stop attachment error and recovery with frozen risk; only after this design actual authenticated COMMIT and RELEASE in isolated Redis.
