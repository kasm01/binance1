# LONG1 + SHORT1 — Persistent DD Mode Journal / Generation Fenced Atomic Reserve V3
14 November 2026 UTC. Isolated research only; NO changes to production Redis, order execution, .env or running services.

## Implemented
- orchestration/risk/isolated_risk_mode_journal_v1.py:
  - AOF durable Redis Lua state journal with explicit source generation, increasing epoch and strictly increasing event source timestamp, immutable historical equity peak and drawdown, deduplicated event IDs.
  - Actual installed central dry risk policy is preserved: 12% drawdown enters REDUCED, 6% recovery goes to RECOVERY, two profitable closes OPENED IN RECOVERY are needed to reach NORMAL; risk stop in RECOVERY returns REDUCED.
  - Synthetic INIT_TEST starts at REDUCED with explicitly provided historical peak; it does NOT infer prior risk state from one current snapshot. Historical data provenance is test-only; no production replay authorization.
  - No action may automatically open the risk gate; each accepted update forces gate CLOSED.
- orchestration/risk/atomic_reserve_mode_journal_v3_research.py:
  - Extends previous isolated V2 Lua to atomically compare risk snapshot mode/epoch/generation/last source time/equity peak with durable mode journal before RESERVE+PREPARE.
  - Rejects forged NORMAL mode, wrong source generation, future epoch and stale snapshot; computes REDUCED sizing 23% of equity (0.5 base request x 0.92 headroom x 0.5 REDUCED multiplier).
  - All V3 reply order_submission_authorized=false; central live Risk Master remains unmodified.
- research/tests/test_risk_mode_journal_aof_v1.py: PASS 14 events/epochs, stale timestamps, source generation conflict, duplicate event, RECOVERY wins, risk stop relapse, AOF restart and preserved state.
- research/tests/test_atomic_mode_journal_reserve_v3_aof.py: PASS integrated mode generation gate + atomic reserved/intent, 23% reduced cap, AOF restart fail-closed.
- research/tests/test_mode_journal_against_central_profile_v1.py: PASS 8 sequential mode transitions matching installed /home/kasm920/binance1-pro/orchestration/risk/central_risk_master.py and central_risk_enforce_dry_profile.py. No profile edit.

## Risk & deployment blockers
1. This is a RESEARCH PROTOTYPE with test-injected state generation and synthetic equity snapshot. A Redis writer with broad access could forge BOTH snapshot and journal; Redis ACL, trusted identity, real Binance evidence, service attestation and source-chain validation are absent.
2. Epoch and source event timestamp checked inside Lua on isolated Redis, not yet tied to actual broker stream sequence; rate limit, delayed signed REST response and race between exchange snapshots still require trusted causal versioning.
3. Redis Lua is atomic to concurrent clients, not an automatic rollback transaction after a runtime or disk failure. Incomplete AOF, crash, disk-full, replica split-brain and unexpected Lua errors must be fault-injected separately.
4. Persistent NORMAL escalation must be driven by verified Binance close fills, net fees and opened-risk-mode identity, rather than caller-supplied pnl/opened-mode in this research code.
5. V3 does not integrate exposure margin, leverage, market-context multiplier, all side/gross/symbol caps of the actual full CentralRiskMaster, signed origin of wallet/equity, native protective Algo lifecycle, nor authenticated COMMIT and RELEASE.
6. Test-mode gate INIT_TEST is synthetic and unsafe for production. Never point research Lua at Redis port 6379. Live LONG1 and native stop are independent; SHORT1 DRY_RUN=1 ARMED=0; central dry-risk service remains the V1 profile.

## Cost and status
- Temporary Redis AOF instances cleaned after tests, no new always-on collectors, no new fee-based data source.
- VM disk around 91% used; preserve limited headroom.
- No live Binance buy/sell, no risk authority migration, no service restart.
