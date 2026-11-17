# LONG1 + SHORT1 — Isolated Redis OOM, Durable Alarm Gate, REDUCED Budget
14 Nov 2026 UTC. Read-only production policy inspection; isolated Redis only for all writes. No live Binance order, service, config or production Redis changes.

## Central Risk Master policy (verified from actual installed LONG repo)
- Source: /home/kasm920/binance1-pro/orchestration/risk/central_risk_enforce_dry_profile.py and central_risk_master.py.
- Profile central_risk_enforce_dry.v1_h046_dd12_r06:
  base request/equity .50, headroom .92, mode multipliers NORMAL 1.0, RECOVERY .70, REDUCED .50; min order notional 10.0; max gross ratio 1.0; long/short cap .5; risk mode DD enter 12% recover 6%, profile production_authorized=False.
- Actual independent policy function evaluate_entry run on 62.66814695 USDT sample equity (portfolio peak 68.08915253):
  NORMAL 28.827347597 USDT notional, RECOVERY 20.179143318, REDUCED 14.413673799.
- These are sample standalone strategy budget ceilings, not wallet margin, order permission, or actual position allocation. Shared portfolio positions and pending orders, trading fees, and leverage are not included in the simple isolated example.
- Existing production central snapshot previously reported REDUCED mode and DD -7.9616%; do not change to NORMAL based on one good tick.
- New orchestration/risk/reduced_mode_budget_evidence_v1.py strictly classifies read-only consistency snapshot and computes non-executable budget only. Fail CLOSED for stale/NaN/missing equity, invalid risk mode, unmatched positions, pending reservations or missing source authority.
- research/tests/test_reduced_mode_budget_evidence_v1.py PASS 13 cases. IMPORTANT this module is NOT installed in the running V2 Lua risk reservation; earlier isolated 46% reservation prototype does not apply REDUCED mode and must NOT be promoted.

## Isolated Redis memory pressure
- research/tests/test_isolated_outbox_oom_failclosed_v1.py PASS with a dedicated Redis 7 Unix-socket-only temporary AOF noeviction instance.
- Reduced Redis maxmemory BELOW current used_memory in TEST INSTANCE ONLY; attempting to ACK an alert failed before inserting local receipt and event.
- Hazard entry stayed pending; risk gate remained CLOSED; central reservation retained.
- Restored the isolated memory ceiling; single local receipt and XACK then succeeded with no missing event.
- Does not prove all Redis disk-full or AOF disk-error scenarios; VM main disk at ~91% utilization, so deliberately no real disk-fill experiment or production Redis CONFIG change attempted.

## Central Control read-only sticky hazard policy
- New orchestration/risk/isolated_central_control_hazard_gate_v1.py checks CLOSED gate, durable owner/reservation ID, latest epoch pointer, append-only audit events, critical hazard outbox and local receipt hash, without writing or authorizing any order.
- research/tests/test_central_control_hazard_gate_isolated_v1.py PASS. Missing local receipt => CRITICAL_FREEZE, completed local receipt => CRITICAL_FREEZE, later positive audit => CRITICAL_FREEZE, AOF Redis restart => CRITICAL_FREEZE.
- Local receipt != external notification; external delivery and supervisor escalation have not been built.
- Source credibility is STILL research-only synthetic evidence. No cryptographic source authorization, Redis ACL service identity, or real broker COMMIT/RELEASE proof. No live Central Control policy has been installed.

## Risk + cost and next steps
- Hard work remaining: authentic source identity and Redis ACL/service account separation, trusted broker snapshot timestamps/epochs, actual multi-order partial-fill and native Algo stop recovery; account margin + BNB conversion/fees and dynamic DD budget under shared slots.
- Add disk write failure by injected low-level error rather than exhausting live VM disk, with validated crash/partial-AOF recovery in isolated store.
- Model full LONG1/SHORT1 as a single Central Risk Master, but allow live migration only after risk/source gates, LONG1 native stop safety, and SHORT1 independent whale/liquidity evidence + net OOS validation.
- No additional always-on collector, external alerts, or paid feed used. No user notification sent.

## Verification
- Live LONG1 supervisor, native stop service, SHORT1 shadow and central dry-risk services remained active.
- SHORT1 DRY_RUN=1, ARMED=0; research test Redis instances were shut down and cleaned.
- Tests:
  research/tests/test_reduced_mode_budget_evidence_v1.py
  research/tests/test_isolated_outbox_oom_failclosed_v1.py
  research/tests/test_central_control_hazard_gate_isolated_v1.py
