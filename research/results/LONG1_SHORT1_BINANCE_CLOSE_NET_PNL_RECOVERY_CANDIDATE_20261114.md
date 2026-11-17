# LONG1 + SHORT1 — Close/Fill Net PnL Evidence + RECOVERY Win Candidate
2026-11-14 UTC. Research-only; no production Redis writes, exchange orders, configuration or service changes.

## New evidence and review queue
- orchestration/risk/close_fill_net_pnl_evidence_v1.py:
  - Validates immutable intent and entry client order ID, distinct entry/exit Binance order IDs, BUY/SELL direction, FILLED order and matching full entry/exit quantities, unique trade IDs, all entry+exit fills, zero-position read, and no remaining normal/Algo open orders.
  - Aggregates Binance USD-M per-fill realizedPnl, USDT commissions on both entry and exit, and externally supplied funding_usdt. This is MATH evidence only: funding source and ledger provenance still need authenticated capture.
  - When a trade commissionAsset is BNB (as previously seen on real NEARUSDT historical trade fills), refuses to count it as USDT: UNPRICED_FEES_HOLD. No point-in-time conversion has been implemented or verified. Partial fills, missing exit, open positions or open native Algo produce UNKNOWN_FREEZE.
  - If fee/funding math checks pass, result is NET_PNL_CANDIDATE_ONLY, eligible_recovery_win=false, can_update_risk_mode=false and can_release=false. A candidate never increments RECOVERY success counter, regardless of positive net number.
- research/tests/test_close_fill_net_pnl_evidence_v1.py PASS 13 checks.
- orchestration/risk/isolated_close_candidate_queue_v1.py:
  - Dedicated isolated Redis Lua review queue; requires current mode state epoch and source generation; stores a content digest + one pending review item + append-only queue event, CLOSED gate, no risk-mode changes.
  - Duplicate same event ID+payload is idempotent, inconsistent replay rejected. Mode generation and epoch mismatch rejected.
- research/tests/test_close_candidate_mode_journal_isolated_aof_v1.py PASS. Mock profit + missing BNB conversion candidates persisted, no RECOVERY win counted; AOF restart retained pending candidates, historical mode and CLOSED gate.

## Actual Binance history context
- Previous exact signed Binance historical read confirmed NEARUSDT FILLED BUY entry order and 8 NEAR filled quantity; observed Binance commissionAsset BNB.
- This phase did NOT fetch a new complete broker position lifecycle or funding events. Consequently NO real historical close has been certified net profitable and NO live risk-mode CLOSE_EVENT has been created.
- A positive mocked USDT-only profit test illustrates numeric accounting, not a genuine verified win. Funding and BNB commission conversion are deliberately unresolved.

## Production blockers
1. The CLOSE_EVENT journal currently accepts caller-supplied pnl/opened_mode; a real writer MUST require trusted exact broker evidence, funding, fee conversion, complete entry+exit intent chain and signed/bound generation. Do not promote positive mock evidence or set NORMAL.
2. Real lifecycle can have multiple partial entry orders, staggered exits, funding in several intervals/assets and varying fees; this V1 only supports a single full entry/single full exit. In such cases it fails closed.
3. Whole-account zero open normal/Algo list and current flat position are point-in-time observations, not atomic proof of no in-flight orders. Real release requires epoch-bound durable order journal and independent broker reconciliation.
4. Source arrays and funding_usdt in this prototype are caller-injected; this is not authentic Binance authorization or external audit/notification.
5. Existing production central risk remains dry-run V1. LONG1 active and native stop active; SHORT1 DRY_RUN=1 ARMED=0. Do not enable live SHORT1 or modify LONG1.
6. The isolated candidate queue and mode journal are research-only; no production key/ACL/server endpoint integration.
7. VM disk about 91% used: temporary Redis children stopped and removed; no new always-on collectors or paid data feeds.

## Next
- Build a bounded, read-only exact Binance source for entry and exit order IDs, complete trade fills and time-bounded funding/commission conversion history, with original risk mode anchored to durable pre-submit intent.
- Add position generation/order sequence reconciliation across partial exits, stop child orders and external manual closes.
- Isolated worker must show authenticated broker source provenance and fail-closed on missing fee conversion/funding before authorized risk journal can ever count a Recovery win.
