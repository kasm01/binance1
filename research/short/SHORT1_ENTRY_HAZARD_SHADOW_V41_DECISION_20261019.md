# SHORT1 Entry Hazard Shadow V41 — 2026-10-19

## Scope
Research/shadow only. No production routing or order veto.

## Historical causal reconstruction
- V30 C600 entries: 1103/1103
- Entry days: 233/233
- Binance Vision aggTrades missing days: 0
- Persistent micro cache: ~2.2 MB
- Snapshots: 5s / 10s / 15s / 20s / 30s
- Labels: 1m / 3m / 5m MAE/MFE and 40bp first-touch

## Key finding
Raw reversal confirmation is not enough. Timing risk becomes meaningful in context.

Research hazard candidate:
- flow_20s <= -0.60
- flow_delta_5_20 <= -0.30

Raw historical group: 36 entries.
Live-contract version additionally requires SHORT reversal already confirmed by 20s:
- 34 PENDING candidates
- 17 READY_AFTER_REVALIDATION when price continues down 20->30s
- 17 REJECT_CANDIDATE when price rebounds 20->30s

Historical contract parity:
- READY_AFTER_REVALIDATION: win ~76.5%, diagnostic PF ~0.93
- REJECT_CANDIDATE: win ~47.1%, diagnostic PF ~0.23, ADVERSE_FIRST ~52.9%

Important: this is not a hard production veto. The pattern is regime-sensitive and sample size is small.

## Live shadow implementation
Module:
- orchestration/telemetry/short1_entry_hazard_shadow_v1.py

Service:
- binance1-short1-entry-hazard-shadow.service

Input:
- V5 model shadow Redis event stream
- Binance Futures INJUSDT aggTrade WebSocket

Output:
- telemetry:short1:inj:entry_hazard_policy_v1

States:
- NOT_APPLICABLE
- NORMAL
- PENDING
- READY_AFTER_REVALIDATION
- REJECT_CANDIDATE

Semantics:
- 20s: detect extreme sell + sell acceleration only after entry reversal is confirmed
- 30s: if price continues down -> READY_AFTER_REVALIDATION
- 30s: if price rebounds/non-continuation -> REJECT_CANDIDATE

All states are TELEMETRY_ONLY / routing OFF.

## Outcome feedback
Analyzer:
- research/short/analyze_short1_entry_hazard_policy_live_v41.py

Timer:
- binance1-short1-entry-hazard-policy-analyzer.timer

The analyzer joins policy events to existing 5m hazard outcomes by source_event_id. It does not call Binance.

## Cost/safety
- No extra REST request added by policy collector.
- One lightweight public aggTrade WebSocket connection.
- ~32 MB RSS observed.
- Production services remain inactive.
- ARMED=0
- LIVE_KILL_SWITCH=1

## Production gate before any routing change
Do not promote until live shadow has enough PENDING/READY/REJECT samples across regimes and confirms lower ADVERSE_FIRST / better execution quality without deleting profitable continuation trades.

## Live integration update
- Binance USDⓈ-M 2026 raw market stream path verified as:
  `wss://fstream.binance.com/market/ws/injusdt@aggTrade`
- Legacy `/ws/...` endpoint accepted the handshake but delivered no aggTrade messages in the VM test; do not use it for this collector.
- First live policy event pair observed at 20:36 UTC:
  - 20S_HAZARD -> NOT_APPLICABLE
  - 30S_REVALIDATION -> NOT_APPLICABLE
  - 5s flow +0.645, 20s flow -0.070, but no 20s downside reversal confirmation, so fail-closed correctly avoided a false hazard signal.
- Status command:
  `scripts/short1_entry_hazard_status.sh`

## V42 WebSocket-native outcome feedback

Added non-routing service:
- `binance1-short1-entry-hazard-outcome-ws.service`
- code: `orchestration/telemetry/short1_entry_hazard_outcome_ws_v1.py`
- output: `telemetry:short1:inj:entry_hazard_policy_v1_outcomes_ws`

Contract:
- consumes only 30S policy events created while the service is already connected,
- uses decision-time 30s last price as evaluation origin,
- computes 1m/3m/5m MAE, MFE and +/-40bp first-touch from WebSocket aggTrades,
- fail-closed on incomplete/restarted tape,
- no REST market-data calls,
- no routing/order side effects.

Analyzer:
- `research/short/analyze_short1_entry_hazard_policy_ws_v42.py`
- exact join on `policy_event_id`
- reports all shadow outcomes and model-gate-eligible (`READY`, `PENDING_CONTEXT`) outcomes separately.

Cost/next consolidation:
- development currently uses two INJ aggTrade WebSocket consumers (policy + outcome) for isolation.
- after validation, low-cost collector should use one shared aggTrade connection feeding both state machines internally / through a local bus.

## V43 single-WebSocket cutover

Validated live parity before cutover:
- policy parity on 2 source candidates / 4 stage events: state equal 4/4
- max absolute flow20 difference: 0.0
- max absolute ret20 difference: 0.0
- max absolute ret30 difference: 0.0
- outcome parity on a live 5m window: exact tick counts and 1m/3m/5m touch/MAE/MFE deltas 0.0

Cutover:
- ACTIVE/ENABLED: `binance1-short1-entry-hazard-v43.service`
- DISABLED/INACTIVE rollback references:
  - `binance1-short1-entry-hazard-shadow.service`
  - `binance1-short1-entry-hazard-outcome-ws.service`
- V43 policy: `telemetry:short1:inj:entry_hazard_policy_v43_shadow`
- V43 outcome: `telemetry:short1:inj:entry_hazard_policy_v43_outcomes_ws`
- analyzer: `research/short/analyze_short1_entry_hazard_v43.py`

Environment authority:
- `SHORT1_HAZARD_FLOW20_MAX=-0.60`
- `SHORT1_HAZARD_FLOW_DELTA_5_20_MAX=-0.30`

These remain SHADOW-ONLY research thresholds. No production routing/order veto exists.
