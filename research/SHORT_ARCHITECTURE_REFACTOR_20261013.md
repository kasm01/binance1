# SHORT Architecture Refactor — 2026-10-13

## Locked target flow

MODEL
→ SIGNAL NORMALIZATION
→ MARKET CONTEXT
→ MODEL SCORE GATE
→ SNIPER
→ ENTRY CONFIRMATION (READY / PENDING / REJECT)
→ PENDING REVALIDATION
→ EXECUTION RISK
→ FINAL SAFETY
→ ORDER
→ POSITION LIFECYCLE
→ TELEMETRY / OUTCOME FEEDBACK

## Single-authority map

- MODEL: `orchestration/scanners/model_worker.py`
  - Binance completed Futures candles
  - shared `features.runtime.build_runtime_features`
  - coin-specific `ModelRegistry/HybridModel`
  - canonical side-specific `model_score`
  - fail-closed when coin-specific model is absent
  - generic fallback disabled in production config
  - no random/fake whale/liquidity data
- SIGNAL NORMALIZATION: `orchestration/aggregator/aggregator.py`
  - normalize/validate canonical event fields only
  - no score recompute, no hard gate, no cooldown, no top-k
- Ranking: `orchestration/selector/top_selector.py`
  - canonical `ranking_score`
  - ranking only; no MODEL/MARKET/EXECUTION hard veto
- MARKET CONTEXT hard gate: `MasterExecutor._evaluate_market_context_stage`
- MODEL SCORE GATE: `MasterExecutor._evaluate_model_score_gate_stage`
  - canonical `model_score` only
  - threshold: `MASTER_MIN_TRADE_SCORE`
- SNIPER: `TradeExecutorShort._evaluate_short_sniper_stage`
- ENTRY CONFIRMATION: `TradeExecutorShort._evaluate_short_entry_confirmation_stage`
  - reversal confirmation only
  - READY / PENDING / REJECT contract
- PENDING REVALIDATION: SHORT pending manager; reuses same downstream authorities
- EXECUTION RISK: SHORT executor only
  - position exists
  - same-symbol open cooldown
  - max-open positions
- FINAL SAFETY: immediate pre-order recheck of the same execution-risk authority
- ORDER: `TradeExecutorShort.open_position_from_signal`
  - sizing/leverage owned by SHORT executor only
- POSITION LIFECYCLE: `TradeExecutorShort._evaluate_short_position_lifecycle_stage`
- TELEMETRY / OUTCOME FEEDBACK: `TradeExecutorShort._emit_short_outcome_feedback`
  - structured log
  - bounded Redis stream `telemetry:short_outcomes`

## Production signal source

Only orchestrated production path is allowed:

`model_worker → aggregator → top_selector → master_executor → intent_bridge → exec_events → TradeExecutorShort`

Config:
- `ORCH_ONLY_MODE=1`
- `SCAN_ENABLE=0`
- `EXEC_EVENTS_ENABLE=1`

Local `HeavyEngine` remains debug/fallback only and cannot run in parallel in orchestrated mode.

## Canonical score contract

Only two score concepts remain in active SHORT orchestration:

- `model_score`: immutable side-specific MODEL output. Hard MODEL SCORE GATE uses only this value.
- `ranking_score`: context-aware ranking value used only to order/select candidates.

Context/ranking may never lift a weak `model_score` through the hard model gate.

Master-issued intents carry:
- `market_context_passed=true`
- `model_score_gate_passed=true`
- `model_score_gate_threshold=<threshold>`
- `model_provider=hybrid_coin_specific`

TradeExecutorShort verifies these contracts and fails closed if either upstream gate contract is missing.

## Removed duplicate / legacy paths

- random production `worker_stub.py`
- duplicate `core/signal_event.py` schema family
- old Aggregator scoring module and quality gates
- Aggregator duplicate cooldown / whale veto / score recompute / top-k
- `DupGuard` legacy module
- TopSelector hidden model-score floor / min-score gate
- TopSelector whale hard gate
- TopSelector selector re-entry cooldown
- repeated ACP limiter passes
- Master synthetic open-position/max-open/symbol-open veto
- Master second sizing/leverage authority
- Master fake heavy-stage hard discard
- Master transport publish cooldown
- IntentBridge synthetic open-state/max-open/dedup/close-cooldown veto
- IntentBridge direct sizing/trailing/stall payload
- direct OPEN fallback in exec-events consumer (all opens must call `execute_decision`)
- parallel `economic_short` MODEL provider
- old continuation engine and `SHORT_CONTINUATION_*`
- old EMA/early-entry/early-TP/fake-breakout/anti-reversal/dead-trade/whale-sizing legacy envs
- obsolete TopSelector/Bridge/Master env knobs verified unused

Deleted active legacy files:
- `core/short_continuation_engine.py`
- `core/signal_event.py`
- `orchestration/scanners/worker_stub.py`
- `orchestration/aggregator/scoring.py`
- `orchestration/state/dup_guard.py`
- `models/short_economic_inference.py`

## Runtime feature authority

`features/runtime.py::build_runtime_features` is the single live feature builder.
`main.py` and `model_worker.py` use the same function.

TensorFlow/Keras import in `HybridModel` is lazy. SGD-only model workers do not pay TensorFlow startup cost.

## Current MODEL safety state

`ALLOW_GENERIC_MODEL_FALLBACK=0`.
Current configured symbols do not yet have validated coin-specific 3m models in the SHORT repo. Therefore MODEL workers fail closed and produce no trading signal until the next MODEL phase creates/validates the model artifacts.

This is intentional and safer than the previous random/generic fallback behavior.

## Verification

- Full active Python compile: PASS
- Modified shell syntax: PASS
- `git diff --check`: PASS
- Synthetic contract smoke:
  - model_score 0.61, price 1.25 → Master intent PASS
  - model_score 0.49 → Master MODEL SCORE GATE REJECT
  - missing market-context contract → executor REJECT
  - missing model-score contract → executor REJECT
  - both upstream contracts present → executor READY
- Production services were not started during refactor.
