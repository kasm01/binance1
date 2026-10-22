# SHORT Architecture Audit — 2026-10-12

Target:
MODEL -> SIGNAL NORMALIZATION -> MARKET CONTEXT -> MODEL SCORE GATE -> SNIPER -> ENTRY CONFIRMATION -> READY/PENDING/REJECT -> PENDING REVALIDATION -> EXECUTION RISK -> FINAL SAFETY -> ORDER -> POSITION LIFECYCLE -> TELEMETRY/OUTCOME FEEDBACK

## Current structural refactor status

- SIGNAL NORMALIZATION: explicit `_normalize_short_signal_stage`
- MARKET CONTEXT: explicit `_build_short_market_context_stage`
- MODEL SCORE GATE: explicit `_evaluate_short_model_score_gate_stage`
- SNIPER: explicit `_evaluate_short_sniper_stage`
- ENTRY CONFIRMATION: explicit `_evaluate_short_entry_confirmation_stage`
- PENDING REVALIDATION: reuses sniper + confirmation + execution-risk authorities
- EXECUTION RISK: explicit `_evaluate_short_execution_risk_stage`
- FINAL SAFETY: common `_resolve_short_open_risk_state` rechecked immediately before order
- ORDER SIZING: single `_resolve_short_order_notional`; `SHORT_SIZING_MODE=fixed|balance_pct`
- Current sizing mode remains `fixed` to preserve production behavior

## Removed duplicate path

Removed obsolete parallel candidate chain:
- `_evaluate_short_candidate_precheck`
- `_evaluate_short_candidate`
- `_apply_short_candidate_decision`

## Dead / misleading env discovered

These are present in `.env` but not referenced by active SHORT code:
- `SHORT_EMA_FILTER_ENABLE`
- `SHORT_EMA99_BELOW_PRICE`
- `SHORT_REQUIRE_EMA7_BELOW_25`
- generic `ENTRY_CONFIRM_ENABLE/BARS/MAX_AGE_SEC` do not control SHORT executor

## Sizing mismatch discovered

Before refactor:
- `.env PER_POSITION_BALANCE_PCT=0.30` existed but SHORT ignored it
- actual order sizing was `BASE_ORDER_NOTIONAL=40`, capped by `MAX_POSITION_NOTIONAL=75`

After refactor:
- `SHORT_SIZING_MODE=fixed` preserves 40 USDT current behavior
- optional `balance_pct` mode is available for later backtest/promotion

## Model audit

Active interval: 3m.

Current loader uses:
- `models/online_model_3m_best.joblib`
- 20 features
- meta `model_meta_3m.json`: best_auc ~0.592858, best_side=long

Existing legacy short artifacts:
- `online_model_1m_short.joblib`
- `online_model_5m_short.joblib`
- `online_model_15m_short.joblib`
- `online_model_1h_short.joblib`

No `online_model_3m_short.joblib` exists.
Legacy short SGD artifacts use 22 features while current 3m/5m best pipeline metadata uses 20 features. Do not hot-swap.

## Next MODEL action

Build a coin-specific SHORT reference starting with LINKUSDT, with a consistent feature schema and economic/OOS validation before promotion.

Production services remain OFF.

## Refactor checkpoint — 2026-10-12

### Entry pipeline now explicit
SIGNAL NORMALIZATION -> MARKET CONTEXT -> MODEL SCORE GATE -> SNIPER -> ENTRY CONFIRMATION -> PENDING REVALIDATION -> EXECUTION RISK -> FINAL SAFETY -> ORDER.

Direct and pending entry paths reuse the same sniper / confirmation / execution-risk authorities. The old parallel candidate/precheck/apply chain was removed.

### Sizing authority
`SHORT_SIZING_MODE=fixed|balance_pct` added. Current runtime remains `fixed`, preserving `BASE_ORDER_NOTIONAL=40`; balance_pct remains inactive until economic validation.

### Dead code removed
Removed obsolete old entry-kline cache/helper chain, unused continuation wrappers, unused pending getter, stale imports, and executor fields that were set but never read.

### Dead env cleanup
Removed legacy EMA short veto keys, old ROI profit-lock/scalp/entry-confirmation/open-filter keys, redundant long-side coin keys, and other keys not read by active SHORT runtime. Whale-family cleanup is intentionally deferred until MARKET CONTEXT refactor.

### Active TradeExecutorShort env surface
33 explicit runtime keys remain (plus dynamically generated `COIN_SHORT_MIN_SCORE_<SYMBOL>` keys).

### Model routing
HybridModel/ModelRegistry now support symbol-specific artifacts and symbol+interval cache keys. `HeavyEngine.run_once(symbol)` resolves the active model per symbol. `ALLOW_GENERIC_MODEL_FALLBACK=1` is temporary until a validated SHORT model exists.

### Research conclusion so far
- Old low-p_buy => SHORT logic is not economically valid on LINK.
- Direct p_short SGD classification is not enough; high classifier score did not translate to negative fee-adjusted future return.
- 160 economic label/threshold combinations produced 0 validation+test jointly positive candidates.
- Raw LINK SHORT lifecycle grid: 0/48 positive after 8 bps round-trip taker fee.
- Nested utility selector showed historical promise but failed genuine Sep14-Oct12 forward data; sequential one-position replay remained strongly negative. LINK rejected as SHORT1 reference.
- Historical fixed-selector screens: AVAX weak/non-robust, NEAR negative, DOGE negative. Candidate universe expansion required.
