# SHORT Architecture Refactor — Final Baseline — 2026-10-13

## Target flow
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

## Production process authority
- `ORCH_ONLY_MODE=1`
- `SCAN_ENABLE=0`
- `EXEC_EVENTS_ENABLE=1`
- Production signal source:
  `model_worker → aggregator → top_selector → master_executor → intent_bridge → exec_events → TradeExecutorShort`
- Local HeavyEngine path is debug/fallback only and cannot run in parallel in orchestrated production.
- Orch-only `main.py` now uses minimal execution bootstrap: Binance client + Redis + TradeExecutorShort + sync/lifecycle + exec-events consumer.
- No local HybridModel, MTF ensemble, WhaleDetector, DepthWS, RiskManager, PositionManager, or HeavyEngine is created in orch-only execution process.

## MODEL contract
- Model worker requires coin-specific model routing.
- `ALLOW_GENERIC_MODEL_FALLBACK=0` (fail closed).
- Canonical decision score is `model_score` only.
- Debug model probability lives only in metadata (`meta.p_buy`).
- Removed model aliases: `score_edge`, model `confidence`, `p_single`, `fast_model_score`.
- Current SHORT repo has no validated coin-specific LINK/AVAX/INJ 3m model yet, so MODEL layer intentionally emits no trade signals until models are created.

## SIGNAL NORMALIZATION
- Aggregator is normalize-only.
- Canonical fields include:
  `symbol, interval, side, price, model_score, model_provider, dedup_key, atr_pct, spread_pct, liq_score, whale_score, whale_dir, whale_action`.
- `raw` is telemetry/debug only and is not a hard-decision authority.
- Removed `cooldown_key` alias; `dedup_key` is the only event identity/dedup field.

## MARKET CONTEXT
- MasterExecutor is the single MARKET CONTEXT hard-gate authority.
- Master reads canonical top-level context only; no `raw → raw → raw` fallback.
- Unknown whale context (`whale_dir=none`, score 0) is neutral.
- Explicit contra/hard-block context can reject when real evidence exists.
- TopSelector never hard-gates market context.

## MODEL SCORE GATE
- Single hard authority: MasterExecutor `MASTER_MIN_TRADE_SCORE=0.501`.
- TopSelector is ranking-only and ranks by canonical `model_score`.
- Removed TopSelector hidden ACP floors, boosts, penalties, hard rejects and score threshold aliases.
- Master publishes `model_score_gate_passed=true` + threshold contract.
- Executor only verifies upstream gate in orchestrated mode; it does not apply a second production threshold.
- Legacy SHORT production env thresholds removed: MIN_SCORE, OPEN_MIN_SCORE, SHORT_THRESHOLD, coin min score and dynamic score keys.

## SNIPER
- Single authority: `TradeExecutorShort._evaluate_short_sniper_stage`.
- Progress calculated once.
- Active config only:
  - `SNIPER_ENTRY_INTERVAL=1m`
  - `SNIPER_ENTRY_MIN_CANDLE_PROGRESS=0.03`
  - `SNIPER_ENTRY_MAX_CANDLE_PROGRESS=0.25`
- Removed `SNIPER_ENTRY_ENABLE` toggle and duplicate `_check_sniper_entry` helper.

## ENTRY CONFIRMATION
- Single authority: SHORT reversal candle confirmation.
- Active config only:
  - `ENTRY_CANDLE_DIR_INTERVAL=1m`
  - `SHORT_CURRENT_REVERSAL_MIN_PCT=0.0001`
- Removed `ENTRY_CANDLE_DIR_FILTER_ENABLE` toggle.
- Disabled/unused ShortContinuationEngine deleted with all `SHORT_CONTINUATION_*` env keys.
- Confirmation miss returns PENDING rather than permanent reject.

## PENDING REVALIDATION
- Independent from lifecycle enable flag.
- Uses the same market-context contract, sniper, confirmation and execution-risk authorities.
- No duplicate manual `position_exists` hard veto.
- Existing semantics retained: pending revalidation occurs on the next candle's early window.

## EXECUTION RISK
- Single authority: TradeExecutorShort.
- Owns:
  - `position_exists`
  - same-symbol reopen cooldown
  - `MAX_OPEN_POSITIONS`
- TopSelector, MasterExecutor and IntentBridge no longer apply virtual max-open/cooldown/open-state vetoes.
- IntentBridge is transport-only.
- MasterExecutor ranks/gates and publishes intent only.

## FINAL SAFETY
- Explicit stage immediately before order submission.
- Re-checks the same execution-risk resolver to close race windows.

## ORDER / sizing
- Single sizing resolver in TradeExecutorShort.
- Production currently `SHORT_SIZING_MODE=fixed`.
- Current active values:
  - `BASE_ORDER_NOTIONAL=40`
  - `MAX_POSITION_NOTIONAL=75`
  - `DEFAULT_ORDER_LEVERAGE=2`
  - `MAX_LEVERAGE=2`
- Balance-percent mode remains optional for later tested use.
- Removed unused smart cooldown, dynamic leverage, whale-notional and duplicate order-risk env keys.

## POSITION LIFECYCLE
- Single lifecycle stage owns exit decision.
- Active lifecycle behavior only:
  - ROI hard stop
  - fixed ROI TP arm + giveback
- Active config:
  - `FIXED_ROI_TP_ENABLE=1`
  - `FIXED_ROI_TP_PCT=0.0055`
  - `FIXED_ROI_TP_GIVEBACK=0.0008`
  - `ROI_HARD_STOP_PCT=-0.0060`
  - `POSITION_SYNC_INTERVAL_SEC=20`
  - `POSITION_LIFECYCLE_INTERVAL_SEC=2`
- Removed 57 unused legacy exit/stall/trailing/step-TP/anti-reversal/profile env keys.

## TELEMETRY / OUTCOME FEEDBACK
- Explicit `_emit_short_outcome_feedback` layer.
- Stores model provider, signal score, threshold, entry/exit, pnl/roi, duration and close reason.
- Bounded Redis stream prevents unbounded data growth.

## Process / cost cleanup
- Fixed universe production uses only `SYMBOLS=LINKUSDT,AVAXUSDT,INJUSDT`.
- Removed W1/W2/W3 manual overrides; 3 symbols now fit in one model worker instead of three Python processes.
- `VOLSEL_ENABLE=0`; ineffective volatility-selector startup work removed from production.
- Volatility selector module remains available as an optional research/ops utility.

## Validation
- Full active Python compile: PASS
- `git diff --check`: PASS
- Canonical end-to-end contract smoke: PASS
  - model_score 0.53 preserved worker → executor
  - Master threshold 0.501 passed
  - upstream market-context and model-score contracts verified by executor
  - explicit contra-whale context rejected only in MARKET CONTEXT
- SNIPER smoke: PASS
- ENTRY CONFIRMATION READY/PENDING smoke: PASS
- Lifecycle hard-stop / TP giveback smoke: PASS
- Outcome feedback bounded Redis smoke: PASS
- Minimal orch-only executor bootstrap smoke: PASS
- Production SHORT services/processes: OFF

## Refactor size (active tracked files)
- ~1,257 inserted lines
- ~3,364 deleted lines
- net large reduction in active code surface

## Next phase
MODEL only:
1. choose/create coin-specific SHORT model(s),
2. validate walk-forward + true holdout,
3. keep `ALLOW_GENERIC_MODEL_FALLBACK=0`,
4. only then tune MARKET CONTEXT / score threshold / sniper / confirmation economically.

## MODEL phase addendum — economic contract hardening
- Added shared `research/short/economic_labels.py` as the single research authority for SHORT lifecycle labels.
- Label contract uses next-candle-open entry, conservative same-bar TP/SL tie handling (SL wins), one round-trip fee charge, and records net return, MFE, MAE, first-hit bar, timeout return and outcome.
- Fixed an off-by-one boundary issue found during smoke testing so the final valid candidate row is retained.
- Added `audit_risk_adjusted_economic_3m.py` using the runtime-compatible 20-feature matrix and strict walk-forward -> locked selection -> untouched holdout flow.
- LINKUSDT risk-adjusted result: 0/3 positive WF folds; holdout ~ -7.66 bps, PF ~0.31. Reject.
- AVAXUSDT risk-adjusted result: 0/3 positive WF folds; locked validation threshold selected zero holdout trades, indicating material distribution shift. Reject.
- No new model was promoted to production; production remains fail-closed.

## MODEL score semantics hardening
- `model_worker.py` no longer blindly assumes HybridModel output means `p_buy` and no longer blindly computes `1 - p_buy`.
- Added explicit metadata contract `score_semantics`.
- Accepted legacy-compatible semantics: `short_probability`/`p_short` or `buy_probability`/`p_buy`.
- Missing/unknown semantics => `NO-SCORE`, fail closed.
- This prepares the worker for a future economic SHORT provider without silently reusing old direction-probability assumptions.

## Second dead-code pass
Removed additional definition-only helpers after repo-wide reference checks:
- `orchestration/aggregator/aggregator.py::_env_float`
- `orchestration/executor/intent_bridge.py::_env_float`
- `orchestration/executor/intent_bridge.py::_safe_list`
- `orchestration/executor/master_executor.py::_meta_get`

The remaining single-reference `TradeExecutorShort.execute_decision`, `_position_sync_loop`, and `_position_lifecycle_loop` are external entrypoints and were intentionally retained.

## Dead env candidates (verified unreferenced in active source)
The following keys remain cleanup candidates in `.env`; automatic secret-file editing was not performed:
- `EXEC_LOOP_SLEEP_SEC`
- `EXEC_STREAM_STALL_RESET_SEC`
- `EXEC_STREAM_STALL_WARN_SEC`
- `LIGHT_SCAN_COOLDOWN_SEC`
- `LIGHT_SCAN_ENABLE`
- `LIGHT_SCAN_EVERY_SEC`
- `LIGHT_SCAN_INTERVALS`
- `LIGHT_SCAN_LOOKBACK`
- `LIGHT_SCAN_TOPK`
- `MODEL_HOLD_BAND`
- `MODEL_LONG_THRESHOLD`
- `SCAN_FAST_CONCURRENCY`
- `SCAN_FAST_EVERY_SEC`
- `SCAN_FAST_LIMIT`
- `SCAN_SKIP_ANOM`

`REDIS_EXEC_STREAM` was explicitly retained because `scripts/orch_postmortem.sh` still consumes it.

## INJ aligned 3m economic audit
- Downloaded `INJUSDT_3m_360d.csv` aligned exactly to LINK/AVAX: 172,800 bars from 2025-09-19 20:24 UTC through 2026-09-14 20:21 UTC.
- Risk-adjusted economic model on INJ: 0/3 positive WF folds, WF ~ -7.33 bps; locked threshold selected zero holdout trades. Reject.

## Base lifecycle economics — 3 coin comparison
Current 18-minute proxy (`TP price=0.275%`, `SL price=0.30%`, `round-trip fee=0.08%`) is strongly fee dominated:
- LINK holdout: ~ -8.07 bps, PF ~0.44
- AVAX holdout: ~ -7.85 bps, PF ~0.44
- INJ holdout: ~ -7.99 bps, PF ~0.51
Gross directional expectancy is close to zero; the round-trip fee consumes almost the entire average expectancy.

A 3-coin raw lifecycle grid across horizons 6/12/20/40 and wider TP/SL values also produced `positive_cells=0` for every tested configuration. Best mean configuration was approximately H=40, TP=0.6%, SL=1.0%, mean ~ -6.92 bps.

### Consequence
Do not spend more CPU blindly trying model families on every 3m bar. Next research should evaluate MODEL economics on the actual downstream candidate distribution (SNIPER + ENTRY CONFIRMATION-compatible entry opportunities), with the same shared fee/lifecycle label contract. This better matches the production architecture and avoids training/evaluating on a much noisier all-bar population that production never trades.

## Real entry / lifecycle telemetry audit — 2026-10-13
- Historical log telemetry yielded 834 unique <=15s entry-state observations: 153 CONFIRM, 656 NOT_DOWN, 25 WEAK.
- Confirmation is economically discriminative in the historical candidate distribution. Using the same conservative next-full-minute outcome proxy:
  - current18: CONFIRM -1.27 bps vs non-confirm -7.16 bps (lift +5.88 bps)
  - wide60: CONFIRM +1.71 bps vs non-confirm -7.68 bps (lift +9.39 bps)
  - wide120: CONFIRM +6.76 bps vs non-confirm -3.04 bps (lift +9.80 bps)
- Therefore ENTRY CONFIRMATION remains an active authority; it is not dead/legacy filtering.
- 140 OPENED + 140 CLOSED log events produced 138 sequential real trade pairs.
- Real raw price PnL: +6.21 bps/trade, 65.2% win rate. Reasons: 90 fixed_roi_tp_giveback, 48 roi_hard_stop.
- Fee sensitivity on paired realized trades:
  - 0 bps fee: +6.21 bps, PF 1.54
  - 6 bps fee: +0.21 bps, PF 1.02
  - 8 bps fee: -1.79 bps, PF 0.87
  - 10 bps fee: -3.79 bps, PF 0.75
- At 8 bps estimated round-trip fee: LINK +3.07 bps / PF 1.30; AVAX -3.91 bps; INJ -3.58 bps.
- Raising reversal minimum alone is not robust: >=1.5 bps is near breakeven overall but inconsistent across chronological thirds.
- Historical MODEL score is not economically monotonic on real trades: matched sample Spearman(score, net8) = -0.134; higher score quartiles were worse. Do not tune MASTER_MIN_TRADE_SCORE upward on the legacy score.
- OUTCOME FEEDBACK now records backward-compatible gross pnl/roi plus estimated round-trip fee and estimated net pnl/roi/bps. New telemetry env key: SHORT_ESTIMATED_ROUND_TRIP_FEE_PCT (default 0.0008). Lifecycle behavior is unchanged by this telemetry-only fee estimate.

## MODEL v2 — economic probability candidate — 2026-10-13
- Created single feature authority `features/short_economic.py`.
- Research pool: 48 scale-free features; production candidate V1: 16 features.
- Removed hour_sin/hour_cos from the production candidate after robustness testing; model remains compatible with 24/7 trading.
- Candidate V1 features: rv60, rv20, ret_6, ema100_slope3, ema100_rel, body_pct, ema10_slope3, ret_60, ema10_rel, ret_mean60, atr14_pct, down_frac20, rv10, ema50_slope3, imb_mean6, ret_3.
- Built 834-candidate current-lifecycle proxy dataset using full-minute bars only after the live snapshot minute, avoiding current-minute future leakage.
- Proxy vs real 137 confirmed trades: Spearman ~0.781; realized TP-giveback mapped to proxy TP-giveback ~97.8%, realized hard-stop mapped to proxy hard-stop ~89.4%. Use proxy as ranking target, not exact PnL estimator.
- Entry confirmation remains economically useful under the lifecycle proxy: CONFIRM +3.13 bps / PF 1.25; NOT_DOWN -4.97 bps / PF 0.72.
- Pooled sparse Ridge without time-of-day features: 3/3 positive WF folds, worst ~+7.03 bps; holdout selected 7 trades ~+14.30 bps / PF 3.63.
- Strict day-forward Ridge: 4/4 trading test days positive, 32 simulated selected; 29 matched realized trades ~+8.81 bps net8, win ~79.3%, PF ~1.97.
- Economic LogisticRegression V1 (16 features) provides a 0..1 probability score. WF: 3/3 positive folds, worst ~+9.32 bps. Holdout selected 7 trades ~+14.30 bps / PF 3.63.
- Strict day-forward logistic: 30 selected, 27 matched realized trades ~+6.25 bps net8, win ~77.8%, PF ~1.64.
- Research-only artifact created: `research/short/models/short_economic_logistic_v1.joblib` + `.json`; metadata `status=research_only`, `promoted=false`, `score_semantics=short_economic_probability`.
- Added `models/short_economic_inference.py` fail-closed loader and inference smoke PASS.
- `model_worker.py` score contract now accepts `short_economic_probability` directly; unknown semantics still fail closed.
- Artifact is NOT copied to production model directory and is NOT wired as production signal source.
- Production SHORT remains OFF.

### Next MODEL step
Run the economic model in shadow-only mode on live completed 3m bars, record probability + 16 features + downstream confirmation state + realized fee-aware outcome, and require materially larger forward evidence before promotion.

## SHORT1 coin-specific model architecture checkpoint
- Final model policy fixed: 1 strategy engine = 1 side = 1 coin = 1 separately trained model.
- Pooled LINK/AVAX/INJ Economic V1 retained as research-only feature/label prototype; pooled shadow disabled.
- Added `research/short/train_short_coin.py`: coin-filter-first, 16 scale-free SHORT economic features, logistic probability score, nested day-forward validation and strict untouched final-day holdout.
- Corrected an initial validation leakage during implementation: top-fraction selection is now performed only on a prior validation day; test/holdout labels are not used to tune fraction/threshold.
- Candidate comparison under identical single-coin protocol:
  - LINKUSDT: day-forward 17 selected trades, weighted -3.17 bps, worst day -7.80 bps; reject as SHORT1 for now.
  - AVAXUSDT: day-forward 7 selected trades, weighted -5.38 bps; strict holdout -3.31 bps; reject.
  - INJUSDT: day-forward 23 selected trades, 3/3 selected days positive, weighted +4.64 bps, worst +0.44 bps; strict holdout 4 trades +10.32 bps, PF 2.09. Selected as SHORT1 research candidate.
- SHORT1 metadata: `research/short/models/short_model_meta_INJUSDT_3m.json`, status=research_only, promoted=false, reference full-train threshold ~0.6601.
- Added `models/short_model_registry.py` mirroring LONG symbol+interval routing. Production defaults to production MODELS_DIR and never falls back to research artifacts. `require_promoted=True` rejects INJ research artifact as expected.
- Updated shadow collector to resolve coin-specific artifacts through ShortModelRegistry. Single-coin INJ smoke produced p_short_economic ~0.9855 with routing=OFF.
- SHORT1 shadow systemd definitions now target only INJUSDT and isolated Redis namespaces `telemetry:short1:inj:*`; services remain disabled/inactive.
