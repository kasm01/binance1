# SHORT1 Single-Position Candidate — 2026-10-14

## Architecture decision

Global 8-bot target:

- SHORT1..SHORT4: one coin + one direction + one coin-specific model each.
- LONG1..LONG4: one coin + one direction + one coin-specific model each.
- One bot/coin can own at most one open exchange position.
- No scale-in, averaging, Layer-2 or Layer-3.
- Worst-case position count is 8, subject to future CENTRAL RISK MASTER limits.
- Existing open position must close by lifecycle before that bot can open again.

## SHORT1 current research core

- Coin: INJUSDT
- Model timeframe: 3m
- Execution/confirmation replay: 1m
- Base model score threshold: 0.540
- Position mode: single
- Normal position notional: 100% equity
- Reduced position notional: 50% equity
- DD reduce: 8%
- DD recover: 4%
- Leverage: 3x
- TP arm: +4.00% ROI
- TP giveback: 0.24% ROI
- Hard/soft stop trigger: -8.97% ROI equivalent
- Catastrophic adverse price: +3.50%
- Stop cluster: 3 stops / 6h -> single-shot 90m new-entry pause; re-arm after stop6 < 3

## Entry confirmation

- Previous completed 1m candle green.
- Current 1m candle reverses down.
- Minimum down reversal 0.01%.
- Research replay executes conservatively at next 1m open after confirmation.

## Money Flow

### SHORT_EXHAUSTION — active research candidate

- 60m net taker-flow ratio <= -20%.
- Action: block NEW SHORT entry.
- Never force-close an already open position.
- Rationale: excessive aggressive sell flow behaved as late/crowded SHORT / seller-exhaustion risk.

### EXTREME_LONG_PRESSURE — telemetry/context only

- 15m net taker-flow ratio >= +10% AND 1m volume z-score >= 2 in research.
- Direct hard L1 veto was not preferred in 1m replay.
- Score-gate premium +0.005 increased return but worsened DD / negative-month count.
- Score-gate premium +0.010 also worsened stability.
- Decision: no hard veto and no score premium for now; keep as telemetry/context evidence.
- 1m volume-z cannot be reproduced exactly from 3m aggregated volume (corr ~0.61), so no extra API request is justified yet.

## 1m execution replay reference

L1-only:
- Return +66.35%
- PF 1.0528
- MaxDD -32.44%
- Worst month -21.11%

L1-only + SHORT_EXHAUSTION(-20%):
- Return +83.75%
- PF 1.0648
- MaxDD -30.39%
- Worst month -19.36%
- Entries 1288/year (~3.53/day)

Rolling comparison, EXH20 vs L1-only:
- 30d worst: -20.74% vs -22.95%
- 90d p05: -14.72% vs -17.66%
- 180d worst: -22.57% vs -29.17%
- 180d positive windows: 74.2% vs 64.5%

## Production refactor status

`core/trade_executor_short.py`:
- single-position risk authority implemented.
- existing position -> `position_exists` fail-closed.
- scale-in/layer decision logic removed.
- sizing mode `equity_single`.
- NORMAL 100% / REDUCED 50% equity sizing retained.

`.env`:
- `SHORT_SIZING_MODE=equity_single`
- `SHORT_NORMAL_POSITION_EQUITY_PCT=1.00`
- `SHORT_REDUCED_POSITION_EQUITY_PCT=0.50`
- old layer settings removed/commented.

Production order services remain OFF.

## Money-flow live pipeline validation

- Existing ModelWorker 3m Binance futures kline fetch already contains quote volume and taker-buy quote volume.
- 15m and 60m taker-flow ratios computed from 3m bars are exactly equal to the aligned 1m aggregation in the one-year INJ sample (corr=1.0, zero numerical error).
- Therefore SHORT_EXHAUSTION requires no new API request / collector.
- ModelWorker -> Aggregator -> MasterExecutor MARKET CONTEXT flow fields added.
- Master MARKET CONTEXT rejects a NEW short intent with reason `short_money_flow_exhaustion` when 60m flow <= -0.20.
- Existing open position lifecycle is untouched.

Score-premium research (L1-only + EXH20):
- premium 0.000: return +83.75%, PF 1.0648, MaxDD -30.39%, 5 negative months.
- premium 0.005: return +91.33%, PF 1.0676, MaxDD -30.73%, 6 negative months.
- premium 0.010: return +85.69%, PF 1.0661, MaxDD -31.18%, 6 negative months.
- Stability preference -> premium 0.000 retained.

## OI / Money-Flow refinement (2026-10-14)

Research-only validation on 1m execution replay:

- Binance 5m OI archive cached for INJUSDT: 2025-10-12 -> 2026-10-13, 105,696 rows, zero 5m gaps.
- SHORT_EXHAUSTION remains: 60m net taker flow <= -20% -> reject new SHORT.
- OI hard-block policies were rejected: they reduced return too much.
- Best architecture is a position-risk multiplier, not an entry veto.

Causal regime definition before entry:
- OI 15m delta > 0
- completed 15m price return and completed 15m taker-flow have the same non-zero sign
- interpretation: confirmed/crowded directional position build

Action:
- keep the trade
- single position only
- multiply the normal/reduced position size by 0.50
- NORMAL: 100% equity -> 50%
- REDUCED: 50% equity -> 25%

Live-safe prev-completed-1m replay (V23):
- return: +87.16%
- PF: 1.0885
- max DD: -25.59%
- worst month: -10.45%
- negative months: 4
- May 2026: -9.63%
- Sep 2026: -10.45%
- entries: 1,288 (same as base)

Baseline L1-only + EXH20:
- return: +83.75%
- PF: 1.0648
- max DD: -30.39%
- worst month: -19.36%
- negative months: 5

Rolling robustness, base -> V23:
- 30d worst: -20.74% -> -14.06%
- 30d p05: -15.88% -> -10.56%
- 90d p05: -14.72% -> -9.25%
- 180d worst: -22.57% -> -1.74%
- 180d positive windows: 74.2% -> 98.4%

Estimated turnover / fee impact:
- entry notional: 1.381m -> 1.071m USDT
- approximate round-trip taker fees @4bps/side: 1104.63 -> 856.49 USDT
- entry count unchanged, so improvement comes from context-aware risk sizing rather than fewer opportunities.

Live integration rule remains NOT enabled yet. Production order services stay OFF until the OI context transport and execution-risk contract are implemented and smoke-tested.


## Canonical market-context / Risk Master integration (2026-10-14)

Implementation status: code-complete for transport/core contracts, production-disabled.

- Added `market_context.v1` transport: ModelWorker -> Aggregator -> MasterExecutor -> IntentBridge -> SHORT executor.
- MARKET CONTEXT now classifies OI regime only; it does not own position sizing.
- Shared classifier: `orchestration/context/market_context_contract.py`.
- Added stateless central `risk_decision.v1` core: `orchestration/risk/risk_master.py`.
- Risk Master combination rule validated:
  - NORMAL x normal context = 1.00
  - REDUCED x normal context = 0.50
  - NORMAL x confirmed-build context = 0.50
  - REDUCED x confirmed-build context = 0.25
- OI collector now returns latest + previous snapshot from the same request (`limit=5`), enabling strictly-prior causal selection at 5m timestamp boundaries with no extra request count.
- Added `entry_market_context.v1` helper using only completed 1m bars for V23-style pre-order context and strictly-prior OI selection. Real INJUSDT smoke passed.
- OI context classification and OI risk reduction remain disabled by default. No production service was started.
- `binance1-orch.service` and `binance1-supervisor.service` remained inactive after validation.

Next validation target before activation: shadow comparison of Master-time `market_context.v1` versus pre-order `entry_market_context.v1`, then decide whether Risk Master must require the pre-order snapshot for OI sizing.


## Canonical market-context / entry-time risk integration (2026-10-14 late)

Implementation state:
- `market_context.v1`: Master classification contract.
- `entry_market_context.v1`: exact pre-order context built from the same 1m rows used by ENTRY CONFIRMATION plus strictly-prior central OI.
- `risk_decision.v1`: shared Risk Master sizing contract.
- `market_context_shadow.v1`: Master-vs-entry telemetry; never changes order decisions.
- Central OI collector user service created as `binance1-oi-context.service`; currently active for research but disabled for autostart.
- Trading services remain inactive.
- OI sizing policy remains OFF (`RISK_OI_CONFIRMED_BUILD_ENABLE=0`).
- Master OI classification and entry-context shadow are ON because they are classification/telemetry only.

Cost / request architecture:
- Central OI remains one request per symbol per 5-minute cycle.
- Entry confirmation 1m fetch changed from limit=3 to limit=20; request count is unchanged.
- The exact 20 rows used by candle confirmation are retained in memory and reused for entry-time 15m price/flow context.
- Shadow reuses the exact pre-order `entry_market_context.v1`; no post-order Binance or Redis reread is required in the normal path.

Historical Master-vs-entry regime comparison (V24, 1,288 entries):
- matched: 1,288 / 1,288
- same normal: 771 (59.86%)
- same confirmed build: 459 (35.64%)
- confirmed build -> normal: 31 (2.41%)
- normal -> confirmed build: 25 (1.94%)
- confirmed-build direction flip: 2 (0.16%)
- any regime/direction change: 58 (4.50%)
- sizing signal changed: 56 (4.35%)
- model-event to confirmation lag: 0m = 668 (51.86%), 1m = 620 (48.14%)

Full lifecycle timing replay (V25):
- MASTER_TIME: return +74.07%, PF 1.0822, MaxDD -25.12%, worst month -12.26%, negative months 5.
- ENTRY_TIME_V23: return +87.16%, PF 1.0885, MaxDD -25.59%, worst month -10.45%, negative months 4.
- Entry-time context therefore adds +13.09 percentage points of return with nearly unchanged MaxDD and materially better worst-month behavior.

Rolling comparison, MASTER_TIME -> ENTRY_TIME_V23:
- 30d worst: -14.21% -> -14.04%
- 30d p05: -11.32% -> -10.46%
- 30d positive windows: 58.33% -> 63.69%
- 90d worst: -18.83% -> -19.34%
- 90d p05: -10.51% -> -9.16%
- 90d positive windows: 74.64% -> 82.25%
- 180d worst: -12.02% -> -1.72%
- 180d p05: -5.54% -> +1.64%
- 180d positive windows: 81.72% -> 98.39%

Turnover / fee proxy:
- MASTER_TIME entry notional: 976,222 USDT; estimated round-trip fee @ 8 bps = 780.98 USDT.
- ENTRY_TIME_V23 entry notional: 1,070,612 USDT; estimated round-trip fee = 856.49 USDT.
- Higher entry-time turnover is intentional: Master-time incorrectly halves some trades that become NORMAL by actual entry. The additional ~75.5 USDT fee proxy accompanies materially higher net replay return.

Architecture decision:
- Master OI context remains useful for classification/telemetry only.
- Position-size OI policy should consume `entry_market_context.v1` after ENTRY CONFIRMATION and before final ORDER sizing.
- If entry context is unavailable, use canonical Master context as neutral-safe fallback; never create a second OI classifier.

## Stop/lifecycle diagnostic (2026-10-15)

Key diagnosis on ENTRY_TIME_V23 baseline:
- 365 losing positions; 60.27% recovered fee-adjusted break-even within 24h, 71.51% within 48h.
- 50.41% reached the ~4% leveraged ROI-arm price region within 24h after being stopped.
- Of 12,389 aligned score>=0.540 + candle-confirmed opportunities, 83.20% occurred while the single SHORT1 position slot was already occupied.
- Of occupied confirmations, 69.80% happened while the open position was underwater.
- Layering remains OFF; multi-coin parallelism belongs to SHORT1-4, not same-symbol scale-in.
- Threshold reduction alone is unsafe: TH0.535 return +61.90%; TH0.530 +93.78% with worse DD/month stability; TH0.520 +38.66% with materially worse DD.

Focused stop search, score remains 0.540:
- S375/C475: return +127.03%, PF 1.1190, MaxDD -25.82%, worst month -14.02%, rolling-180 worst -5.10%.
- S375/C500: return +142.11%, PF 1.1271, MaxDD -25.39%, worst month -13.36%, rolling-180 worst -0.52%.
- S400/C475: return +158.79%, PF 1.1372, MaxDD -27.05%, worst month -12.86%, rolling-180 worst +0.98%.
- S400/C500 (prior WIDE2): return +153.78%, PF 1.1336, MaxDD -26.92%, worst month -12.25%.
- S400/C525: return +171.88%, PF 1.1458, MaxDD -26.21%, worst month -12.47%, rolling-180 worst +1.54%, p05 +9.94%, 100% positive 180d windows.
- S425/C500: return +89.44%, PF 1.0982, MaxDD -28.80%.
- S425/C525: return +93.39%, PF 1.1029, MaxDD -27.70%.

Current provisional stop leader: soft stop ~4.00%, catastrophic ~5.25%. One upper catastrophic boundary check (5.50%) is pending before closing the stop grid.

## SHORT1 lifecycle / soft-entry refinement (2026-10-15/16)

Research decision after full-year causal 1m replay (INJUSDT, one position, no layers):

### Stop region
- Current narrow stop was materially over-aggressive.
- Fine grid supports soft adverse-price stop near 4.00% and catastrophic near 5.00-5.25%.
- S400/C525: return +171.88%, PF 1.14585, MaxDD -26.21%, worst month -12.47%.
- Soft 4.25% degraded sharply, so do not extrapolate wider.

### Stop re-entry
- Natural <=5m re-entries looked poor in isolation, but hard cooldown changes the full lifecycle non-linearly.
- 15m stop-only re-entry lock produced the strongest risk-balanced standalone variant:
  return +162.08%, PF 1.15354, MaxDD -21.08%, worst month -8.62%, 4 negative months.
- Rolling-180: worst -3.19%, p05 +5.40%, median +41.64%, positive windows 98.92%.

### Context-confirmed soft score
- Do NOT simply lower global score threshold.
- 0.530-0.540 with only price+flow alignment increased tail risk.
- 0.530-0.540 only when strictly causal entry context confirms SHORT build
  (oi15 > 0, completed ret15 < 0, completed flow15 < 0) was strong:
  return +179.50%, PF 1.1530, MaxDD -24.40%, 3 negative months, worst month -10.61%.
  rolling-180 worst +10.15%, p05 +17.18%, positive windows 100%.
- Raising soft floor to 0.535 reduced performance (+163.42%), so 0.530 remains the research candidate.

### OI direction policy
- Keep confirmed-build x0.50 sizing for BOTH directions.
- Adverse-LONG-only x0.50 underperformed: +118.19%, PF 1.09847, DD -28.42%.
- OI sizing OFF underperformed badly: +111.97%, PF 1.08058, DD -34.49%, worst month -19.19%.

### Combined candidate
Hard score >=0.540; soft score 0.530-0.540 only SHORT-build context; no layers;
soft stop 4.00% adverse price with live 30s confirmation; catastrophic 5.25%;
15m stop-only same-symbol re-entry lock; money-flow60 exhaustion <= -20% reject;
confirmed-build OI sizing x0.50.

Full-year result:
- return +176.13% (MTM reconstruction +176.18%)
- PF 1.15836
- MaxDD -22.57%
- entries 1,105
- negative months 3
- worst month -10.89%
- first 6m +111.02%
- last 6m +30.86%
- rolling 30d worst -14.01%, p05 -9.31%, positive 68.45%
- rolling 90d worst -14.04%, p05 -5.70%, positive 85.87%
- rolling 180d worst +7.80%, p05 +15.69%, median +43.63%, positive 100%

Architecture rule for live integration:
- score >= 0.540 => normal MODEL SCORE READY
- 0.530 <= score < 0.540 => PENDING_CONTEXT, never direct READY
- after ENTRY CONFIRMATION, exact entry_market_context.v1 must confirm SHORT build to promote to READY
- score < 0.530 => REJECT
- keep a single score authority; executor must consume upstream state rather than create an independent threshold.

Causality caveat: historical soft-stop confirmation approximates unavailable 30s data using two consecutive 1m breaches. Live uses 30-second confirmation; validate in shadow before live enable.

## V26-V29 lifecycle refinement (2026-10-19 continuation)

### Stop recovery diagnosis
- 365 losing trades in baseline V23.
- Net break-even recovery after stop: 1h 9.04%, 3h 27.67%, 6h 41.37%, 12h 51.78%, 24h 60.27%, 48h 71.51%.
- 24h TP-arm-like favorable recovery: 50.41%.
- Conclusion: baseline stop was materially early, but simply holding stopped positions would block many new confirmations.

### Stop grid V26
Baseline V23 (~2.99% soft / 3.5% catastrophic): return +87.16%, PF 1.0885, MaxDD -25.59%.
Key tested points:
- 3.75 / 5.00: +142.11%, PF 1.1271, MaxDD -25.39%.
- 4.00 / 4.75: +158.79%, PF 1.1372, MaxDD -27.05%.
- 4.00 / 5.00: +153.78%, PF 1.1336, MaxDD -26.92%.
- 4.00 / 5.25: +171.88%, PF 1.14585, MaxDD -26.21%, worst month -12.47%, 5 negative months.
- 4.25 / 5.00: +89.44%, PF 1.0982, MaxDD -28.80%.
- 4.25 / 5.25: +93.39%, PF 1.1029, MaxDD -27.70%.
Interpretation: soft-stop optimum is concentrated near ~4.0%; catastrophic stop has a broader useful region around ~5.0-5.25%.

### Re-entry V27 diagnosis
Natural baseline behavior already re-enters rapidly after losses: 47.7% <=5m, 64.7% <=15m, 71.0% <=30m.
Simple stop-reentry locks on 4.00/5.25:
- 3m: return +141.26%, PF 1.1337, MaxDD -26.92%, worst month -13.12%, 4 negative months.
- 5m: return +103.82%, PF 1.1051, MaxDD -26.74%, worst month -13.31%, 5 negative months.
- 10m: return +113.51%, PF 1.1197, MaxDD -26.90%, worst month -13.66%, 4 negative months.
- 15m: return +162.08%, PF 1.1535, MaxDD -21.08%, worst month -8.62%, 4 negative months.
Interpretation: naive short cooldowns are nonlinear; 15m is a strong production-safety candidate, but must be revalidated with clean persistent equity curves before activation.

### Next persistent replays
- `research/short/replay_short1_reentry_v27.py`: NO_LOCK vs RE15, persistent output.
- `research/short/replay_short1_soft_entry_v28.py`: score 0.530-0.540 allowed only with bearish price+flow alignment, with/without OI>0 short-build confirmation.
- `research/short/replay_short1_oi_direction_v29.py`: compare OI_NONE, BOTH50, and adverse-LONG-only 50% sizing under entry-time context.

Production services remain OFF; no production `.env` activation in this research stage.

## V30 combined candidate update (2026-10-19)

Extended catastrophic boundary with soft stop fixed at 4.00% showed a plateau:
- C550: +186.71%, PF 1.15346, MaxDD -25.77%, 4 negative months.
- C575: +197.34%, PF 1.15696, MaxDD -25.51%, 5 negative months.
- C600: +197.25%, PF 1.15639, MaxDD -24.84%, 5 negative months.
Standalone stop-only return plateaued around 5.75-6.00%; no reason to extrapolate wider.

Combined V30 keeps:
- one position / no layers
- hard score >= 0.540
- 0.530 <= score < 0.540 only if exact entry-time context confirms SHORT build
- money-flow60 exhaustion <= -20% reject
- confirmed-build OI sizing x0.50 for both build directions
- 15-minute stop-only same-symbol re-entry lock
- soft stop 4.00% adverse price, historical two-consecutive-1m confirmation approximation
- catastrophic stop candidate 5.75 vs 6.00

C575 combined:
- return +185.28%
- PF 1.16590
- MaxDD -20.74%
- 1,105 entries / 1,104 closed
- win rate 76.99%
- 3 negative months; worst month -7.92%
- first 6m +111.05%; last 6m +35.17%
- rolling 30d worst -11.15%, p05 -7.77%, positive 67.26%
- rolling 90d worst -12.57%, p05 -3.59%, positive 88.41%
- rolling 180d worst +6.80%, p05 +15.63%, median +44.14%, positive 100%

C600 combined — current research leader:
- return +225.89%; final MTM equity 3,259.53 from 1,000 start
- PF 1.19133
- MaxDD -20.74%
- 1,103 entries / 1,102 closed
- win rate 77.04%
- 3 negative months; worst month -7.92%
- first 6m +123.58%; last 6m +45.76%
- rolling 30d worst -11.15%, p05 -7.67%, median +4.17%, positive 71.13%
- rolling 90d worst -12.57%, p05 -2.84%, median +25.06%, positive 89.13%
- rolling 180d worst +14.30%, p05 +23.75%, median +55.43%, positive 100%
- close reasons: 849 TP giveback, 249 confirmed soft stops, 4 catastrophic stops

Research decision: use 6.00% catastrophic as the next shadow/live-validation candidate, not because standalone return is higher (it is essentially flat vs 5.75) but because the full combined lifecycle is materially stronger and catastrophic events are rare. Do not widen beyond 6.00% without new evidence.

Causality/live caveat remains: historical soft stop uses two consecutive 1m breaches as approximation; live executor uses 30-second confirmation. Production activation requires shadow/live-safe validation before enabling trading services.

## SHORT1 V30 live-shadow promotion phase — 2026-10-19

### Final research candidate (C600)
- hard score >= 0.540
- soft score 0.530–0.540 only after exact entry-time `confirmed_build + short`
- one position, layers OFF
- soft stop: 4.00% adverse price / ROI -12% at 3x, live 30s persistence confirmation
- catastrophic stop: 6.00%
- stop-only same-symbol re-entry lock: 15m
- money-flow60 exhaustion <= -20% => REJECT
- confirmed-build Risk Master sizing x0.50 (NORMAL=.50, REDUCED=.25)
- 365d C600 combined replay: +225.89%, PF 1.1913, MaxDD -20.74%, worst month -7.92%, 3 negative months, rolling-180 worst +14.30%.

### Critical recent-forward warning
The 365d replay uses causal WF V4 scores. The research final/allbar live-shadow model was not a promoted production artifact and used an older training cutoff. A real shadow-event replay over 2026-10-13 20:09 UTC through 2026-10-19 ~18:06 UTC produced 5 V30 entries, 4 closed (1 win / 3 losses), realized -10.72%, MTM about -11.06%, PF ~0.10, MaxDD ~-12.56%. This is serious but remains inside the historical V30 6-day tail: historical rolling-6d worst ~-13.57%, p01 ~-10.72%.

Do not overfit a high-score veto from this short run. Historical C600 score >=0.60 was strong (PF ~1.42). The correct response is forward validation, not adding another threshold.

### Fresh V5 model check
Fresh V5 candidate trained on the 460d source through 2026-10-13 19:53:59 UTC; last label timestamp 18:50:59 UTC, before the first shadow event at 20:09, so the forward shadow window is leakage-free. Same model family/features/target as V3. Fresh-vs-old score correlation on real shadow events was ~0.982 and Fresh V5 produced the same five V30 forward entries and same recent-forward loss. Therefore model staleness alone did not explain the adverse week.

Fresh V5 is intentionally NOT promoted. Production `models/` still has no INJ promoted artifact. Trade services remain disabled/inactive and system remains DISARMED (`ARMED=0`, `LIVE_KILL_SWITCH=1`).

### Exact V5 live shadow architecture
- `binance1-short-shadow.service`: non-routing V5 score + 1m confirmation collector.
- stream: `telemetry:short1:inj:model_shadow_v5`
- provider: `short_entry_quality_fresh_logistic_v5`
- same 1m request enlarged to 65 rows; request count unchanged. Carries `money_flow_60m` plus exact `entry_market_context.v1` with strictly-prior central OI.
- `binance1-oi-context.service`: enabled/active, one public OI request per 5m for INJ.
- `binance1-short1-v30-shadow.service`: non-routing paper lifecycle, stream `telemetry:short1:v30:lifecycle_shadow_v5`, isolated state `shadow:short1:v30:v5:*`.
- mark price cost control: no price REST calls while flat; while a paper position is open, `/premiumIndex` at 5s cadence. Binance futures websocket handshake worked but delivered no frames on this VM, so websocket was rejected as unreliable.
- V30 lifecycle synthetic contract passed: soft SHORT-build x0.50 open, TP/giveback, hard full-size open, 30s confirmed stop, 15m re-entry rejection.

### Promotion gate
`research/short/evaluate_short1_v30_promotion_gate.py` writes `research/short/results/SHORT1_V30_LIVE_PROMOTION_GATE.json` and never auto-promotes. A 30-minute systemd timer evaluates it using Redis only (no market API cost).

Current status: `OBSERVE` because the exact V5 shadow sample is insufficient. Provider verification already passes. The gate does not issue an early REJECT before at least 15 closed paper positions and 10 observed days. `PRODUCTION_ELIGIBLE` requires at least 40 closed positions, 20 observed days, PF > 1.15, positive realized return and realized MaxDD better than -15%.

`scripts/short1_promote_v5.sh` is manual and fail-closed: it refuses unless gate is `PRODUCTION_ELIGIBLE` AND the system is DISARMED. It has been smoke-tested to refuse at current `OBSERVE` status. Do not bypass this gate.
