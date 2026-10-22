# SHORT1 V47 Path-Dependent Hazard Replay — 2026-10-20

## Scope
- Canonical C600/V30 state-machine replay preserved.
- Historical microstructure reconstructed for all 15,810 full-minute confirmation candidates across 266 days.
- V44 shadow contract applied only as hazard overlay:
  - 20s confirmed + flow20 <= -0.60 + flow(20)-flow(5) <= -0.30 => PENDING
  - 20->30s price rebound/non-down => REJECT_CANDIDATE
  - 20->30s price continues down => READY_AFTER_REVALIDATION
  - 20s NOT_APPLICABLE falls back to baseline confirmation; it is NOT a reject.
- Order-book V44 remains telemetry-only and is NOT used for PnL routing.

## Baseline parity
- closed: 1102 vs canonical 1102
- net PnL: 2259.585051138395 vs canonical 2259.585051138395
- absolute difference: 0.0

## Annual comparison
Baseline C600:
- return +225.8933%
- PF 1.19133
- MaxDD -20.7434%
- entries 1103 / closed 1102

V47 hazard path:
- return +237.4934%
- PF 1.19742
- MaxDD -20.0266%
- entries 1105 / closed 1104
- delta return +11.6001 pp
- MaxDD improvement +0.7169 pp
- 524 PENDING; 273 REJECT_CANDIDATE; 251 READY_AFTER_REVALIDATION

## Calendar comparison
- Daily active positive rate: 60.83% -> 61.25%
- Daily p05 worsened: -2.51% -> -3.47% (risk-state/sizing interaction)
- Weekly active positive rate: 66% -> 66%
- Worst week: -11.60% -> -10.55%
- Monthly positive rate unchanged: 76.92%
- Worst month MTM return: -8.22% -> -7.46%

## Deep-dive months
### 2026-04
- Return: +1.8930% -> +2.9677%
- PF: 1.206 -> 1.360
- period MaxDD: -4.375% -> -2.840%
- removed baseline entries: 2; added entries: 2
- removed included -44.75 stop and +13.86 TP; replacement path net better.

### 2026-05
- Return: -6.7867% -> -7.3342% (worse)
- PF: 0.918 -> 0.912
- period MaxDD: -12.512% -> -14.869% (worse)
- Entry selection normalized by PnL/notional improved (-0.2947 -> -0.2349 aggregate normalized PnL), but Risk Master sizing state changed.
- Baseline May entries were only ~25% or ~50% equity; V47 had four ~100% equity entries.
- May 1 and May 8 stop trades were ~50% sizing baseline but ~100% sizing V47, doubling loss magnitude.
- Conclusion: May deterioration is primarily ENTRY FILTER x DD/RISK-SIZING STATE interaction, not simple evidence that V44 selection is worse.

### 2026-07
- Return effectively unchanged: -0.3776% -> -0.3776%
- PF unchanged 0.926; MaxDD unchanged -3.202%
- No baseline entries removed/added.
- Four baseline losses: 2 NORMAL, 2 NOT_APPLICABLE; none matched PENDING/REJECT hazard.
- Conclusion: current extreme-sell hazard does NOT address July failure mode. Do not tighten same threshold to solve July.

### 2026-09
- Return: -8.2248% -> -7.4637%
- PF: 0.855 -> 0.870
- period MaxDD: -16.686% -> -15.699%
- Sep 19 improves by ~+1.11 pp daily return; PnL -238.84 -> -208.51.
- Removed 5 baseline entries and added 5 alternate-path entries.
- Removed losses included -65.55, -32.11, -31.78; removed winners +24.45 and +11.00.
- Added path included two large stops (-67.39, -66.02) plus three wins (+20.97,+18.81,+23.56).
- Net month improved, but V44 does not solve all September loss modes.

## Interpretation
1. V47 produces a modest but real full-path annual improvement with exact baseline parity.
2. The effect is regime dependent; May demonstrates interaction with central Risk Master sizing state.
3. July failures are a different entry-quality regime and require separate research (20s non-confirm / BUY_UP / moderate SELL_DOWN cases), not more aggressive extreme-sell thresholds.
4. September improves but still contains many NORMAL/NOT_APPLICABLE losses; order-book persistence and delayed confirmation are appropriate next research axes.
5. No production routing change. V44 remains shadow/telemetry only.
