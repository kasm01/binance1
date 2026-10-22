# SHORT1 V50 Context-Risk Decision — 2026-10-22

## Purpose
Investigate the May/September weakness of SHORT1 V49B R70/W2 without overfitting a month-specific hard veto.

## Root cause
September W38-W39 losses were concentrated in entries where at least two of the following causal context signals were adverse to a new SHORT:
- completed/available 60m taker flow > 0
- strictly completed 15m taker flow > 0
- strictly-prior OI15 > 0

Baseline annual attribution:
- ADV2=false: 652 trades, +2821.59 PnL, PF 1.4269, win-rate 80.06%
- ADV2=true: 452 trades, -468.81 PnL, PF 0.9108, win-rate 73.45%

This supports a sizing reduction, not a hard entry block.

## Dynamic V50 validation
Same 365d V49B simulator/state machine, same entries/exits; only entry notional is multiplied when ADV2 is true.

| Profile | Return | PF | Max DD | Negative months | May | Jul | Sep |
|---|---:|---:|---:|---:|---:|---:|---:|
| V49B baseline | +235.21% | 1.1983 | -20.03% | 3 | -5.03% | -0.38% | -7.16% |
| ADV2 x0.75 | +249.96% | 1.2264 | -17.08% | 3 | -5.87% | -1.07% | -2.40% |
| ADV2 x0.50 | **+254.37%** | **1.2679** | **-15.49%** | **2** | -3.65% | -1.76% | **+1.69%** |

ADV2 x0.50 affected 453 entries; trade count stayed 1105 entries / 1104 closes.

## May finding
May W21 is not the same regime as September. Its dominant weakness is lack of 20s downward continuation (`NOT_APPLICABLE` in V47 terminology). However `NOT_APPLICABLE` has positive expectancy over the full year (+611.63 PnL, PF 1.1737), so a global veto/reduction would be overfit.

A `REDUCED + NOT_APPLICABLE` overlay is not promoted either: the non-ADV2 subset remains positive expectancy and a second rule would add complexity for a month-specific improvement.

## Decision
1. Promote `ADVERSE_2OF3 x0.50` to **SHORT1 research candidate V50**.
2. Keep it as a sizing multiplier, never a hard block.
3. Do not add a May-specific micro/hazard gate.
4. Keep V47/V44 microstructure telemetry observational until a broadly positive forward/out-of-sample rule is proven.
5. Do not change production/runtime SHORT routing yet.

## Next architecture step
Proceed to the shared Central Risk Master shadow contract using the portfolio MTM research candidate DD profile 12% reduce / 6% recover, with research exposure caps gross 1.00x, long 0.50x, short 0.50x, abs-net 0.50x, symbol 0.50x.