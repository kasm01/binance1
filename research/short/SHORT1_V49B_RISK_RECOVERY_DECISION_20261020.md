# SHORT1 V49B Risk Recovery Research Decision — 2026-10-20

Research only. No production settings changed.

## Context
V47 Entry Hazard improved annual return and MaxDD, but May 2026 deteriorated because improved April equity moved Risk Master from REDUCED to NORMAL before the next adverse regime. Same May entries could therefore be ~2x larger.

## Tests
- Fixed-time RECOVERY (6h/12h) was rejected: empty time is not evidence of safe recovery.
- Trade-confirmed RECOVERY tested with 70% and 75% sizing.
- RECOVERY enters when realized DD recovers to <=4% from peak.
- Any recovery stop sends state back to REDUCED.
- TP closes count as successful recovery trades.
- NORMAL requires N successful recovery trades.

## Best research candidate
RECOVERY sizing = 70% equity
NORMAL promotion = 2 successful RECOVERY trades

Results:
- Annual return: +235.21%
- PF: 1.19833
- MaxDD: -20.03%
- April: +2.97%, PF 1.3602, DD -2.84%
- May: -5.03%, PF 0.9381, DD -12.70%
- July: -0.378%, unchanged control month
- September: -7.16%, PF 0.8701, DD -15.70%

Comparisons:
- C600 baseline annual: +225.89%, PF 1.1913, MaxDD -20.74%
- V47 direct recovery annual: +237.49%, PF 1.1974, MaxDD -20.03%, May -7.33%
- V49B R70/W1: +229.97%
- V49B R70/W2: +235.21%  <-- best balance
- V49B R70/W3: +230.22%
- V49B R75/W2: +231.58%

Decision: keep as research candidate only. Do not modify production Risk Master yet. Next validation should test robustness by regime/time split and shadow-state implementation before any routing effect.
