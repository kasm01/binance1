# SHORT1 INJUSDT — Real logged losses 1m/3m/5m entry trends, exit ROI & rebound
14 November 2026. Historical 21 July–2 September 2026 trades in SHORT_REAL_TRADES_WITH_MODEL_SCORE.csv; 60 INJUSDT log-joined trades, 23 roi_hard_stop losses, 37 fixed ROI TP exits.

## Scope and non-lookahead
- Data: local continuous INJUSDT 1m Jul21–Sep2, 3m 2025Jul–2026Oct, 5m 2025Oct–2026Oct. All entry features computed only from FULLY COMPLETED bars before exact logged open timestamp.
- A candle showing bullish close is not equivalent to structural uptrend; 3-bar returns cover 3m in 1m, 9m in 3m and 15m in 5m.
- Post-stop future 1m high/low are observations, not guaranteed fill or PnL. Intrabar exact sequence and funding unknown.
- Historical stop reasons/ROI reflect the OLD SHORT bot from Jul–Sep. They are NOT simulations of the current SHORT1 -8.97% ROI stop / +1.25% ROI TP.
- Log-joined transactions have not been reconciled against exchange trade IDs here; assumed net8 USDT values estimate net from historical net8 return fraction * recorded notional.

## Entry comparison
| Signal at entry | Losses among 23 | Winners among 37 | Conditional loss rate if filter present |
|---|---:|---:|---:|
| Last fully closed 1m bullish | 23 | 37 | 38.3% |
| 1m last three close-to-close bars rise | 15 | 23 | 39.5% |
| 3m last three close-to-close bars rise | 11 | 16 | 40.7% |
| 5m last three close-to-close bars rise | 10 | 19 | 34.5% |
| All three recent frames positive | 6 | 10 | 37.5% |
| Prior 15m and 60m uptrend | 10 | 10 | 50.0% |
| Prior 15m and 60m taker-buy flow | 11 | 11 | 50.0% |

### Historical exit amounts
- Stop trades: 23/60, 38.3% of sample. Mean logged ROI -0.813%, median -0.766%. Stop exit price was on average 0.304% ABOVE entry.
- Winning trades: 37/60, mean logged ROI 0.691%, median 0.653%.
- Approximate historical net8 * notional sum -0.8504 USDT; losses -3.8497, winners 2.9993. NOT broker reconciled actual realized USDT.
- Median gross best ROI observed inside FULLY ENCLOSED 1m candles during logged losing positions ~0.376%: some moves started favorably before the stop but cannot determine an executable giveback from this statistic.

## Reversal after stop
| After stop close | Return at least to old SHORT entry | Reach price for +1.25% gross ROI from old entry |
|---|---:|---:|
| 15 min | 6/23 | 0/23 |
| 30 min | 10/23 | 2/23 |
| 60 min | 14/23 | 4/23 |
| 120 min | 18/23 | 9/23 |
| 240 min | 21/23 | 15/23 |

- Among 21 that returned to old entry within 4h, median first retouch is 37 minutes after stop; 15/23 later touched the old hypothetical +1.25% ROI price, median first touch 82 min among those 15.
- Additional adverse HIGH above stop exit BEFORE return to old entry had median ~0.319% price; max ~1.345% price. Full 4h median maximum further upside ~0.838% price. Ignoring this exposure and simply widening stop can worsen realized losses drastically.
- Future moves may include a new separate SHORT entry, not necessarily a held original position. The old SHORT strategy had no 4h timeout equivalent.

## Suggested shadow gates, NOT production rules
1. At SHORT open require fresh completed 3m/5m trend and 15m/60m price/flow history. Do not reject purely because last 1m candle is bullish: all 60 historical trades share that property.
2. Run 15m+60m jointly rising as CAUTION/PENDING shadow tag, not hard reject: in sample it catches 10 losses but also rejects 10 winners.
3. Test loss-specific missing entry reversal features and actual Sniper/Entry Confirmation log snapshots. No evidence yet that simple 5m-up veto improves losses.
4. Replay early profit lock and lower SL against full 1m OHLC sequence with stop-first assumption, actual fees and no simultaneous duplicate trades; compare true net USDT.
5. Use Sept–Nov fresh out-of-sample after thresholds fixed; no live change now.

## Per-loss audit (timestamps UTC)
| Open UTC | Logged ROI | Previous 3m trend (9m) | Previous 5m trend (15m) | Prior 60m | Old entry retouch after stop (min) | Hypothetical +1.25% price after stop (min) |
|---|---:|---:|---:|---:|---:|---:|
| 2026-07-21 09:07 | -0.95% | -0.11% | -0.24% | +0.24% | 26 | — |
| 2026-07-21 11:12 | -1.11% | -0.07% | -0.07% | +0.43% | 20 | 82 |
| 2026-07-21 15:32 | -0.91% | -0.25% | -0.49% | -0.08% | 101 | — |
| 2026-07-21 18:35 | -0.92% | -0.41% | -0.04% | +0.38% | 62 | — |
| 2026-07-21 20:25 | -0.94% | -0.24% | -0.15% | -0.64% | 206 | — |
| 2026-07-21 21:00 | -0.90% | +0.09% | +0.02% | +0.15% | 37 | 174 |
| 2026-07-22 00:42 | -1.31% | +0.30% | +0.24% | +0.62% | 37 | 230 |
| 2026-07-22 02:10 | -0.96% | -0.28% | -0.19% | +0.06% | 166 | 179 |
| 2026-07-22 02:21 | -0.96% | +0.32% | +0.09% | +0.47% | 133 | 141 |
| 2026-08-31 23:10 | -0.61% | +0.08% | +0.22% | -0.69% | 17 | 38 |
| 2026-09-01 01:31 | -0.74% | -0.22% | -0.06% | -0.02% | 10 | — |
| 2026-09-01 02:06 | -0.61% | +0.06% | -0.02% | -0.59% | — | — |
| 2026-09-01 02:25 | -0.77% | +0.02% | +0.39% | +0.27% | 116 | — |
| 2026-09-01 03:02 | -0.76% | -0.12% | +0.47% | +0.71% | 6 | 80 |
| 2026-09-01 03:57 | -1.07% | -0.24% | -0.60% | +0.22% | 4 | 33 |
| 2026-09-01 11:20 | -0.62% | +0.06% | -0.43% | -0.04% | 68 | 98 |
| 2026-09-01 15:37 | -0.61% | +0.47% | +0.43% | -0.77% | 16 | 24 |
| 2026-09-01 20:51 | -0.63% | -0.02% | -0.44% | -0.66% | 8 | 237 |
| 2026-09-01 21:31 | -0.61% | +0.06% | +0.31% | -0.33% | 14 | 186 |
| 2026-09-01 23:16 | -0.61% | -0.04% | -0.13% | -0.10% | 52 | 79 |
| 2026-09-02 02:17 | -0.82% | +0.44% | +0.38% | +0.46% | — | — |
| 2026-09-02 02:59 | -0.62% | +0.37% | +0.66% | +2.31% | 52 | 69 |
| 2026-09-02 05:14 | -0.66% | -0.06% | -0.43% | +0.37% | 5 | 23 |
