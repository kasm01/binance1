# SHORT1 / INJUSDT — Live V5/V6 shadow Signal Bus deployment
Date 13 Nov 2026 UTC. Purpose: prove causal short V6 model -> PENDING -> independently fresh model -> shadow READY on live model feed, without routing a single order.

## Hard safety boundaries
- SHORT1 .env: DRY_RUN=1 ARMED=0 SYMBOL=INJUSDT STRATEGY_ID=SHORT1. The sidecar refuses DRY_RUN=0, ARMED=1, wrong symbol/strategy. LONG1 NEARUSDT is live and was not edited/restarted here.
- Input streams (read only): telemetry:short1:inj:model_shadow_v5; telemetry:short1:inj:model_shadow_v6.
- Output streams (only): shadow:short1:inj:signal_bus:v1 (all shadow 3m decisions), shadow:short1:inj:pending_fsm:v1 (pending, fresh-model and direct shadow READY), plus shadow:short1:inj:pending_fsm_state:v1 and shadow:short1:inj:signal_bus_last_close_ms:v1.
- No writes to signals_stream_short1, candidates_stream_short1, top5_stream_short1, trade_intents_stream_short1, exec_events_stream_short1, any LONG stream or Binance endpoint. Production stream lengths/last IDs checked before and after first real shadow write and remained unchanged.
- Source enforces model records routing=OFF, correct SHORT1/INJUSDT/3m and completed bar timestamps. No market data download or extra collector. Models already use their own existing paid/free data access.
- Sidecar is a shadow *observer*. Its SHADOW_READY is NOT an authorized ORDER, not a native executor Entry Confirmation / Master score gate / Risk Master approval / execution return.

## Implementation
File: orchestration/shadow/short1_inj_shadow_signal_bus_v1.py
Systemd: ~/.config/systemd/user/binance1-short1-inj-shadow-signal-bus.service
Unit enabled at default.target and active, user linger enabled. New process ~20 MB resident service memory by systemd MemoryCurrent. No new Binance calls; small Redis poll every 20s.
- Uses **V6** as primary score threshold to study probability of SHORT 4% ROI within 60m conditional on adverse path; **V5** is independent entry-quality advisory, NOT multiplied/averaged into a fake probability.
- Align both V5 and V6 only by identical completed 3m candle close timestamp. No mixed bars or future candles. Rejects stale events >8m or records arriving from future. Uses a unique V6 source_event_id and derived shadow intent_id.
- A V6 threshold pass but missing proxy confirmation becomes SHADOW_PENDING for up to 15 minutes; future independent V6 with new source model event, new shadow intent ID, newer completed 3m close and V6 hard threshold + new confirmation produces SHADOW_READY_FRESH_MODEL. Same intent/model event cannot promote itself.
- A V6 pass + source model confirmation with no prior pending produces SHADOW_READY_DIRECT. All outcomes are shadow only.
- V5 threshold, V6 threshold, 1m progress and confirmation state are recorded separately. V5 does not veto a V6 observation; no new production filter introduced.
- Exactly once checkpoint and pending state with Redis optimistic WATCH/MULTI: paired shadow stream writes and checkpoint stored transactionally; restart/duplicate poll does not duplicate latest record. 2000 entry stream and 1000 transition stream bounded for RAM/cost.
- Service restarts automatically only on process failure. If VM off, no new shadow data is collected here. On restart only fresh events considered; historical replay is a separate read-only mode.

## Tests and preliminary evidence
- Pure tests: research/tests/test_short1_inj_shadow_signal_bus_v1.py (distinct new intent, same source event block, mismatched 3m bar block, stale/future block, direct vs pending, armed/live flags forbidden).
- Historical last 24h already collected native shadow telemetry, 480 aligned completed 3m INJ bars:
  415 SHADOW_REJECT,
  47 SHADOW_PENDING *events* (not necessarily unique episodes),
  9 SHADOW_WAIT,
  2 SHADOW_READY_DIRECT,
  7 SHADOW_READY_FRESH_MODEL (7 unique new intent IDs).
  These are modeled shadow decisions; actual order fills, additional Market Context, Sniper, native 1m Entry Confirmation, Execution Risk and Final Safety not evaluated.
- Before deploy historical source last exec_intents approx 13d old; V5/V6 current. The new isolated shadow bus does not make executable short pipeline current.
- One real shadow write used V6 score 0.57145 below its 0.80829 threshold, produced SHADOW_REJECT. Production stream lengths/last IDs unchanged. Subsequent --once test returned WAIT_NEW_3M_BAR and no duplicate shadow/production writes.
- Systemd active at first check, MainPID 931366, NRestarts=0; MemoryCurrent 20508672 bytes (~19.6 MiB).

## Operational commands (read-only)
cd ~/binance1
systemctl --user status binance1-short1-inj-shadow-signal-bus.service --no-pager
journalctl --user -u binance1-short1-inj-shadow-signal-bus.service --no-pager -n 35
./venv/bin/python orchestration/shadow/short1_inj_shadow_signal_bus_v1.py --replay --count 480
./venv/bin/python research/tests/test_short1_inj_shadow_signal_bus_v1.py
./venv/bin/python -m py_compile orchestration/shadow/short1_inj_shadow_signal_bus_v1.py

## Next steps
- Observe new genuinely live completed 3m bars and real PENDING→fresh model SHADOW_READY transitions; verify first live event, restart idempotency, no cross-stream writes.
- Build an isolated shadow market-context/Execution Risk/Final Safety *evaluation*, never call actual executor's order path. Match unique signals to subsequent closes and net-fee USDT using deterministic fills, slippage and sensitivity bands.
- Native SHORT1 PENDING code remains staged but SHORT main/executor is OFF; no end-to-end production SHORT1 signal loop. Do not call this a live trading bot or live strategy validation.
- Only after this, central live Risk Master for 8 unique coins, SHORT1 native stop auto-reconciliation/service, per-strategy Redis/PID/log isolation, audited dry-run lifecycle, then separately authorized live promotion.

## Shadow-ready fee sensitivity: observational, NOT a trade replay
- Additional bounded two-page Binance public 1m history download, 1560 contiguous one-minute bars (26h window), saved at research/short/data/INJUSDT_1m_shadow_fee_markout_20261113.csv. No new collector and no background data poll.
- Source read-only: last 480 shared 3m V5/V6 model records, 9 UNIQUE shadow-ready events (7 after new independent V6 model intent and 2 direct).
- Entry assumed opening price of first 1m candle beginning AFTER V6 model observation timestamp. Fixed markout exits at 15/30/60 minutes; exact hypothetical short price return (entry-exit)/entry.
- Illustrative costs: taker roundtrip commission 0.08% notional and total roundtrip price slippage 0.06% notional, 3x leverage, no funding (rates are assumptions NOT verified user-specific Binance fees).
- Approximate ROI on posted margin: 3*(price_short_return_pct - .08% - .06%). Actual order sizing, margin, exchange fill price, dynamic stop and profit lock not simulated.
- 9 unique events:
  15m mean underlying SHORT +0.2097%, illustrative 3x net ROI +0.2092%; 5/9 positive.
  30m mean underlying SHORT +0.4679%, illustrative 3x net ROI +0.9837%; 7/9 positive.
  60m mean underlying SHORT -0.0881%, illustrative 3x net ROI -0.6843%; 2/9 positive.
- This strongly argues that holding time / ROI locking needs separate replay, not that 30-minute exit is profitable. Only 9 selected observations, no out-of-sample or multiple-comparison adjustment; cannot establish edge.
- Results: research/results/SHORT1_INJ_LIVE_SHADOW_READY_FEE_SCENARIO_20261113.csv
- Script: research/tests/short1_inj_live_shadow_ready_fee_proxy_v1.py
