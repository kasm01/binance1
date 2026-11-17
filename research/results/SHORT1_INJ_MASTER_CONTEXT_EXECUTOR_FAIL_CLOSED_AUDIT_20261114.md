# SHORT1 INJUSDT – Master Context / Score Gate fail-closed audit
14 Nov 2026 UTC. Research only. No production config or live order.

## Evidence
- 1900 paired real V5/V6 model events replayed with original source emission timestamps: 3 SHADOW_READY_DIRECT and 8 SHADOW_READY_FRESH_MODEL.
- Of 11 shadow READY, 10 match the frozen V7 research score and the original V6 source score.
- Calling ACTUAL SHORT executor 1m Entry Confirmation on archived partial 1m snapshot gives 8 READY, 2 PENDING.
- 8 READY were matched back to their original V6 source event JSON and tested through strict source contract plus ACTUAL SHORT executor Market Context and Model Score Gate methods, WITHOUT spoofing upstream pass flags.
- Strict Market Context BLOCK_CONTEXT 8/8; missing original fields: price, whale_dir, whale_score, spread_pct, atr_pct, liq_score.
- Executor Market Context locally READY 8/8 (it is a carrier, not an authoritative nonorchestrated veto). Actual upstream Model Score Gate REJECT 8/8: upstream_model_score_gate_required.
- Therefore 0 Execution Risk, 0 Final Safety, 0 executable orders authorized.

## Security finding
- ACTUAL MasterExecutor market context code has defaults 0/none for missing candidate fields. This may mask genuinely absent whale/spread/liquidity information as a neutral market state. A strict fail-closed source precheck is required before market context produces production authority.
- Actual executor market and score stage methods TRUST upstream orchestration pass flags. An adversarial regression test supplying execution_source=orchestrated, market_context_passed=True and model_score_gate_passed=True makes both isolated downstream stage methods READY. This is NOT proof of a live exploit or unsafe live order, but shows upstream provenance and ownership must be enforced at the trusted interface; arbitrary research Redis events cannot become canonical ready intents.
- Correct canonical approval must bind source event ID, completed candle ID, fresh model target and threshold semantics, authenticated owner, unique intent, timestamp-consistent Market Context, and central portfolio risk approval.
- The SHORT1 main executor is OFF; DRY_RUN=1, ARMED=0. Production Risk Master profile ENFORCE_DRY_RUN and production_authorized=False. No live orders issued.

## Existing sources
- V5/V6 live shadow and independent V30 lifecycle shadow; public orderbook V44 shadow, centrally collected OI context.
- V44 events currently do not provide all canonical Master whale direction/score, ATR, spread, liquidity fields as direct fields. Timestamp-consistent provenance mapping needed. Older whale shadow source is stale.
- Reuse existing streams and cached public market data; no duplicate collectors or extra paid streams until evidence requires them.

## Tests
- research/tests/test_short1_master_fail_closed_contract_v1.py: 8 checks and positive identification of forged-upstream-flags trust boundary.
- research/tests/test_short1_v6_master_contract_v1.py: PASS 7.
- research/tests/test_short1_inj_shadow_signal_bus_v1.py: PASS 9.
- research/tests/test_short_pending_new_model_link_v1.py: PASS.
- No actual Master approval was faked or inserted, no Redis writes, no Binance orders.

## Next acceptance gates
1. Verify market data field lineage/freshness as-of exact source event for each required Master field; fail closed on missing/stale values.
2. Bind trusted Master authorizations to verified upstream provenance, not pass flags alone.
3. Complete live-shadow-only canonical intent audit and PENDING fresh-source transition, without touching production streams.
4. Implement SHORT1 persistent native Algo stop reconciler and central Risk Master strategy coordination, then test full dry-run lifecycle.
5. No V7 production promotion; frozen OOS was negative overall. Keep LONG1+SHORT1 priority.
