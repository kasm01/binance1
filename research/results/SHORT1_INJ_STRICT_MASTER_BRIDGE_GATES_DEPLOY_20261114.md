# SHORT1 INJUSDT: strict pre-Master / Bridge guards deployed
14 Nov 2026 UTC. SHORT1 DRY_RUN=1 ARMED=0, no new orders, model promotion OFF. LONG1 unchanged.

## Findings
- Previously the Aggregator normalized absent whale, spread, ATR, liquidity and flow inputs to 0/none, erasing original field presence. Master could then treat missing inputs as neutral market states.
- Model semantics mismatch remains: V6 old target arm4roi/mae2pct vs V7 TP125/SL299 research goal. Research V7 OOS average net ROI remained negative, so no model is production-authorized.

## Code changes
- orchestration/context/short1_strict_market_contract.py: new source-completeness validator only for SHORT INJUSDT. Requires source ID, completed 3m bar timestamp, fresh source age, model provider/target, fixed source threshold, finite price/ATR/spread/liquidity/whale/flow data, explicit source field presence, verified neutral whale, source promotion flag and separately enabled local operator policy.
- Operator policy SHORT1_V7_PRODUCTION_APPROVED is default OFF and not set. A Redis event with model_promotion_authorized=True cannot by itself enable model promotion. No model approval or new live stream launched.
- orchestration/aggregator/aggregator.py captures input field presence before normalizing to neutral defaults; forwards source/model target/threshold fields. Explicitly forces model_promotion_authorized False on aggregated events.
- orchestration/executor/master_executor.py checks strict SHORT1 source contract before Market Context and score gate, avoiding neutral defaults. Other strategy/coin paths unchanged.
- orchestration/executor/master_executor.py includes relevant original context and provenance fields when publishing intent.
- orchestration/executor/intent_bridge.py rechecks strict SHORT1 input before passing to exec events. Non-INJ generic transport behavior retained.

## Tests
14 independent SHORT1 test groups PASS and 0 FAIL, including new 13-scenario validator/Master/Bridge test, historical provenance regression updated to reject unauthorized INJ, V6 and FSM tests, PENDING checks, native stop and stop reconciliation tests.
No order/broker writes in tests. All code compiled.

## Service state
LONG1 supervisor, LONG1 native stop, SHORT1 V5/V6 shadow, isolated SHORT1 shadow bus, and central risk dry-run service active. SHORT1 main executor OFF. SHORT1 DRY_RUN=1 ARMED=0 and unchanged TP=+1.25% ROI, hard stop=-8.97% ROI gross (2.99% adverse price at 3x).

## Limitations
- Source field presence is a missing-data diagnostic and cannot authenticate a maliciously crafted Redis event. A true production provenance chain still requires authenticated or signed canonical Master intent and executor-side verification.
- SHORT1 will remain blocked until real fresh market data sources and an approved, correctly targeted and tested model exist; do not turn on approval flag just to force trades.
- SHORT1 persistent native Algo reconciliation service and shared central production Risk Master remain unfinished.
- Remaining six bots unaffected.
