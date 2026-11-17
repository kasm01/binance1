"""Real Master/Bridge entry boundaries on SHORT1-INJ, no Redis or Binance writes."""
import math,time
from unittest.mock import patch
from datetime import datetime,timezone
from orchestration.context.short1_strict_market_contract import check,FIELDS,PROVIDER,TARGET
from orchestration.aggregator.aggregator import Aggregator
from orchestration.executor.master_executor import MasterExecutor
from orchestration.executor.intent_bridge import IntentBridge
def candidate():
 now=int(time.time()*1000)
 close=(now//180000)*180000-1
 return dict(
   candidate_id='test-cand',intent_id='test-intent',
   symbol='INJUSDT',side='short',interval='3m',
   source_signal_event_id='test-source',source_completed_close_ms=close,
   ts_utc=datetime.now(timezone.utc).isoformat(),
   model_provider=PROVIDER,model_target=TARGET,
   model_promotion_authorized=True,model_score=.75,
   model_probability_threshold=.7123981411511489,
   source_field_presence=list(FIELDS),
   price=8.12,spread_pct=.0002,atr_pct=.01,liq_score=.55,
   whale_dir='short',whale_score=.5,money_flow_15m=-.15,
   money_flow_60m=-.1,price_return_15m=-.002)
a=candidate()
assert check(a)==(False,'operator_model_promotion_disabled')
with patch.dict('os.environ',{'SHORT1_V7_PRODUCTION_APPROVED':'1'}):
 assert check(a)[0],check(a)
def denied(x,reason):
 with patch.dict('os.environ',{'SHORT1_V7_PRODUCTION_APPROVED':'1'}):
  good,why=check(x)
 assert not good,(reason,good,why)
 return why
assert denied(dict(a,model_promotion_authorized=False),'unpromoted')=='model_not_promoted'
assert denied(dict(a,model_target='short_arm4roi_within60m_mae2pct'),'oldmodel')=='model_target_not_approved'
assert denied(dict(a,source_field_presence=[]),'default values') .startswith('source_market_fields_missing')
assert denied(dict(a,source_signal_event_id=''),'empty intent source')=='source_id_or_completed_candle_invalid'
assert denied(dict(a,source_completed_close_ms=a['source_completed_close_ms']-180000*4),'stale 3m')=='source_not_fresh'
assert denied(dict(a,price=float('nan')),'nonfinite price')=='nonfinite_market_field:price'
assert denied(dict(a,whale_dir='none'),'unknown neutrality')=='unverified_neutral_whale'
with patch.dict('os.environ',{'SHORT1_V7_PRODUCTION_APPROVED':'1'}):
 assert check(dict(a,whale_dir='none',whale_observation_status='confirmed_neutral'))[0]
assert denied(dict(a,source_field_presence=[k for k in FIELDS if k!='whale_score']),'no whale source').startswith('source_market_fields_missing')
# Actual Aggregator normalizer captures key ABSENCE prior to normalizing.
agg=Aggregator.__new__(Aggregator)
original=dict(event_id='evt',symbol='INJUSDT',side_candidate='short',
   interval='3m',price=8.12,whale_score=.0)
normalized=agg._normalize_event('123-0',original)
assert 'whale_dir' not in normalized['_short1_source_field_presence']
# Actual Master function denies incomplete candidate BEFORE invoking its market evaluator.
master=MasterExecutor.__new__(MasterExecutor)
assert master._make_intent(dict(a,model_promotion_authorized=False)) is None
assert master._make_intent(dict(a,source_field_presence=[])) is None
# Actual bridge refuses incomplete/unpromoted source before any Redis xadd.
b=IntentBridge.__new__(IntentBridge)
b.publish_allowed=True;b._norm_side=lambda item:'short'
assert b._publish_exec('source-stream',dict(a,model_promotion_authorized=False)) is None
assert b._publish_exec('source-stream',dict(a,source_field_presence=[])) is None
print('PASS 13 SHORT1 strict source-field, provenance and Master/Bridge fail-closed checks, no external writes')
