from decimal import Decimal
from orchestration.risk.reduced_mode_budget_evidence_v1 import evaluate
e=dict(state='CONSISTENT_READONLY',can_authorize_new_reservation=False,
       equity_usdt='62.66814695',available_balance_usdt='62.66814695',
       central_snapshot_age_seconds=15,risk_mode='REDUCED',
       account_position_count=0,local_position_count=0,
       central_dry_pending_reservation_count=0)
v=evaluate(e,strategy='LONG1')
assert v['state']=='BUDGET_EVIDENCE_ONLY'
assert Decimal(v['illustrative_notional'])==Decimal('62.66814695')*Decimal('.23')
assert not v['order_authorized'] and not v['reservation_authorized']
assert Decimal(evaluate(dict(e,risk_mode='NORMAL'),strategy='LONG1')['illustrative_notional'])==Decimal('62.66814695')*Decimal('.46')
assert Decimal(evaluate(dict(e,risk_mode='RECOVERY'),strategy='SHORT1')['illustrative_notional'])==Decimal('62.66814695')*Decimal('.322')
assert evaluate(dict(e,central_snapshot_age_seconds=121),strategy='LONG1')['state']=='UNKNOWN_FREEZE'
assert evaluate(dict(e,risk_mode='WRONG'),strategy='LONG1')['state']=='UNKNOWN_FREEZE'
assert evaluate(dict(e,equity_usdt='NaN'),strategy='LONG1')['state']=='UNKNOWN_FREEZE'
assert evaluate(dict(e,central_dry_pending_reservation_count=1),strategy='LONG1')['state']=='UNKNOWN_FREEZE'
assert evaluate(dict(e,account_position_count=1),strategy='LONG1')['state']=='UNKNOWN_FREEZE'
assert evaluate(dict(e,can_authorize_new_reservation=True),strategy='LONG1')['state']=='UNKNOWN_FREEZE'
assert evaluate(dict(e,state='UNKNOWN'),strategy='LONG1')['state']=='UNKNOWN_FREEZE'
assert evaluate(e,strategy='LONG2')['state']=='UNKNOWN_FREEZE'
assert evaluate(dict(e,equity_usdt='40',available_balance_usdt='40'),strategy='SHORT1')['state']=='REJECT_BUDGET'
half=evaluate(e,strategy='LONG1',market_multiplier='.5')
assert half['state']=='REJECT_BUDGET'
assert Decimal(half['illustrative_notional'])==Decimal(v['illustrative_notional'])*Decimal('.5')
print('PASS 13 REDUCED/NORMAL/RECOVERY sizing and stale/unverified/busy failure cases')
