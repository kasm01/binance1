from core.strategy_symbol_policy import validate_assignment,validate_8_assignments
from core.trade_executor_short import TradeExecutorShort
assert validate_assignment('LONG1','NEARUSDT')[0]
assert validate_assignment('SHORT1','INJUSDT')[0]
assert validate_assignment('SHORT1','NEARUSDT')==(False,'strategy_symbol_mismatch')
assert not validate_assignment('SHORT2','SOLUSDT')[0]
p={**{f'LONG{i}':s for i,s in enumerate(('NEARUSDT','SOLUSDT','SUIUSDT','ADAUSDT'),1)},
   **{f'SHORT{i}':s for i,s in enumerate(('INJUSDT','DOGEUSDT','ARBUSDT','LINKUSDT'),1)}}
assert validate_8_assignments(p)[0]
p['SHORT4']='NEARUSDT';assert not validate_8_assignments(p)[0]
class Dummy:
    strategy_id='SHORT1'
    _symbol_open_locks={}
    dry_run=True
d=Dummy()
r=TradeExecutorShort.open_position_from_signal(d,'NEARUSDT','short','1m',{})
assert r['reason']=='strategy_symbol_mismatch',r
d.dry_run=False
r=TradeExecutorShort.open_position_from_signal(d,'INJUSDT','short','1m',{})
assert r['reason']=='central_risk_live_not_authorized',r
print('PASS: LONG1=NEAR, SHORT1=INJ, all 8 symbols unique, wrong coin blocked, accidental SHORT live blocked')
