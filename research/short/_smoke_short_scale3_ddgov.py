from core.trade_executor_short import TradeExecutorShort

class C:
    def futures_account_balance(self): return [{'asset':'USDT','balance':'1000'}]
    def futures_account(self): return {'availableBalance':'1000'}

x=TradeExecutorShort(client=C(),dry_run=True,max_leverage=3)
x._get_short_equity_usdt_sync=lambda:1000.0

n1=x._resolve_short_order_notional(symbol='INJUSDT')
assert abs(n1-1000.0)<1e-9, n1
r1=x._finalize_short_open(symbol='INJUSDT',interval='1m',qty=n1/10.0,order_price=10.0,leverage=3,now_ts=1.0,order={},entry_meta={'signal_score':.72})
assert r1['entry_layers']==1
assert x._resolve_short_open_risk_state(symbol='INJUSDT',now_ts=1000,signal_score=.72) is None

n2=x._resolve_short_order_notional(symbol='INJUSDT')
assert abs(n2-500.0)<1e-9, n2
r2=x._finalize_short_open(symbol='INJUSDT',interval='1m',qty=n2/11.0,order_price=11.0,leverage=3,now_ts=1000.0,order={},entry_meta={'signal_score':.73})
assert r2['entry_layers']==2 and r2['scale_in']

n3=x._resolve_short_order_notional(symbol='INJUSDT')
assert abs(n3-500.0)<1e-6, n3
r3=x._finalize_short_open(symbol='INJUSDT',interval='1m',qty=n3/12.0,order_price=12.0,leverage=3,now_ts=2000.0,order={},entry_meta={'signal_score':.74})
pos=x._get_position('INJUSDT')
assert r3['entry_layers']==3 and len(pos['layers'])==3
assert abs(pos['notional']-2000.0)<1e-6, pos['notional']
block=x._resolve_short_open_risk_state(symbol='INJUSDT',now_ts=3000,signal_score=.90)
assert block and block['reason']=='max_entry_layers', block

# Drawdown governor hysteresis
x._positions.clear(); x._short_equity_peak=1000.0; x._short_risk_mode='NORMAL'
x._get_short_equity_usdt_sync=lambda:920.0
g=x._resolve_short_drawdown_mode(920.0)
assert g['mode']=='REDUCED',g
nr=x._resolve_short_order_notional(symbol='INJUSDT')
assert abs(nr-460.0)<1e-6,nr
x._get_short_equity_usdt_sync=lambda:960.0
g2=x._resolve_short_drawdown_mode(960.0)
assert g2['mode']=='NORMAL',g2

print('SCALE3_DDGOV_SMOKE=PASS')
print('normal_layers',1000,500,500,'max_notional',2000)
print('reduced_layer1',nr,'risk_modes',g['mode'],'->',g2['mode'])
print('avg_entry',pos['entry_price'],'layers',len(pos['layers']))

# Lifecycle profile checks
x2=TradeExecutorShort(client=C(),dry_run=True,max_leverage=3)
p={'symbol':'INJUSDT','side':'short','entry_price':100.0,'qty':1.0,'leverage':3,'best_roi_pct':0.0}
soft=x2._evaluate_short_dynamic_stop_state(symbol='INJUSDT',position=p,price=103.0,roi=-0.09)
assert soft and soft['state']=='HOLD' and soft['reason']=='dynamic_stop_pending',soft
p['dynamic_stop_pending_ts']=p.get('dynamic_stop_pending_ts',0)-31.0
confirmed=x2._evaluate_short_dynamic_stop_state(symbol='INJUSDT',position=p,price=103.0,roi=-0.09)
assert confirmed and confirmed['state']=='CLOSE' and confirmed['reason']=='dynamic_stop_confirmed',confirmed
p2={'symbol':'INJUSDT','side':'short','entry_price':100.0,'qty':1.0,'leverage':3,'best_roi_pct':0.0}
cat=x2._evaluate_short_dynamic_stop_state(symbol='INJUSDT',position=p2,price=103.6,roi=-0.108)
assert cat and cat['state']=='CLOSE' and cat['reason']=='dynamic_stop_catastrophic',cat
tp=x2._resolve_short_fixed_roi_tp_state(old_best_roi=0.039,best_roi=0.040)
assert tp and abs(tp['trigger_roi']-0.0376)<1e-12,tp
print('LIFECYCLE_SMOKE=PASS',soft['reason'],confirmed['reason'],cat['reason'],tp['trigger_roi'])
