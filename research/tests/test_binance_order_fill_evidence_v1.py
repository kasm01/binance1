"""Read-only order ledger correctness and no-release assertions."""
from orchestration.risk.binance_order_fill_evidence_v1 import reconcile_orders
def order(status='FILLED',qty='2',executed='2'):
 return dict(orderId=1,clientOrderId='cid1',symbol='NEARUSDT',
             origQty=qty,executedQty=executed,status=status,side='BUY',positionSide='LONG')
def trade(qty='2',id=2):
 return dict(orderId=1,id=id,symbol='NEARUSDT',qty=qty,price='2.0',
             commission='0.00002',commissionAsset='BNB',realizedPnl='0.2')
def algo(status='EXPIRED'):
 return dict(symbol='NEARUSDT',algoId=3,clientAlgoId='b1l1_1',algoStatus=status)
good=reconcile_orders('NEARUSDT',[order()],[trade()],[algo()])
assert good['state']=='ORDER_FILL_MATCHED_SAMPLE'
assert good['match_states']['CONFIRMED_FILLED']==1 and good['linked_fill_count']==1
assert good['commissions_by_asset']['BNB']=='0.00002'
assert good['algo_status']['EXPIRED']==1 and not good['safe_to_auto_release']
assert reconcile_orders('NEARUSDT',[order()],[],[])['state']=='INCOMPLETE_OR_EMPTY_HISTORY'
assert reconcile_orders('NEARUSDT',[order()],[trade('1')],[])['match_states']['PARTIAL_OR_MISSING_FILLS']==1
assert reconcile_orders('NEARUSDT',[order('CANCELED','2','0')],[],[])['match_states']['TERMINAL_NO_FILL']==1
assert reconcile_orders('NEARUSDT',[order()],[trade('3')],[])['state']=='UNKNOWN'
assert reconcile_orders('NEARUSDT',[order()],[trade(),trade()],[])['state']=='UNKNOWN'
assert reconcile_orders('NEARUSDT',[order()],[dict(trade(),symbol='INJUSDT')],[])['state']=='UNKNOWN'
assert reconcile_orders('NEARUSDT',[order()],{},[])['state']=='UNKNOWN'
assert reconcile_orders('NEARUSDT',[],[],[])['state']=='INCOMPLETE_OR_EMPTY_HISTORY'
assert not reconcile_orders('NEARUSDT',[],[],[])['safe_to_auto_release']
print('PASS 12 fill/order/algo no-release edge checks')
