from orchestration.risk.close_fill_net_pnl_evidence_v1 import evaluate
i=dict(intent_id='i1',reservation_id='g1',client_order_id='entry1',
       owner='LONG1',symbol='NEARUSDT',side='BUY',opened_risk_mode='RECOVERY')
entry=dict(orderId=10,clientOrderId='entry1',symbol='NEARUSDT',side='BUY',
           status='FILLED',origQty='2',executedQty='2')
ex=dict(orderId=11,clientOrderId='exit1',symbol='NEARUSDT',side='SELL',
        status='FILLED',origQty='2',executedQty='2')
def fill(oid,n,pnl='0',asset='USDT',comm='0.01',qty='2'):
 return dict(orderId=oid,id=n,symbol='NEARUSDT',qty=qty,realizedPnl=pnl,
             commissionAsset=asset,commission=comm)
ef=[fill(10,1)]
xf=[fill(11,2,pnl='0.05')]
pos=[dict(symbol='NEARUSDT',positionAmt='0',positionSide='LONG')]
def run(**kw):
 data=dict(intent=i,entry_order=entry,exit_order=ex,entry_fills=ef,
   exit_fills=xf,position_rows=pos,open_orders=[],open_algos=[],funding_usdt='0')
 data.update(kw)
 return evaluate(**data)
good=run()
assert good['state']=='NET_PNL_CANDIDATE_ONLY' and good['net_pnl_usdt']=='0.03'
assert good['candidate_recovery_win'] is True
assert good['eligible_recovery_win'] is False and good['can_update_risk_mode'] is False
assert run(exit_fills=[fill(11,2,pnl='0.01')])['candidate_recovery_win'] is False
assert run(entry_fills=[fill(10,1,asset='BNB',comm='0.00002')])['state']=='UNPRICED_FEES_HOLD'
assert run(funding_usdt=None)['state']=='FUNDING_UNKNOWN_HOLD'
assert run(open_orders=[dict(symbol='NEARUSDT',orderId=15)])['state']=='UNKNOWN_FREEZE'
assert run(open_algos=[dict(symbol='NEARUSDT',algoId=12)])['state']=='UNKNOWN_FREEZE'
assert run(position_rows=[dict(pos[0],positionAmt='1')])['state']=='UNKNOWN_FREEZE'
assert run(exit_order=dict(ex,executedQty='1'))['state']=='UNKNOWN_FREEZE'
assert run(entry_fills=[fill(10,1),fill(10,1)])['state']=='UNKNOWN_FREEZE'
assert run(intent=dict(i,reservation_id=''))['state']=='UNKNOWN_FREEZE'
assert run(intent=dict(i,opened_risk_mode='NORMAL'))['candidate_recovery_win'] is False
assert run(exit_fills=[fill(11,2,pnl='0.05',asset='BNB')])['state']=='UNPRICED_FEES_HOLD'
print('PASS 13 net-pnl/entry-exit/funding/BNB/owner/position scenarios; promotion always forbidden')
