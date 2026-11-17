from orchestration.risk.position_stop_lifecycle_evidence_v1 import assess
r=dict(request_id='intent-1',reservation_id='generation-1',strategy_id='LONG1',symbol='NEARUSDT')
l=dict(intent_id='intent-1',reservation_id='generation-1',symbol='NEARUSDT',side='BUY',
       stage='PARTIAL',filled_qty='2',original_qty='4',client_algo_id='b1l1_owned')
pos=[dict(symbol='NEARUSDT',positionSide='LONG',positionAmt='2')]
stop=[dict(symbol='NEARUSDT',clientAlgoId='b1l1_owned',side='SELL',
           positionSide='LONG',orderType='STOP_MARKET',closePosition=True,
           workingType='MARK_PRICE',algoStatus='NEW',algoId=123)]
def v(rr=None,ll=None,p=None,orders=None,a=None):
 return assess(reservation=r if rr is None else rr,ledger=l if ll is None else ll,
               positions=pos if p is None else p,orders=[] if orders is None else orders,
               algos=stop if a is None else a)
assert v()['state']=='PROTECTED_POSITION_OBSERVED' and not v()['can_release']
assert v(a=[])['state']=='FREEZE_UNPROTECTED'
assert v(ll=dict(l,client_algo_id='bad'))['state']=='FREEZE_UNPROTECTED'
assert v(ll=dict(l,filled_qty='1'))['reason']=='broker_position_vs_fill_qty_mismatch'
assert v(p=[dict(pos[0],positionAmt='3')])['state']=='UNKNOWN'
assert v(p=[dict(pos[0],positionAmt='bad')])['state']=='UNKNOWN'
assert v(rr=dict(r,reservation_id='other'))['state']=='UNKNOWN'
assert v(p=[],a=[])['state']=='BLOCK_RELEASE'
assert v(p=[],a=[],orders=[dict(symbol='NEARUSDT',orderId=3)])['state']=='BLOCK_RELEASE'
assert v(p=[],a=[],ll=dict(l,stage='FLAT_RECONCILED'))['state']=='BLOCK_RELEASE'
flat=dict(l,stage='FLAT_RECONCILED',verified_exit_order_id='ord-2',verified_exit_qty='2')
assert v(p=[],a=[],ll=flat)['state']=='FLAT_HISTORY_CONSISTENT' and not v(p=[],a=[],ll=flat)['can_release']
assert v(p=[],a=[],ll=dict(flat,verified_exit_qty='1'))['state']=='BLOCK_RELEASE'
short_r=dict(r,strategy_id='SHORT1',symbol='INJUSDT')
short_l=dict(l,symbol='INJUSDT',side='SELL',client_algo_id='b1s1_owned')
short_pos=[dict(symbol='INJUSDT',positionSide='SHORT',positionAmt='-2')]
short_stop=[dict(stop[0],symbol='INJUSDT',clientAlgoId='b1s1_owned',
                 side='BUY',positionSide='SHORT')]
assert assess(reservation=short_r,ledger=short_l,positions=short_pos,orders=[],algos=short_stop)['state']=='PROTECTED_POSITION_OBSERVED'
assert assess(reservation=short_r,ledger=short_l,positions=short_pos,orders=[],algos=[])['state']=='FREEZE_UNPROTECTED'
print('PASS 13 LONG/SHORT partial fill/stop/flat/order-fencing checks; no release authority')
