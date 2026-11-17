from orchestration.risk.unknown_order_recovery_evidence_v1 import classify_order_recovery as ev
base=dict(intent_id='intent',reservation_id='res',
          client_order_id='cid',symbol='NEARUSDT',side='BUY',stage='UNKNOWN',
          original_qty='3')
def order(status='FILLED',qty='3',exec='3'):
 return dict(clientOrderId='cid',symbol='NEARUSDT',side='BUY',
      orderId=200,origQty=qty,executedQty=exec,status=status)
def trades(s,i):
 return [dict(symbol='NEARUSDT',orderId=200,id=1000,qty='1'),
         dict(symbol='NEARUSDT',orderId=200,id=1001,qty='2')]
def run(obj=None,read=None,fill=trades):
 return ev(record=base if obj is None else obj,
    read_order=(lambda s,c:order()) if read is None else read,read_fills=fill)
good=run()
assert good['state']=='FULLY_FILLED_EVIDENCE' and good['verified_qty']=='3'
assert not good['can_commit'] and not good['can_resubmit'] and not good['can_release']
assert run(read=lambda s,c:None)['state']=='UNKNOWN'
def network(s,c):raise TimeoutError('simulated')
assert run(read=network)['state']=='UNKNOWN'
assert run(read=lambda s,c:dict(order(),clientOrderId='other'))['state']=='UNKNOWN'
assert run(read=lambda s,c:dict(order(),origQty='4'))['state']=='UNKNOWN'
assert run(read=lambda s,c:dict(order(),status='NEW',executedQty='0'))['state']=='OPEN_OR_PARTIAL'
assert run(read=lambda s,c:dict(order(),status='PARTIALLY_FILLED',executedQty='1'))['state']=='OPEN_OR_PARTIAL'
assert run(read=lambda s,c:dict(order(),status='CANCELED',executedQty='1'))['state']=='PARTIAL_TERMINAL_UNRESOLVED'
assert run(read=lambda s,c:dict(order(),status='CANCELED',executedQty='0'))['state']=='TERMINAL_ZERO_FILL_EVIDENCE'
assert run(fill=lambda s,i:trades(s,i)[:1])['state']=='UNKNOWN'
assert run(fill=lambda s,i:trades(s,i)[:1]*3)['state']=='UNKNOWN'
assert run(obj=dict(base,stage='PREPARED'))['state']=='UNKNOWN'
print('PASS 12 unknown-order recovery/filled/partial/timeout/mismatch/no-resubmit checks')
