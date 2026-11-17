from orchestration.risk.binance_exact_readonly_adapter_v1 import ReadOnlyBinanceIntentAdapter as Adapter
calls=[]
def fake_get(url,args):
 calls.append((url,args))
 if url.endswith('/order'):return dict(symbol=args['symbol'],clientOrderId=args['origClientOrderId'])
 if url.endswith('/userTrades'):return []
 if url.endswith('/algoOrder'):return dict(symbol='INJUSDT',clientAlgoId=args['clientAlgoId'])
 return []
x=Adapter(get=fake_get,positions=lambda sym:[])
assert x.get_order('INJUSDT','b1s1_123')['clientOrderId']=='b1s1_123'
assert x.get_fills('INJUSDT','123')==[]
assert x.get_positions('INJUSDT')==[]
assert x.get_open_orders('INJUSDT')==[]
assert x.get_open_algos('INJUSDT')==[]
assert x.get_exact_algo('INJUSDT','b1s1_123')['symbol']=='INJUSDT'
assert all(z[0].startswith('/fapi/v1/') for z in calls)
assert any(c[0]=='/fapi/v1/userTrades' and c[1]['orderId']=='123' for c in calls)
for op in (lambda:x.get_order('SOLUSDT','x'),lambda:x.get_order('INJUSDT',''),
           lambda:x.get_exact_algo('INJUSDT','%invalid')):
 try:op()
 except ValueError:pass
 else:raise AssertionError('must_reject')
truncated=Adapter(get=lambda path,args: ([dict()] * 100) if path.endswith('userTrades') else [])
try:truncated.get_fills('INJUSDT','123')
except ValueError:pass
else:raise AssertionError('must_fail_truncation')
print('PASS 12 signed GET only, exact order identity, no truncated fills, strict symbols')
