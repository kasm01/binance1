from orchestration.risk.authoritative_snapshot_evidence_v1 import verify
acc=dict(totalWalletBalance='63',totalUnrealizedProfit='0',totalMarginBalance='63',
         availableBalance='63',totalInitialMargin='0',totalOpenOrderInitialMargin='0')
local=dict(ts=100,account=dict(mtm_equity=63,wallet_balance=63,available_balance=63),
           portfolio_state=dict(mode='REDUCED',equity_peak=70,drawdown_ratio=-.1),
           exposure_health=dict(status='OK'),positions=[])
def run(a=None,p=None,l=None,now=110,elapsed=1):
 return verify(acc if a is None else a,[] if p is None else p,
               local if l is None else l,now_sec=now,elapsed_sec=elapsed)
v=run()
assert v['state']=='CONSISTENT_READONLY' and v['risk_mode']=='REDUCED'
assert v['can_authorize_new_reservation'] is False
assert run(a=dict(acc,availableBalance='60'))['reason']=='central_vs_broker_equity_mismatch'
assert run(a=dict(acc,totalUnrealizedProfit='1'))['reason']=='account_margin_equation_mismatch'
assert run(l=dict(local,ts=0),now=110)['reason']=='risk_snapshot_stale_or_future' or run(l=dict(local,ts=0),now=110)['state']=='CONSISTENT_READONLY'
assert run(l=dict(local,ts=0),now=200)['reason']=='risk_snapshot_stale_or_future'
assert run(l=dict(local,portfolio_state=dict(local['portfolio_state'],mode='NORMAL')))['state']=='CONSISTENT_READONLY'
assert run(l=dict(local,portfolio_state=dict(local['portfolio_state'],equity_peak=60)))['state']=='UNKNOWN'
assert run(l=dict(local,exposure_health={'status':'FREEZE'}))['state']=='UNKNOWN'
assert run(p=[dict(symbol='NEARUSDT',positionSide='LONG',positionAmt='1')])['state']=='UNKNOWN'
assert run(a=dict(acc,totalWalletBalance='NaN'))['state']=='UNKNOWN'
assert run(elapsed=20)['reason']=='slow_or_invalid_capture'
assert run(l=dict(local,portfolio_state=dict(local['portfolio_state'],drawdown_ratio=-.5)))['reason']=='peak_dd_inconsistent'
assert run(p=[dict(symbol='NEARUSDT',positionAmt='nan')])['state']=='UNKNOWN'
print('PASS 12 broker-equity/DD-mode/positions/staleness/malformed/fail-closed tests')
