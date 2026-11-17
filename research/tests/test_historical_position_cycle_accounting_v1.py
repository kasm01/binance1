"""Minimal synthetic 2 complete cycles, exact fee income and funding join."""
from copy import deepcopy
from orchestration.risk.historical_position_cycle_accounting_v1 import analyze
s='NEARUSDT'
def order(n,side,t,qty='2',reduce=False):
 return dict(symbol=s,orderId=n,clientOrderId='cid-'+str(n),time=t,
             side=side,positionSide='LONG',reduceOnly=reduce,
             status='FILLED',origQty=qty,executedQty=qty)
orders=[order(1,'BUY',1000),order(2,'SELL',2000,reduce=True),
        order(3,'BUY',3000),order(4,'SELL',4000,reduce=True)]
def trade(order_id,time,pnl='0',fee='.001',asset='BNB'):
 return dict(symbol=s,orderId=order_id,id=order_id,time=time,qty='2',
             price='4',realizedPnl=pnl,commission=fee,commissionAsset=asset)
trades=[trade(1,1100),trade(2,1900,pnl='.10'),trade(3,3100),trade(4,3900,pnl='-.20')]
def fee(t):
 return dict(symbol=s,tradeId=t['id'],time=t['time'],incomeType='COMMISSION',
             asset=t['commissionAsset'],income=str(-float(t['commission'])))
fees=[fee(t) for t in trades]
fund=[dict(symbol=s,incomeType='FUNDING_FEE',asset='USDT',
           tranId=991,time=3500,income='-.01')]
def run(oo=orders,tt=trades,ff=fees,fu=fund):return analyze(s,oo,tt,ff,fu)
v=run()
assert v['state']=='HISTORICAL_CYCLES_ACCOUNTED_FEES_UNPRICED'
assert v['cycle_count']==2 and v['matched_fills']==4
assert v['gross_realized_usdt']=='-0.10'
assert v['funding_usdt']=='-0.01'
assert v['commissions_bnb']=='0.004'
assert v['cycles'][1]['definitely_loss_given_nonnegative_bnb_fee']
assert not v['net_pnl_usdt_certified'] and not v['can_authorize_recovery_win']
assert run(oo=orders[:3])['state']=='UNKNOWN_FREEZE'
assert run(oo=orders[::-1])['state']=='HISTORICAL_CYCLES_ACCOUNTED_FEES_UNPRICED'
assert run(oo=[dict(orders[0],side='SELL')]+orders[1:])['reason']=='not_strict_buy_then_reduce_sell'
assert run(oo=[dict(orders[0],positionSide='SHORT')]+orders[1:])['reason']=='not_strict_buy_then_reduce_sell'
assert run(oo=[dict(orders[0],status='PARTIALLY_FILLED')]+orders[1:])['state']=='UNKNOWN_FREEZE'
assert run(oo=[dict(orders[0],executedQty='1')]+orders[1:])['state']=='UNKNOWN_FREEZE'
assert run(tt=trades[:3])['state']=='UNKNOWN_FREEZE'
assert run(tt=trades[:3]+[dict(trades[3],id=1)])['reason']=='duplicate_or_missing_trade_identity'
assert run(ff=fees[:3])['reason']=='missing_fee_income_by_trade'
assert run(ff=fees+[fees[0]])['reason']=='orphan_or_duplicate_fee_income'
assert run(ff=[dict(fees[0],income='-.002')]+fees[1:])['reason']=='fee_income_mismatch'
assert run(ff=[dict(fees[0],asset='USDT')]+fees[1:])['reason']=='fee_income_mismatch'
assert run(fu=[dict(fund[0],time=2500)])['reason']=='funding_not_uniquely_attributable'
assert run(fu=fund*2)['reason']=='invalid_or_duplicate_funding'
assert run(fu=[dict(fund[0],asset='BNB')])['reason']=='invalid_funding_row'
assert run(oo=orders+[orders[1],orders[3]])['reason']=='duplicate_order_id'
print('PASS 18 closed-cycle, commission-income, funding, missing data and safety checks')
