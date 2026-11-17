"""Bounded historical USD-M LONG closed-cycle accounting (READ ONLY).

Exact trade-id commission-income reconciliation in BNB/USDT, and only
uniquely-attributable USDT funding inside an observed open-position window.
No BNB->USDT price inference and no risk-mode/position-release authority.
All cycles are HISTORICAL CANDIDATES, never durable pre-submit provenance.
"""
from __future__ import annotations
from collections import defaultdict
from decimal import Decimal,InvalidOperation

def dec(v):
 try:
  x=Decimal(str(v))
  return x if x.is_finite() else None
 except (ValueError,TypeError,InvalidOperation):return None

def analyze(symbol,orders,trades,commission_income,funding_income):
 fail=dict(state='UNKNOWN_FREEZE',reason='invalid_or_incomplete_history',
   cycles=[],net_pnl_usdt_certified=False,
   can_authorize_recovery_win=False,can_release=False)
 if symbol!='NEARUSDT':
  return dict(fail,reason='scope_nearusdt_only')
 if not all(isinstance(v,list) for v in (orders,trades,commission_income,funding_income)):
  return dict(fail,reason='missing_history_feed')
 if not orders or not trades or len(orders)>=100 or len(trades)>=100 or len(commission_income)>=100 or len(funding_income)>=100:
  return dict(fail,reason='empty_or_potentially_truncated_history')
 if len(orders)%2:return dict(fail,reason='unpaired_orders')
 if any(not isinstance(x,dict) or x.get('symbol')!=symbol for seq in
        (orders,trades,commission_income,funding_income) for x in seq):
  return dict(fail,reason='wrong_symbol_or_malformed_row')
 by_order=defaultdict(list)
 t_by_id={}
 for trade in trades:
  tid=str(trade.get('id',''))
  oid=str(trade.get('orderId',''))
  if tid in ('','None') or oid in ('','None') or tid in t_by_id:
   return dict(fail,reason='duplicate_or_missing_trade_identity')
  qty=dec(trade.get('qty'));price=dec(trade.get('price'))
  fee=dec(trade.get('commission'));pnl=dec(trade.get('realizedPnl'))
  ms=trade.get('time')
  if None in (qty,price,fee,pnl) or qty<=0 or price<=0 or fee<0 or not isinstance(ms,int) or ms<=0:
   return dict(fail,reason='invalid_trade_values')
  if trade.get('commissionAsset') not in ('USDT','BNB'):
   return dict(fail,reason='unsupported_fee_asset')
  by_order[oid].append(trade);t_by_id[tid]=trade
 # Match commission income by EXACT trade ID, amount, asset, and source type.
 seen_income=set()
 for row in commission_income:
  tid=str(row.get('tradeId',''))
  if tid not in t_by_id or tid in seen_income:
   return dict(fail,reason='orphan_or_duplicate_fee_income')
  seen_income.add(tid)
  trade=t_by_id[tid]
  income=dec(row.get('income'))
  if row.get('incomeType')!='COMMISSION' or income is None or row.get('asset')!=trade.get('commissionAsset') or income!=-dec(trade['commission']):
   return dict(fail,reason='fee_income_mismatch')
  if abs(int(row.get('time',0))-int(trade['time']))>60000:
   return dict(fail,reason='fee_income_time_mismatch')
 if len(seen_income)!=len(trades):
  return dict(fail,reason='missing_fee_income_by_trade')
 sorted_orders=sorted(orders,key=lambda o:int(o.get('time',0)))
 if len({str(x.get('orderId')) for x in sorted_orders})!=len(sorted_orders):
  return dict(fail,reason='duplicate_order_id')
 pairs=[];covered=set();prior_exit=-1
 for i in range(0,len(sorted_orders),2):
  e,x=sorted_orders[i:i+2]
  if not (e.get('side')=='BUY' and e.get('reduceOnly') is False and
          x.get('side')=='SELL' and x.get('reduceOnly') is True and
          e.get('positionSide')=='LONG' and x.get('positionSide')=='LONG'):
   return dict(fail,reason='not_strict_buy_then_reduce_sell')
  if e.get('status')!='FILLED' or x.get('status')!='FILLED':
   return dict(fail,reason='nonfilled_entry_or_exit')
  eq=dec(e.get('executedQty'));xq=dec(x.get('executedQty'))
  if eq is None or xq is None or eq<=0 or eq!=xq or eq!=dec(e.get('origQty')) or xq!=dec(x.get('origQty')):
   return dict(fail,reason='unmatched_full_position_qty')
  ef=by_order[str(e['orderId'])];xf=by_order[str(x['orderId'])]
  if not ef or not xf:return dict(fail,reason='missing_order_fills')
  if sum((dec(t['qty']) for t in ef),Decimal('0'))!=eq or sum((dec(t['qty']) for t in xf),Decimal('0'))!=xq:
   return dict(fail,reason='fill_qty_mismatch')
  et=min(int(t['time']) for t in ef);xt=max(int(t['time']) for t in xf)
  if et<=prior_exit or xt<=et:return dict(fail,reason='overlapping_or_inverted_position_cycles')
  prior_exit=xt
  pnl=sum((dec(t['realizedPnl']) for t in ef+xf),Decimal('0'))
  commission_bnb=sum((dec(t['commission']) for t in ef+xf if t['commissionAsset']=='BNB'),Decimal('0'))
  commission_usdt=sum((dec(t['commission']) for t in ef+xf if t['commissionAsset']=='USDT'),Decimal('0'))
  covered.update(str(t['id']) for t in ef+xf)
  pairs.append(dict(cycle_no=i//2+1,entry_ms=et,exit_ms=xt,
    qty=str(eq),gross_realized_usdt=str(pnl),
    commission_bnb=str(commission_bnb),commission_usdt=str(commission_usdt),
    funding_usdt='0',gross_after_known_usdt_charges=None,
    net_usdt_unknown_if_bnb_fee=commission_bnb>0))
 if covered!=set(t_by_id):return dict(fail,reason='orphan_trades_outside_cycles')
 used_funding=set()
 for row in funding_income:
  if row.get('incomeType')!='FUNDING_FEE' or row.get('asset')!='USDT' or row.get('tranId') is None:
   return dict(fail,reason='invalid_funding_row')
  fid=str(row['tranId'])
  value=dec(row.get('income'))
  t=row.get('time')
  if fid in used_funding or value is None or not isinstance(t,int):
   return dict(fail,reason='invalid_or_duplicate_funding')
  used_funding.add(fid)
  matches=[p for p in pairs if p['entry_ms']<t<p['exit_ms']]
  if len(matches)!=1:
   return dict(fail,reason='funding_not_uniquely_attributable')
  p=matches[0]
  p['funding_usdt']=str(dec(p['funding_usdt'])+value)
 gross=Decimal(0);bnb=Decimal(0);usdtfee=Decimal(0);fund=Decimal(0)
 for p in pairs:
  g=dec(p['gross_realized_usdt']);fee=dec(p['commission_usdt']);funding=dec(p['funding_usdt'])
  after=g-fee+funding
  p['gross_after_known_usdt_charges']=str(after)
  p['definitely_loss_given_nonnegative_bnb_fee']=after<=0
  gross+=g;usdtfee+=fee;fund+=funding;bnb+=dec(p['commission_bnb'])
 return dict(state='HISTORICAL_CYCLES_ACCOUNTED_FEES_UNPRICED' if bnb else 'HISTORICAL_CYCLES_CANDIDATE_ONLY',
   reason='no_durable_source_generation_or_trusted_bnb_usdt_valuation',
   cycles=pairs,cycle_count=len(pairs),matched_fills=len(trades),
   exact_commission_income_matches=len(seen_income),
   funding_rows_uniquely_assigned=len(used_funding),
   gross_realized_usdt=str(gross),
   funding_usdt=str(fund),
   commissions_bnb=str(bnb),commissions_usdt=str(usdtfee),
   gross_after_known_usdt_charges=str(gross+fund-usdtfee),
   net_pnl_usdt_certified=False,can_authorize_recovery_win=False,can_release=False)
