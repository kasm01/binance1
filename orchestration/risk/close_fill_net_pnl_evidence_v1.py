"""Strict *candidate* close/fill net-PnL evidence, RESEARCH ONLY.

A closing order's realizedPnl is not a whole round-trip's net return.
Entry commissions, funding and any BNB commissions must also be accounted
for. No eligible recovery win without unambiguous USDT-converted costs.
Caller can forge rows: no source attestation or live promotion.
"""
from __future__ import annotations
from decimal import Decimal,InvalidOperation

def d(x):
 try:
  v=Decimal(str(x))
  return v if v.is_finite() else None
 except (TypeError,ValueError,InvalidOperation):return None
def evaluate(*,intent,entry_order,exit_order,entry_fills,exit_fills,
             position_rows,open_orders,open_algos,
             funding_usdt=None,external_fee_conversion=None):
 fail=dict(state='UNKNOWN_FREEZE',reason='insufficient_or_untrusted_evidence',
           eligible_recovery_win=False,can_update_risk_mode=False,
           can_release=False,net_pnl_usdt=None)
 if not isinstance(intent,dict) or not isinstance(entry_order,dict) or not isinstance(exit_order,dict):
  return dict(fail,reason='missing_intent_or_order')
 sym=intent.get('symbol');side=intent.get('side')
 if sym not in ('NEARUSDT','INJUSDT') or side not in ('BUY','SELL') or (
     intent.get('owner')=='LONG1' and (sym!='NEARUSDT' or side!='BUY')) or (
     intent.get('owner')=='SHORT1' and (sym!='INJUSDT' or side!='SELL')) or (
     intent.get('owner') not in ('LONG1','SHORT1')):
  return dict(fail,reason='strategy_symbol_side_invalid')
 if not intent.get('reservation_id') or not intent.get('intent_id') or not intent.get('client_order_id'):
  return dict(fail,reason='intent_provenance_missing')
 if intent.get('opened_risk_mode') not in ('NORMAL','RECOVERY','REDUCED'):
  return dict(fail,reason='opened_mode_unverified')
 entry_id=entry_order.get('orderId')
 exit_id=exit_order.get('orderId')
 if entry_id is None or exit_id is None or str(entry_id)==str(exit_id):
  return dict(fail,reason='missing_or_reused_order_identity')
 if (entry_order.get('symbol')!=sym or exit_order.get('symbol')!=sym or
     entry_order.get('clientOrderId')!=intent['client_order_id'] or
     entry_order.get('side')!=side or exit_order.get('side')==side or
     exit_order.get('side') not in ('BUY','SELL')):
  return dict(fail,reason='entry_exit_identity_mismatch')
 if (entry_order.get('status')!='FILLED' or exit_order.get('status')!='FILLED'):
  return dict(fail,reason='nonterminal_or_partial_order')
 qty=d(entry_order.get('executedQty'))
 ex_qty=d(exit_order.get('executedQty'))
 orig=d(entry_order.get('origQty'))
 exit_orig=d(exit_order.get('origQty'))
 if None in (qty,ex_qty,orig,exit_orig) or qty<=0 or qty!=ex_qty or qty!=orig or qty!=exit_orig:
  return dict(fail,reason='not_fully_closed_exact_qty')
 if not all(isinstance(z,list) for z in (entry_fills,exit_fills,position_rows,open_orders,open_algos)):
  return dict(fail,reason='invalid_broker_lists')
 if open_orders or open_algos:return dict(fail,reason='open_order_or_algo_exists')
 if not position_rows:return dict(fail,reason='missing_position_observation')
 for row in position_rows:
  if not isinstance(row,dict) or row.get('symbol')!=sym:
   return dict(fail,reason='wrong_symbol_position')
  a=d(row.get('positionAmt'))
  if a is None or a!=0:return dict(fail,reason='position_not_confirmed_flat')
 ids=set();total_commission=Decimal(0);pnl=Decimal(0)
 for orderid,fills in ((entry_id,entry_fills),(exit_id,exit_fills)):
  if not fills or len(fills)>=100:
   return dict(fail,reason='missing_or_potentially_truncated_fills')
  filled=Decimal(0)
  for row in fills:
   if not isinstance(row,dict) or row.get('symbol')!=sym or str(row.get('orderId'))!=str(orderid):
    return dict(fail,reason='fill_order_identity_mismatch')
   trade_id=row.get('id')
   if trade_id is None or str(trade_id) in ids:
    return dict(fail,reason='duplicate_or_missing_trade_id')
   ids.add(str(trade_id))
   fqty=d(row.get('qty'));fee=d(row.get('commission'));rpnl=d(row.get('realizedPnl'))
   if None in (fqty,fee,rpnl) or fqty<=0 or fee<0:
    return dict(fail,reason='missing_or_invalid_fill_costs')
   asset=row.get('commissionAsset')
   if asset=='USDT':
    converted=fee
   elif asset:
    # A supplied conversion is NOT itself proof of conversion integrity.
    # Until trusted per-fill BNB/USDT price data exists, no risk upgrade.
    return dict(fail,state='UNPRICED_FEES_HOLD',
                reason='non_usdt_commissions_need_authenticated_conversion')
   else:
    return dict(fail,reason='missing_fee_asset')
   total_commission+=converted
   pnl+=rpnl
   filled+=fqty
  if filled!=qty:return dict(fail,reason='entry_or_exit_fill_qty_mismatch')
 funding=d(funding_usdt)
 if funding is None:
  return dict(fail,state='FUNDING_UNKNOWN_HOLD',
              reason='funding_for_position_lifetime_unverified')
 # Must include funding receipts for whole lifetime and ALL entry/exits.
 net=pnl-total_commission+funding
 # Caller-provided proofs can only produce candidate status, never a
 # signed or authorized risk mode promotion.
 return dict(fail,state='NET_PNL_CANDIDATE_ONLY',
    reason='math_matched_but_broker_source_and_funding_provenance_untrusted',
    net_pnl_usdt=str(net),gross_realized_usdt=str(pnl),
    usdt_commissions=str(total_commission),
    funding_usdt=str(funding),
    candidate_recovery_win=(intent['opened_risk_mode']=='RECOVERY' and net>0),
    verified_entry_exit_fill_count=len(ids))
