"""Strict historical USD-M order/trade evidence. Read-only, no release authority.

Binance allOrders is not complete lifetime history; userTrades may include
fills for the same order at different times. Terminal order and fill matching
is evidence only and never authorizes reserve release without flat, no
inflight orders, generation-bound ownership and independent broker checks.
"""
from __future__ import annotations
from collections import defaultdict
from decimal import Decimal,InvalidOperation
from typing import Any
import hashlib

def num(value: Any):
    try:
        result=Decimal(str(value))
        return result if result.is_finite() else None
    except (TypeError,ValueError,InvalidOperation):
        return None

def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()[:14]

def reconcile_orders(symbol,orders,trades,algos):
    if not all(isinstance(z,list) for z in (orders,trades,algos)):
        return dict(state='UNKNOWN',reason='non_list_history',safe_to_auto_release=False)
    if any(not isinstance(z,dict) or z.get('symbol')!=symbol for v in (orders,trades,algos) for z in v):
        return dict(state='UNKNOWN',reason='bad_history_row_or_symbol',safe_to_auto_release=False)
    by_order=defaultdict(list)
    for t in trades:
        if t.get('orderId') is None or t.get('id') is None:
            return dict(state='UNKNOWN',reason='trade_missing_order_or_trade_id',safe_to_auto_release=False)
        qty=num(t.get('qty'));price=num(t.get('price'))
        if qty is None or qty<=0 or price is None or price<=0:
            return dict(state='UNKNOWN',reason='invalid_trade_qty_price',safe_to_auto_release=False)
        by_order[str(t['orderId'])].append(t)
    seen=set()
    records=[]
    linked_fills=0
    fees=defaultdict(Decimal)
    realized=Decimal('0')
    states=defaultdict(int)
    for order in orders:
        oid=order.get('orderId')
        cid=order.get('clientOrderId')
        if oid is None or cid is None or str(oid) in seen:
            return dict(state='UNKNOWN',reason='missing_or_duplicate_order_identity',safe_to_auto_release=False)
        seen.add(str(oid))
        status=str(order.get('status') or '').upper()
        orig=num(order.get('origQty'));executed=num(order.get('executedQty'))
        if orig is None or executed is None or orig<=0 or executed<0 or executed>orig:
            return dict(state='UNKNOWN',reason='inconsistent_order_quantity',safe_to_auto_release=False)
        fills=by_order[str(oid)]
        # distinct trade IDs, no double-counted fills.
        if len({str(t['id']) for t in fills})!=len(fills):
            return dict(state='UNKNOWN',reason='duplicate_trade_id',safe_to_auto_release=False)
        filled=sum((num(t.get('qty')) for t in fills),Decimal('0'))
        full_trade_match=(filled==executed)
        if filled>executed:
            return dict(state='UNKNOWN',reason='fill_qty_exceeds_order',safe_to_auto_release=False)
        # FILLED order without full trades cannot be independently confirmed.
        state=('CONFIRMED_FILLED' if status=='FILLED' and executed==orig and full_trade_match and fills
            else 'TERMINAL_NO_FILL' if status in ('CANCELED','EXPIRED','EXPIRED_IN_MATCH') and executed==0 and not fills
            else 'PARTIAL_OR_MISSING_FILLS' if executed>0 and not full_trade_match
            else 'OPEN_OR_UNRESOLVED')
        if status=='FILLED' and executed==orig and full_trade_match and fills:
            for fill in fills:
                f=num(fill.get('commission'));p=num(fill.get('realizedPnl'))
                asset=fill.get('commissionAsset')
                if f is None or f<0 or not asset or p is None:
                    return dict(state='UNKNOWN',reason='fill_fee_or_pnl_missing',safe_to_auto_release=False)
                fees[str(asset)]+=f;realized+=p;linked_fills+=1
        states[state]+=1
        records.append(dict(order_hash=digest(oid),client_order_hash=digest(cid),
            status=status,order_qty=str(orig),executed_qty=str(executed),
            matched_trade_qty=str(filled),fill_count=len(fills),state=state,
            side=order.get('side'),position_side=order.get('positionSide')))
    orphan_fills={k:len(v) for k,v in by_order.items() if k not in seen}
    algostates=defaultdict(int)
    algo_records=[]
    for algo in algos:
        algo_id=algo.get('algoId');client=algo.get('clientAlgoId')
        if algo_id is None or client is None or not algo.get('algoStatus'):
            return dict(state='UNKNOWN',reason='algo_missing_identity_status',safe_to_auto_release=False)
        algostates[str(algo['algoStatus']).upper()]+=1
        # An EXPIRED conditional order is NOT a proof that the position closed.
        algo_records.append(dict(algo_hash=digest(algo_id),client_algo_hash=digest(client),
            state=str(algo['algoStatus']).upper(),triggered_order_hash=(
                digest(algo['actualOrderId']) if algo.get('actualOrderId') else None)))
    clean_all=bool(orders) and not orphan_fills and states.get('PARTIAL_OR_MISSING_FILLS',0)==0 and states.get('OPEN_OR_UNRESOLVED',0)==0
    return dict(symbol=symbol,
        state='ORDER_FILL_MATCHED_SAMPLE' if clean_all else 'INCOMPLETE_OR_EMPTY_HISTORY',
        history_is_complete=False,
        safe_to_auto_release=False,
        risk_master_can_commit=False,
        reason='history_is_sample_not_durable_intent_journal',
        order_count=len(orders),trade_count=len(trades),algo_count=len(algos),
        linked_fill_count=linked_fills,orphan_trade_order_count=len(orphan_fills),
        match_states=dict(states),algo_status=dict(algostates),
        realized_pnl_usdt_before_fees=str(realized),
        commissions_by_asset={k:str(v) for k,v in fees.items()},
        orders=records,algos=algo_records)
