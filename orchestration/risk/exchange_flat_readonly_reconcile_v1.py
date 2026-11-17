"""Strict read-only Binance USD-M exposure reconciliation for LONG1/SHORT1.

Never writes Binance/Redis or authorizes production. Broker flat check uses
positionRisk, openOrders and openAlgoOrders; all 3 sources are required.
The report is evidence only, NOT an authenticated release/commit token.
"""
from __future__ import annotations
from datetime import datetime,timezone
import math,time
from typing import Any,Callable

SYMBOLS=('NEARUSDT','INJUSDT')

def _positive_or_zero(v):
    try:
        x=float(v)
        return x if math.isfinite(x) and x>=0 else None
    except (ValueError,TypeError):
        return None

def _strict_array(v,kind,symbol):
    if not isinstance(v,list):
        raise ValueError(f'{kind}_invalid_shape')
    for x in v:
        if not isinstance(x,dict):
            raise ValueError(f'{kind}_non_dict_row')
        if x.get('symbol')!=symbol:
            raise ValueError(f'{kind}_unexpected_symbol')
    return v

def evaluate_symbol(symbol:str,position_rows:Any,order_rows:Any,algo_rows:Any):
    if symbol not in SYMBOLS:
        return dict(state='UNKNOWN',reason='symbol_not_allowed',symbol=symbol,release_candidate=False)
    try:
        positions=_strict_array(position_rows,'positions',symbol)
        regular=_strict_array(order_rows,'openOrders',symbol)
        algos=_strict_array(algo_rows,'openAlgoOrders',symbol)
        live_qty=[]
        exposure=[]
        for row in positions:
            if 'positionAmt' not in row or 'positionSide' not in row:
                raise ValueError('position_row_missing_qty_or_side')
            side=str(row['positionSide']).upper()
            if side not in ('BOTH','LONG','SHORT'):
                raise ValueError('position_side_invalid')
            try:amount=float(row['positionAmt'])
            except (TypeError,ValueError):raise ValueError('position_amount_invalid')
            if not math.isfinite(amount):raise ValueError('position_amount_nonfinite')
            if abs(amount)>1e-10:
                live_qty.append(dict(position_side=side,abs_qty=abs(amount)))
                exposure.append(dict(position_side=side,position_amt=amount))
        for order in regular:
            if not order.get('orderId') and not order.get('clientOrderId'):
                raise ValueError('regular_order_missing_identity')
        for algo in algos:
            if not algo.get('algoId') and not algo.get('clientAlgoId'):
                raise ValueError('algo_order_missing_identity')
        any_open=bool(live_qty or regular or algos)
        return dict(symbol=symbol,
            state='BLOCK_RELEASE' if any_open else 'FLAT_CANDIDATE',
            reason=('position_or_order_open' if any_open else 'all_three_sources_empty_of_exposure'),
            position_nonzero_count=len(live_qty),
            position_legs=exposure,
            regular_order_count=len(regular),
            algo_order_count=len(algos),
            release_candidate=not any_open,
            requires_independent_recheck=True,
            risk_master_approved=False)
    except (ValueError,TypeError) as exc:
        return dict(symbol=symbol,state='UNKNOWN',
            reason=str(exc),release_candidate=False,
            requires_independent_recheck=True,risk_master_approved=False)

def read_only_snapshot(
    position_risk:Callable[[str],Any],
    normal_orders:Callable[[str],Any],
    algo_orders:Callable[[str],Any],
    *,
    symbols=SYMBOLS,
    clock:Callable[[],float]=time.time,
):
    start=clock()
    records={}
    for symbol in symbols:
        observations=[]
        errors=[]
        for name,call in [('position',position_risk),('regular',normal_orders),('algo',algo_orders)]:
            try:
                observations.append(call(symbol))
            except Exception as exc:
                observations.append(None)
                errors.append(name+':'+type(exc).__name__)
        if errors:
            records[symbol]=dict(symbol=symbol,state='UNKNOWN',
                reason='source_read_failed',failed_sources=errors,release_candidate=False,
                requires_independent_recheck=True,risk_master_approved=False)
        else:
            records[symbol]=evaluate_symbol(symbol,*observations)
    end=clock()
    can_release_candidate=all(row.get('state')=='FLAT_CANDIDATE' for row in records.values())
    # The three calls for each symbol are not an atomic Binance snapshot.
    return dict(schema='binance_usdm_flat_readonly.v1',
       checked_at_utc=datetime.fromtimestamp(end,timezone.utc).isoformat(),
       elapsed_ms=round((end-start)*1000,2),
       symbols=records,
       overall_state=('FLAT_CANDIDATE' if can_release_candidate else 'BLOCK_OR_UNKNOWN'),
       safe_to_auto_release=False,
       authorized_to_open=False,
       caveat='Only a non-atomic read-only observation. Requires independent recheck and ownership, broker order/fill reconciliation before releasing central reservation.')
