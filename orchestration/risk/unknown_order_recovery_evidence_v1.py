"""Read-only reconciliation of a previously persisted UNKNOWN submit.

Caller MUST supply stored client_order_id and reservation/intent binding.
A 404/error/empty list is UNKNOWN, NOT authority to submit again. This is
evidence classification, not trusted Master/commit/auto-release authority.
"""
from __future__ import annotations
from decimal import Decimal,InvalidOperation
from typing import Any

def _qty(raw):
    try:
        v=Decimal(str(raw))
        return v if v.is_finite() and v>=0 else None
    except (ValueError,TypeError,InvalidOperation):
        return None

def classify_order_recovery(*,record:dict,read_order,read_fills):
    base=dict(state='UNKNOWN',can_resubmit=False,can_commit=False,
              can_release=False,broker_verified=False,
              reason='no_definitive_exchange_proof')
    for field in ('intent_id','reservation_id','client_order_id','symbol','side'):
        if not record.get(field):
            return dict(base,reason='ledger_identity_missing:'+field)
    if record.get('stage') not in ('SUBMITTING','UNKNOWN','ACKED','PARTIAL','FILLED'):
        return dict(base,reason='ledger_state_not_recoverable')
    try:
        # Expected to be GET /fapi/v1/order?symbol=...&origClientOrderId=...
        order=read_order(str(record['symbol']),str(record['client_order_id']))
    except Exception as exc:
        return dict(base,reason='order_lookup_failed:'+type(exc).__name__)
    if not isinstance(order,dict):
        return dict(base,reason='order_lookup_invalid_payload')
    if str(order.get('symbol'))!=str(record['symbol']) or str(order.get('clientOrderId'))!=str(record['client_order_id']):
        return dict(base,reason='order_identity_mismatch')
    orderid=order.get('orderId')
    if orderid is None or str(orderid)=='' or str(order.get('side'))!=str(record['side']):
        return dict(base,reason='order_id_or_side_invalid')
    amount=_qty(order.get('origQty'));done=_qty(order.get('executedQty'))
    original=_qty(record.get('original_qty'))
    if None in (amount,done,original) or amount<=0 or amount!=original or done>amount:
        return dict(base,reason='order_size_contract_mismatch')
    status=str(order.get('status') or '').upper()
    if status in ('NEW','PARTIALLY_FILLED','PENDING_NEW','PENDING_CANCEL'):
        return dict(base,state='OPEN_OR_PARTIAL',reason='order_is_not_terminal',
                    broker_verified=True,order_id_present=True)
    if status=='FILLED':
        if done!=amount:
            return dict(base,reason='filled_status_qty_mismatch')
        try:
            fills=read_fills(str(record['symbol']),str(orderid))
        except Exception as exc:
            return dict(base,reason='trade_lookup_failed:'+type(exc).__name__)
        if not isinstance(fills,list) or not fills:
            return dict(base,reason='filled_without_trade_evidence')
        total=Decimal('0');seen=set()
        for fill in fills:
            if not isinstance(fill,dict) or str(fill.get('orderId'))!=str(orderid) or str(fill.get('symbol'))!=str(record['symbol']):
                return dict(base,reason='trade_order_identity_mismatch')
            fid=fill.get('id');qty=_qty(fill.get('qty'))
            if fid is None or str(fid) in seen or qty is None or qty<=0:
                return dict(base,reason='duplicate_or_bad_fill')
            seen.add(str(fid));total+=qty
        if total!=done:
            return dict(base,reason='incomplete_fill_history')
        return dict(base,state='FULLY_FILLED_EVIDENCE',reason='order_and_trades_match',
                    broker_verified=True,order_id_present=True,
                    verified_fill_count=len(fills),verified_qty=str(total))
    if status in ('CANCELED','EXPIRED','EXPIRED_IN_MATCH','REJECTED'):
        if done!=0:
            return dict(base,state='PARTIAL_TERMINAL_UNRESOLVED',reason='terminal_with_partial_fill',
                        broker_verified=True,order_id_present=True)
        # Terminal never means the overall position/stop/order state is flat.
        return dict(base,state='TERMINAL_ZERO_FILL_EVIDENCE',
                    reason='terminal_order_zero_executed_qty',broker_verified=True,
                    order_id_present=True)
    return dict(base,reason='unknown_exchange_order_status')
