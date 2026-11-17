"""Research-only partial fill/position/native stop lifecycle safety evidence.

NOT a production authority. Requires positionRisk + openOrders + openAlgoOrders.
Recorded intent and reservation IDs must match; source evidence is never
authenticated by this module, so no decision can directly enable RELEASE.
"""
from __future__ import annotations
from decimal import Decimal,InvalidOperation
def q(x):
    try:
        v=Decimal(str(x))
        return v if v.is_finite() and v>=0 else None
    except (ValueError,TypeError,InvalidOperation):
        return None
def assess(*,reservation,ledger,positions,orders,algos):
    deny=dict(state='UNKNOWN',reason='unverified',
              can_release=False,can_open_new=False,can_commit=False)
    if not isinstance(reservation,dict) or not isinstance(ledger,dict):
        return dict(deny,reason='missing_reservation_or_ledger')
    for a,b in [('reservation_id','reservation_id'),
                ('request_id','intent_id'),('symbol','symbol')]:
        if not reservation.get(a) or str(reservation[a])!=str(ledger.get(b)):
            return dict(deny,reason='reservation_intent_identity_mismatch')
    if reservation.get('strategy_id') not in ('LONG1','SHORT1'):
        return dict(deny,reason='unsupported_strategy')
    expected_symbol='NEARUSDT' if reservation['strategy_id']=='LONG1' else 'INJUSDT'
    expected_side='BUY' if reservation['strategy_id']=='LONG1' else 'SELL'
    expected_stop_side='SELL' if expected_side=='BUY' else 'BUY'
    expected_position_side='LONG' if expected_side=='BUY' else 'SHORT'
    if reservation['symbol']!=expected_symbol or ledger.get('side')!=expected_side:
        return dict(deny,reason='strategy_symbol_direction_mismatch')
    if not isinstance(positions,list) or not isinstance(orders,list) or not isinstance(algos,list):
        return dict(deny,reason='missing_broker_source')
    for label,rows in [('positions',positions),('regular_orders',orders),('algo',algos)]:
        if any(not isinstance(x,dict) or x.get('symbol')!=expected_symbol for x in rows):
            return dict(deny,reason='invalid_'+label+'_row')
    legs=[]
    for x in positions:
        if x.get('positionSide') not in ('BOTH','LONG','SHORT'):
            return dict(deny,reason='invalid_position_side')
        try:
            val=q(abs(Decimal(str(x['positionAmt']))))
        except (ValueError,TypeError,InvalidOperation,KeyError):
            val=None
        if val is None:
            return dict(deny,reason='invalid_position_qty')
        if val>0:legs.append(x)
    if len(legs)>1:
        return dict(deny,reason='multiple_position_legs')
    if legs:
        pos=legs[0]
        if pos.get('positionSide') not in (expected_position_side,'BOTH'):
            return dict(deny,reason='opposite_position_side')
        if ledger.get('stage') not in ('PARTIAL','FILLED','COMMITTED','ACTIVE'):
            return dict(deny,reason='open_position_without_fill_ledger')
        filled=q(ledger.get('filled_qty'))
        if filled is None or filled<=0:
            return dict(deny,reason='no_confirmed_fill_qty')
        # Previously parsed and checked positive finite above.
        if val!=filled:
            return dict(deny,reason='broker_position_vs_fill_qty_mismatch')
        expected_client_stop=ledger.get('client_algo_id')
        valid=[a for a in algos if a.get('clientAlgoId')==expected_client_stop
               and a.get('side')==expected_stop_side
               and a.get('positionSide')==expected_position_side
               and (a.get('orderType') or a.get('type'))=='STOP_MARKET'
               and str(a.get('closePosition')).lower()=='true'
               and a.get('workingType')=='MARK_PRICE'
               and a.get('algoStatus') in ('NEW','WORKING','PENDING')]
        if len(valid)!=1 or not expected_client_stop:
            return dict(deny,state='FREEZE_UNPROTECTED',
                        reason='missing_or_unverified_native_stop')
        return dict(deny,state='PROTECTED_POSITION_OBSERVED',
                    reason='matched_position_and_owned_mark_stop',
                    confirmed_position_qty=str(filled))
    # Exchange shows flat; in-flight normal order or conditional Algo still
    # makes any release unsafe. Empty book is NOT a durable terminal receipt.
    if orders or algos:
        return dict(deny,state='BLOCK_RELEASE',reason='flat_with_open_orders_or_algos')
    if ledger.get('stage') not in ('FLAT_RECONCILED','CLOSED'):
        return dict(deny,state='BLOCK_RELEASE',reason='flat_but_exit_history_unreconciled')
    if not ledger.get('verified_exit_order_id') or ledger.get('verified_exit_qty')!=ledger.get('filled_qty'):
        return dict(deny,state='BLOCK_RELEASE',reason='exit_identity_or_qty_missing')
    return dict(deny,state='FLAT_HISTORY_CONSISTENT',
                reason='flat_broker_and_exit_history_match_but_not_authenticated')
