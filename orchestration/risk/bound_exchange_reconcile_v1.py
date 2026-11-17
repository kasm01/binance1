"""Read-only exact-intent Binance evidence binding for research SHORT1/LONG1.

Requires the immutable Redis reservation+intent identity and exact client ID,
then evaluates signed exchange order/trades and current position/orders/Algo.
All source lookups must succeed; UNKNOWN => freeze. It never grants COMMIT,
RELEASE, or ORDER authority; a broker GET is not an authenticated writer.
"""
from __future__ import annotations
from decimal import Decimal,InvalidOperation
import time
from orchestration.risk.unknown_order_recovery_evidence_v1 import classify_order_recovery
from orchestration.risk.position_stop_lifecycle_evidence_v1 import assess

def _q(v):
    try:
        z=Decimal(str(v))
        return z if z.is_finite() and z>=0 else None
    except (ValueError,TypeError,InvalidOperation):return None

def bind(*,reservation,ledger,read_order,read_fills,read_positions,
         read_open_orders,read_open_algo,clock=time.monotonic,max_capture_sec=15):
    fail=dict(state='UNKNOWN_FREEZE',reason='unverified',can_submit=False,
              can_commit=False,can_release=False,
              source_binding_verified=False)
    if not isinstance(reservation,dict) or not isinstance(ledger,dict):
        return dict(fail,reason='missing_immutable_pair')
    if (not reservation.get('request_id') or
        str(reservation.get('request_id'))!=str(ledger.get('intent_id')) or
        not reservation.get('reservation_id') or
        str(reservation.get('reservation_id'))!=str(ledger.get('reservation_id')) or
        str(reservation.get('strategy_id'))!=str(ledger.get('owner')) or
        str(reservation.get('symbol'))!=str(ledger.get('symbol')) or
        not ledger.get('client_order_id')):
        return dict(fail,reason='immutable_reservation_intent_mismatch')
    sym=str(reservation['symbol'])
    side=str(ledger.get('side'))
    if (reservation.get('strategy_id')=='LONG1' and (sym!='NEARUSDT' or side!='BUY')) or (
        reservation.get('strategy_id')=='SHORT1' and (sym!='INJUSDT' or side!='SELL')):
        return dict(fail,reason='strategy_symbol_side_mismatch')
    if reservation.get('strategy_id') not in ('LONG1','SHORT1'):
        return dict(fail,reason='unsupported_strategy')
    if ledger.get('stage') not in ('SUBMITTING','UNKNOWN','ACKED','PARTIAL','FILLED','COMMITTED','ACTIVE','FLAT_RECONCILED'):
        return dict(fail,reason='ledger_state_not_reconcilable')
    started=clock()
    try:
        order=read_order(sym,ledger['client_order_id'])
        # Historical trade filtering must bind same order ID, never just all
        # symbol trades; the source wrapper must query that exact ID.
        def trades(symbol,orderid):return read_fills(symbol,orderid)
        restored=dict(ledger,stage=ledger['stage'] if ledger['stage'] in
                      ('SUBMITTING','UNKNOWN','ACKED','PARTIAL','FILLED') else 'FILLED')
        historical=classify_order_recovery(record=restored,
                    read_order=lambda s,c:order,read_fills=trades)
        positions=read_positions(sym)
        regular=read_open_orders(sym)
        algos=read_open_algo(sym)
    except Exception as exc:
        return dict(fail,reason='broker_source_failed:'+type(exc).__name__)
    elapsed=clock()-started
    if elapsed<0 or elapsed>max_capture_sec:
        return dict(fail,reason='broker_capture_stale_or_slow',elapsed_sec=round(elapsed,3))
    if not isinstance(positions,list) or not isinstance(regular,list) or not isinstance(algos,list):
        return dict(fail,reason='broker_open_sources_invalid')
    if any(not isinstance(r,dict) or r.get('symbol')!=sym for seq in (positions,regular,algos) for r in seq):
        return dict(fail,reason='broker_open_source_symbol_mismatch')
    if historical.get('state') not in ('FULLY_FILLED_EVIDENCE','OPEN_OR_PARTIAL',
                                       'TERMINAL_ZERO_FILL_EVIDENCE','PARTIAL_TERMINAL_UNRESOLVED'):
        return dict(fail,reason='exact_order_unresolved:'+str(historical.get('reason')),
                    historical_state=historical.get('state'))
    if historical.get('state')!='FULLY_FILLED_EVIDENCE':
        return dict(fail,state='HOLD_RECONCILIATION',
                    reason='exact_order_not_fully_filled',
                    historical_state=historical.get('state'))
    qty=_q(historical.get('verified_qty'))
    if qty is None or qty<=0:
        return dict(fail,reason='missing_verified_qty')
    existing=_q(ledger.get('filled_qty'))
    if ledger.get('stage') in ('SUBMITTING','UNKNOWN'):
        # An unknown POST can be FILLED while the journal remains zero.
        # The mismatch requires a separate authenticated journal repair,
        # not a false claim that the broker's fill evidence is missing.
        return dict(fail,state='RECOVERED_FILL_PENDING_JOURNAL',
                    reason='broker_filled_but_ledger_uncommitted',
                    broker_verified_qty=str(qty))
    if existing is None or existing!=qty:
        return dict(fail,reason='ledger_fill_vs_broker_mismatch')
    if regular:
        return dict(fail,state='HOLD_RECONCILIATION',
                    reason='other_normal_orders_still_open')
    life=assess(reservation=reservation,ledger=ledger,positions=positions,
                orders=regular,algos=algos)
    # Ensure an observed stop belongs to the immutable ledger, not any
    # same-symbol protective Algo stop.
    if life['state']=='PROTECTED_POSITION_OBSERVED':
        return dict(fail,state='PROTECTED_EVIDENCE_ONLY',
                    reason='entry_fill_position_and_stop_match',
                    broker_verified_qty=str(qty),lifecycle_state=life['state'],
                    source_binding_verified=True)
    if life['state']=='FREEZE_UNPROTECTED':
        return dict(fail,state='FREEZE_UNPROTECTED',
                    reason=life['reason'],lifecycle_state=life['state'])
    if life['state']=='FLAT_HISTORY_CONSISTENT':
        return dict(fail,state='FLAT_EVIDENCE_ONLY',
                    reason='flat_history_matches_untrusted_exit_identity',
                    lifecycle_state=life['state'],
                    source_binding_verified=True)
    return dict(fail,state='HOLD_RECONCILIATION',
                reason=life.get('reason','unknown_position_stop_state'),
                lifecycle_state=life.get('state'))
