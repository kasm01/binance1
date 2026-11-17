"""Actual existing broker reconciliation + immutable Stage26-like source mapping.

No Redis, Binance, order, network, or production writes.
"""
from __future__ import annotations
import copy,json
from research.staging.stage28_reservation_exact_broker_bridge_v1 import assess_stage28

EPOCH=7
FENCE=42
NOW=1000500

def fixtures(strategy='SHORT1'):
    long=(strategy=='LONG1')
    sym='NEARUSDT' if long else 'INJUSDT'
    side='BUY' if long else 'SELL'
    pos_side='LONG' if long else 'SHORT'
    qty='2' if long else '-2'
    stop_side='SELL' if long else 'BUY'
    rid=f"epoch-{EPOCH}-fence-{FENCE}"
    reservation=dict(request=f"intent-{strategy}",owner=f"owner-{strategy}",
       clientid=f"order-{strategy}",strategy=strategy,
       symbol=sym,side='long' if long else 'short',epoch=EPOCH,fence=FENCE,
       digest='f'*64,approved=46,expires_at=NOW+10000)
    ledger=dict(intent_id=reservation['request'],reservation_id=rid,
       reservation_owner=reservation['owner'],epoch=EPOCH,fence=FENCE,
       owner=strategy,symbol=sym,side=side,client_order_id=reservation['clientid'],
       stage='FILLED',original_qty='2',filled_qty='2',
       client_algo_id=f"stop-{strategy}")
    order=dict(symbol=sym,clientOrderId=reservation['clientid'],
       orderId=77,side=side,status='FILLED',origQty='2',executedQty='2')
    fills=[dict(symbol=sym,orderId=77,id=1,qty='1'),
           dict(symbol=sym,orderId=77,id=2,qty='1')]
    positions=[dict(symbol=sym,positionSide=pos_side,positionAmt=qty)]
    algo=[dict(symbol=sym,clientAlgoId=ledger['client_algo_id'],algoId=101,
       side=stop_side,positionSide=pos_side,orderType='STOP_MARKET',
       closePosition=True,workingType='MARK_PRICE',algoStatus='NEW')]
    return reservation,ledger,order,fills,positions,algo

checks={}
for strategy in ('LONG1','SHORT1'):
    reservation,ledger,order,fills,positions,algo=fixtures(strategy)
    def check(name,*,r=None,l=None,o=None,f=None,p=None,a=None,
              orders=None,epoch=EPOCH,fence=FENCE,now=NOW,source_fail=None,
              expected='UNKNOWN_FREEZE'):
        row=assess_stage28(
            reservation=reservation if r is None else r,
            ledger=ledger if l is None else l,
            read_order=lambda sym,cid: (_ for _ in ()).throw(TimeoutError('network')) if source_fail=='order' else (order if o is None else o),
            read_fills=lambda sym,oid: (_ for _ in ()).throw(TimeoutError('network')) if source_fail=='fills' else (fills if f is None else f),
            read_positions=lambda sym: (_ for _ in ()).throw(TimeoutError('network')) if source_fail=='positions' else (positions if p is None else p),
            read_open_orders=lambda sym: (_ for _ in ()).throw(TimeoutError('network')) if source_fail=='orders' else ([] if orders is None else orders),
            read_open_algo=lambda sym: (_ for _ in ()).throw(TimeoutError('network')) if source_fail=='algo' else (algo if a is None else a),
            expected_epoch=epoch,expected_fence=fence,now_ms=now)
        assert row['state']==expected,(strategy,name,row)
        assert not any(row[z] for z in ('can_submit','can_resubmit','can_commit','can_release',
                                       'live_order_authorized','external_broker_proof_authenticated'))
        checks[strategy+'_'+name]=row['state']
    check('protected',expected='PROTECTED_EVIDENCE_ONLY')
    check('wrong_epoch',epoch=8)
    check('wrong_fence',fence=43)
    check('stale_ledger_epoch',l=dict(ledger,epoch=8))
    check('stale_ledger_fence',l=dict(ledger,fence=99))
    check('owner_mismatch',l=dict(ledger,reservation_owner='other'))
    check('intent_mismatch',l=dict(ledger,intent_id='other'))
    check('client_order_id_mismatch',l=dict(ledger,client_order_id='other'))
    check('reservation_digest_missing',r=dict(reservation,digest=''))
    check('reservation_expired',now=NOW+10001)
    check('wrong_order_identity',o=dict(order,clientOrderId='another'))
    check('wrong_order_side',o=dict(order,side='BUY' if strategy=='SHORT1' else 'SELL'))
    check('partial_order',o=dict(order,status='PARTIALLY_FILLED',executedQty='1'),
          expected='HOLD_RECONCILIATION')
    check('fills_incomplete',f=fills[:1])
    check('filled_order_uncommitted',l=dict(ledger,stage='UNKNOWN',filled_qty='0'),
          expected='RECOVERED_FILL_PENDING_JOURNAL')
    check('native_stop_missing',a=[],expected='FREEZE_UNPROTECTED')
    check('foreign_native_stop',a=[dict(algo[0],clientAlgoId='foreign')],expected='FREEZE_UNPROTECTED')
    check('contract_price_stop',a=[dict(algo[0],workingType='CONTRACT_PRICE')],expected='FREEZE_UNPROTECTED')
    check('position_qty_mismatch',p=[dict(positions[0],positionAmt='3')],expected='HOLD_RECONCILIATION')
    check('open_order_present',orders=[dict(symbol=reservation['symbol'],orderId=888)],
          expected='HOLD_RECONCILIATION')
    check('source_unavailable',source_fail='order')
    check('fills_unavailable',source_fail='fills')
    check('position_unavailable',source_fail='positions')
    check('algo_unavailable',source_fail='algo')
    check('flat_but_no_exit_proof',p=[],a=[],expected='HOLD_RECONCILIATION')
    check('flat_exit_history_candidate',p=[],a=[],
          l=dict(ledger,stage='FLAT_RECONCILED',
                 verified_exit_order_id='exit1',verified_exit_qty='2'),
          expected='FLAT_EVIDENCE_ONLY')
    check('flat_exit_qty_mismatch',p=[],a=[],
          l=dict(ledger,stage='FLAT_RECONCILED',
                 verified_exit_order_id='exit1',verified_exit_qty='1'),
          expected='HOLD_RECONCILIATION')
print(json.dumps({'strategies':2,'matrix_cases':len(checks),
  'protected_position_evidence_scenarios':2,
  'automatic_commits':0,'automatic_releases':0,
  'duplicate_submits_authorized':0,
  'real_exchange_calls':0,'production_redis_writes':0,
  'result':'PASS'},sort_keys=True))
