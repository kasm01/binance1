#!/usr/bin/env python3
"""Bounded signed GET only: last 100 orders/trades/algo per symbol.

No Binance writes or Redis mutation; aggregate result stores hashed IDs only.
"""
from pathlib import Path
from dotenv import load_dotenv
import json,time
load_dotenv('.env')
load_dotenv('.runtime_secrets.env',override=False)
from core.binance_futures_v3 import signed_get
from orchestration.risk.binance_order_fill_evidence_v1 import reconcile_orders
res={}
for symbol in ('NEARUSDT','INJUSDT'):
 try:
    orders=signed_get('/fapi/v1/allOrders',{'symbol':symbol,'limit':100})
    fills=signed_get('/fapi/v1/userTrades',{'symbol':symbol,'limit':100})
    algos=signed_get('/fapi/v1/allAlgoOrders',{'symbol':symbol,'limit':100})
    res[symbol]=reconcile_orders(symbol,orders,fills,algos)
 except Exception as exc:
    res[symbol]=dict(state='UNKNOWN',reason=type(exc).__name__,
        safe_to_auto_release=False,history_is_complete=False)
report=dict(schema='short1_long1_historical_order_ledger_read_only_v1',
 checked_at_unix_sec=time.time(),symbol_evidence=res,
 safe_to_auto_release=False,
 master_approved=False,
 notes=[
 'This is at most 100 latest records per symbol, not a complete transaction/intent journal.',
 'AllAlgoOrders history is not a proof that expiry means an exchange position closed.',
 'BNB commissions are native BNB units and cannot be subtracted from USDT PnL without exchange conversion at fill time.',
 'Query response includes hashed IDs only; no API keys or raw order client IDs saved.',
 'Neither positions nor orders are modified; release requires owner token and independent flat evidence.'
 ])
file=Path('research/results/LONG1_SHORT1_HISTORICAL_FILL_LEDGER_READONLY_20261114.json')
file.parent.mkdir(parents=True,exist_ok=True)
file.write_text(json.dumps(report,indent=2))
for symbol,v in res.items():
 print(symbol,'STATE',v['state'],'ORDERS',v.get('order_count'),'FILLS',v.get('trade_count'),
       'MATCH',v.get('match_states'),'ORPHAN',v.get('orphan_trade_order_count'),
       'ALGOS',v.get('algo_status'),'COMMISSION_UNITS',list((v.get('commissions_by_asset') or {}).keys()))
print('AUTORELEASE',report['safe_to_auto_release'],'REPORT',file)
