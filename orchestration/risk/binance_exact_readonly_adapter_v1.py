"""Strict, bounded signed GET adapter for research-only exact intent recovery.

No POST/PUT/DELETE. Never synthesizes an order or retrieves entire history.
An adapter read is NOT signed authorization to commit/release central funds.
"""
from __future__ import annotations
from typing import Callable
from core.binance_futures_v3 import signed_get,position_risk

ALLOWED=('NEARUSDT','INJUSDT')
def _sym(symbol):
    symbol=str(symbol).strip().upper()
    if symbol not in ALLOWED:raise ValueError('symbol_not_allowed')
    return symbol
def _id(value):
    value=str(value or '').strip()
    if not value or len(value)>64 or any(c not in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-' for c in value):
        raise ValueError('invalid_exact_order_identity')
    return value
class ReadOnlyBinanceIntentAdapter:
    def __init__(self,*,get:Callable=signed_get,positions:Callable=position_risk):
        self._get=get
        self._positions=positions
    def get_order(self,symbol,client_order_id):
        sym=_sym(symbol);cid=_id(client_order_id)
        value=self._get('/fapi/v1/order',{'symbol':sym,'origClientOrderId':cid})
        if not isinstance(value,dict):raise ValueError('invalid_exact_order_response')
        return value
    def get_fills(self,symbol,order_id):
        sym=_sym(symbol);oid=_id(order_id)
        rows=self._get('/fapi/v1/userTrades',{'symbol':sym,'orderId':oid,'limit':100})
        if not isinstance(rows,list) or any(not isinstance(x,dict) for x in rows):
            raise ValueError('invalid_exact_fills_response')
        # Limit=100 cannot prove complete fills for orders with >=100 trades;
        # force UNKNOWN instead of assuming truncated history is complete.
        if len(rows)>=100:raise ValueError('potentially_truncated_fills')
        return rows
    def get_positions(self,symbol):
        sym=_sym(symbol);rows=self._positions(sym)
        if not isinstance(rows,list):raise ValueError('invalid_position_risk_response')
        return rows
    def get_open_orders(self,symbol):
        sym=_sym(symbol)
        rows=self._get('/fapi/v1/openOrders',{'symbol':sym})
        if not isinstance(rows,list):raise ValueError('invalid_open_orders_response')
        return rows
    def get_open_algos(self,symbol):
        sym=_sym(symbol)
        rows=self._get('/fapi/v1/openAlgoOrders',{'symbol':sym})
        if not isinstance(rows,list):raise ValueError('invalid_open_algo_response')
        return rows
    def get_exact_algo(self,symbol,client_algo_id):
        sym=_sym(symbol);cid=_id(client_algo_id)
        row=self._get('/fapi/v1/algoOrder',{'clientAlgoId':cid})
        if not isinstance(row,dict) or row.get('symbol')!=sym or str(row.get('clientAlgoId'))!=cid:
            raise ValueError('algo_identity_mismatch')
        return row
