"""Binance USD-M SHORT1 STOP_MARKET conditional Algo API adapter.

Never opens a position. All trade writes explicitly invoked only from the
live executor. No background networking. Prefix isolates SHORT1 stop ownership.
"""
from __future__ import annotations
import hashlib
import hmac
import os
import time
from urllib.parse import urlencode
from typing import Any
import requests
from core.binance_futures_v3 import signed_get, _base_url

PREFIX="b1s1_"

def _signed(method:str, path:str, args:dict[str,Any]):
    key=str(os.getenv("BINANCE_API_KEY") or "").strip()
    secret=str(os.getenv("BINANCE_API_SECRET") or "").strip()
    if not key or not secret:raise RuntimeError("missing_binance_credentials")
    p={**args,"timestamp":int(time.time()*1000),"recvWindow":5000}
    query=urlencode(p)
    sig=hmac.new(secret.encode(),query.encode(),hashlib.sha256).hexdigest()
    response=requests.request(method,f"{_base_url()}{path}?{query}&signature={sig}",
       headers={"X-MBX-APIKEY":key},timeout=8,allow_redirects=False)
    if response.status_code!=200:
        raise RuntimeError(f"algo_http_{response.status_code}:{response.text[:170]}")
    return response.json()

def get_open(symbol:str):
    out=signed_get("/fapi/v1/openAlgoOrders",{"symbol":str(symbol).upper()})
    if not isinstance(out,list):raise RuntimeError("invalid_algo_order_list")
    return out

def is_short_close_stop(v:dict[str,Any],symbol:str):
    return (v.get("symbol")==symbol and v.get("side")=="BUY"
            and v.get("positionSide")=="SHORT"
            and (v.get("orderType") or v.get("type"))=="STOP_MARKET"
            and str(v.get("closePosition")).lower()=="true")

def place(symbol:str, trigger_price:float, *, working_type="MARK_PRICE", price_protect=False):
    symbol=symbol.upper().strip()
    if not symbol or trigger_price<=0:raise ValueError("invalid_stop_symbol_or_price")
    existing=get_open(symbol)
    candidates=[v for v in existing if is_short_close_stop(v,symbol)]
    if candidates:
        raise RuntimeError("existing_short_stop_requires_explicit_reconciliation")
    cid=PREFIX+str(int(time.time()*1000))
    payload={"algoType":"CONDITIONAL","symbol":symbol,"side":"BUY",
        "positionSide":"SHORT","type":"STOP_MARKET",
        "triggerPrice":str(trigger_price),"closePosition":"true",
        "workingType":working_type,"priceProtect":"TRUE" if price_protect else "FALSE",
        "clientAlgoId":cid}
    try:
        created=_signed("POST","/fapi/v1/algoOrder",payload)
    except Exception:
        known=[v for v in get_open(symbol) if v.get("clientAlgoId")==cid]
        if not known:raise
        created=known[0]
    matching=[v for v in get_open(symbol) if v.get("clientAlgoId")==cid
              and is_short_close_stop(v,symbol)]
    if len(matching)!=1 or abs(float(matching[0].get("triggerPrice") or 0)-trigger_price)>1e-8:
        raise RuntimeError("short_algo_stop_not_verified")
    return dict(status="placed",stop_price=trigger_price,order=matching[0],created=created)

def cancel_owned(symbol:str,algo_id:Any):
    symbol=symbol.upper().strip()
    orders=get_open(symbol)
    owned=[v for v in orders if str(v.get("algoId"))==str(algo_id)
          and str(v.get("clientAlgoId") or "").startswith(PREFIX)
          and is_short_close_stop(v,symbol)]
    if not owned:return dict(status="not_owned_or_not_open")
    result=_signed("DELETE","/fapi/v1/algoOrder",{"symbol":symbol,"algoId":str(algo_id)})
    if any(str(v.get("algoId"))==str(algo_id) for v in get_open(symbol)):
        raise RuntimeError("short_algo_stop_still_open_after_cancel")
    return dict(status="canceled",result=result)
