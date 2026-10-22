from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests


def _base_url() -> str:
    if str(os.getenv("BINANCE_TESTNET", "0")).strip().lower() in {"1", "true", "yes", "on"}:
        return "https://testnet.binancefuture.com"
    return str(os.getenv("BINANCE_FUTURES_BASE_URL", "https://fapi.binance.com") or "https://fapi.binance.com").rstrip("/")


def _credentials() -> tuple[str, str]:
    key = str(os.getenv("BINANCE_API_KEY", "") or "").strip()
    secret = str(os.getenv("BINANCE_API_SECRET", "") or "").strip()
    if not key or not secret:
        raise RuntimeError("binance_futures_v3_missing_credentials")
    return key, secret


def signed_get(path: str, params: Optional[Dict[str, Any]] = None, timeout: float = 8.0) -> Any:
    key, secret = _credentials()
    p: Dict[str, Any] = dict(params or {})
    p.setdefault("timestamp", int(time.time() * 1000))
    p.setdefault("recvWindow", 5000)
    query = urlencode(p)
    signature = hmac.new(secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
    url = f"{_base_url()}{path}?{query}&signature={signature}"
    resp = requests.get(url, headers={"X-MBX-APIKEY": key}, timeout=float(timeout), allow_redirects=False)
    if resp.status_code != 200:
        body = (resp.text or "")[:240].replace("\n", " ")
        raise RuntimeError(f"binance_futures_v3_http_{resp.status_code}:{body}")
    return resp.json()


def account() -> Dict[str, Any]:
    out = signed_get("/fapi/v3/account")
    if not isinstance(out, dict):
        raise RuntimeError("binance_futures_v3_account_invalid_payload")
    return out


def balance() -> List[Dict[str, Any]]:
    out = signed_get("/fapi/v3/balance")
    if not isinstance(out, list):
        raise RuntimeError("binance_futures_v3_balance_invalid_payload")
    return [row for row in out if isinstance(row, dict)]


def position_risk(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {}
    if symbol:
        params["symbol"] = str(symbol).upper().strip()
    out = signed_get("/fapi/v3/positionRisk", params=params)
    if isinstance(out, dict):
        out = [out]
    if not isinstance(out, list):
        raise RuntimeError("binance_futures_v3_position_invalid_payload")
    return [row for row in out if isinstance(row, dict)]
