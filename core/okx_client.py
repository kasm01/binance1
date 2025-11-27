# core/okx_client.py

import os
import time
import hmac
import base64
import json
from hashlib import sha256
from typing import Any, Dict, Optional, List

import requests

# python-dotenv varsa yerelde .env okumak için kullanacağız
try:
    from dotenv import load_dotenv
except ImportError:  # Cloud Run'da olmayabilir, sorun değil
    load_dotenv = None


class OkxClient:
    """
    Basit OKX REST client
    - env'den OKX_API_KEY, OKX_API_SECRET, OKX_PASSPHRASE, OKX_BASE_URL okur
    - server time kullanarak imza atar (timestamp hatasını kaldırmak için)
    - basit balance, klines, order fonksiyonları içerir
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        passphrase: Optional[str] = None,
        base_url: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        # Yerelde çalışıyorsan .env yükle (Cloud Run'da zaten ENV ile gelecek)
        if load_dotenv is not None:
            load_dotenv(dotenv_path=".env", override=True)

        self.api_key = api_key or os.getenv("OKX_API_KEY", "")
        self.api_secret = api_secret or os.getenv("OKX_API_SECRET", "")
        self.passphrase = passphrase or os.getenv("OKX_PASSPHRASE", "")
        self.base_url = base_url or os.getenv("OKX_BASE_URL", "https://www.okx.com")

        self.session = session or requests.Session()

        # Basit debug
        print("[OKX DEBUG] API_KEY:", (self.api_key[:4] + "****") if self.api_key else "EMPTY")
        print("[OKX DEBUG] SECRET set?    ", bool(self.api_secret))
        print("[OKX DEBUG] PASSPHRASE set?", bool(self.passphrase))

        if not (self.api_key and self.api_secret and self.passphrase):
            raise ValueError(
                "OKX API bilgileri eksik: OKX_API_KEY / OKX_API_SECRET / OKX_PASSPHRASE"
            )

    # -------------------------------------------------------------------------
    #  Low-level yardımcılar
    # -------------------------------------------------------------------------

    def _timestamp(self) -> str:
        """
        OKX'in public time endpoint'inden server timestamp alır.
        Eğer herhangi bir sorun olursa local time'a düşer.
        """
        try:
            r = self.session.get(self.base_url + "/api/v5/public/time", timeout=5)
            r.raise_for_status()
            data = r.json()
            server_ms = data["data"][0]["ts"]      # "1732612345678" (ms)
            ts = str(float(server_ms) / 1000.0)    # saniye cinsinden string
            return ts
        except Exception as e:
            print("[OKX WARN] server time alınamadı, local time kullanılacak:", e)
            return str(time.time())

    @staticmethod
    def _sign(message: str, secret: str) -> str:
        """
        HMAC-SHA256 + base64 imza
        """
        return base64.b64encode(
            hmac.new(secret.encode(), message.encode(), sha256).digest()
        ).decode()

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        auth: bool = False,
    ) -> Dict[str, Any]:
        """
        Genel HTTP wrapper.
        - method: "GET" / "POST"
        - path: "/api/v5/..."
        - params: query string
        - body: JSON body
        - auth: True ise imzalı istek
        """
        url = self.base_url + path
        method = method.upper()

        headers: Dict[str, str] = {}
        req_body_str = ""

        if auth:
            ts = self._timestamp()
            # GET isteklerde signature body kısmı "" olmalı
            if method == "GET":
                sign_body = ""
            else:
                sign_body = json.dumps(body or {})
                req_body_str = sign_body

            sign_msg = ts + method + path + sign_body
            signature = self._sign(sign_msg, self.api_secret)

            headers.update(
                {
                    "OK-ACCESS-KEY": self.api_key,
                    "OK-ACCESS-SIGN": signature,
                    "OK-ACCESS-TIMESTAMP": ts,
                    "OK-ACCESS-PASSPHRASE": self.passphrase,
                    "Content-Type": "application/json",
                }
            )
        else:
            if method != "GET":
                headers["Content-Type"] = "application/json"
            if body:
                req_body_str = json.dumps(body)

        # Debug (istersen kapatabilirsin)
        # print("[OKX DEBUG] Request:", method, url)
        # print("[OKX DEBUG] Headers:", headers)
        # if req_body_str:
        #     print("[OKX DEBUG] Body:", req_body_str)

        resp = self.session.request(
            method,
            url,
            headers=headers,
            params=params if method == "GET" else None,
            data=req_body_str if method != "GET" and req_body_str else None,
            timeout=10,
        )

        if resp.status_code != 200:
            print(f"[OKX DEBUG] HTTP {resp.status_code} for {path}")
            print("[OKX DEBUG] Raw body:", resp.text[:400])

        resp.raise_for_status()
        data = resp.json()

        # OKX response format: {"code":"0","msg":"","data":[...]}
        code = str(data.get("code", "0"))
        msg = data.get("msg", "")
        if code != "0":
            print(f"[OKX ERROR] code={code}, msg={msg}")
            raise RuntimeError(f"OKX API error: {code} {msg}")

        return data

    # -------------------------------------------------------------------------
    #  Yüksek seviye fonksiyonlar
    # -------------------------------------------------------------------------

    def get_usdt_balance(self) -> float:
        """
        /api/v5/account/balance
        totalEq veya detaylardan USDT bakiyeyi döndürür.
        """
        data = self._request("GET", "/api/v5/account/balance", auth=True)
        d0 = (data.get("data") or [{}])[0]

        # Önce totalEq
        total_eq = d0.get("totalEq")
        try:
            if total_eq is not None:
                return float(total_eq)
        except Exception:
            pass

        # Olmazsa detaylardan USDT ara
        details: List[Dict[str, Any]] = d0.get("details") or []
        for item in details:
            if item.get("ccy") == "USDT":
                eq = item.get("eq") or item.get("availEq")
                if eq is not None:
                    return float(eq)

        return 0.0

    def get_klines(
        self,
        inst_id: str = "BTC-USDT-SWAP",
        bar: str = "1m",
        limit: int = 100,
    ) -> List[List[str]]:
        """
        /api/v5/market/candles
        OKX formatında klines döndürür (liste listesi).
        """
        params = {
            "instId": inst_id,
            "bar": bar,
            "limit": str(limit),
        }
        data = self._request("GET", "/api/v5/market/candles", params=params, auth=False)
        # data["data"] zaten klines
        return data.get("data", [])

    def place_order(
        self,
        inst_id: str,
        side: str,
        sz: str,
        td_mode: str = "cross",
        ord_type: str = "market",
        px: Optional[str] = None,
        pos_side: Optional[str] = None,
        cl_ord_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        /api/v5/trade/order ile basit order açma.
        Örnek:
            place_order(
               inst_id="BTC-USDT-SWAP",
               side="buy",
               sz="1",
               td_mode="cross",
               ord_type="market",
            )
        """
        body: Dict[str, Any] = {
            "instId": inst_id,
            "side": side,            # "buy" / "sell"
            "tdMode": td_mode,       # "cross" / "isolated"
            "ordType": ord_type,     # "market" / "limit"
            "sz": sz,
        }
        if px is not None:
            body["px"] = px
        if pos_side is not None:
            body["posSide"] = pos_side   # "long" / "short"
        if cl_ord_id is not None:
            body["clOrdId"] = cl_ord_id

        data = self._request("POST", "/api/v5/trade/order", body=body, auth=True)
        # genelde data["data"][0] order detayı
        items = data.get("data") or []
        return items[0] if items else {}

