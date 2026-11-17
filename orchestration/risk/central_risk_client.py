from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, Optional

import redis


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


class CentralRiskClient:
    """Thin Redis client for the single central risk authority.

    This client contains no portfolio policy. It only requests a decision from
    the central gate and returns the authoritative response.
    """

    def __init__(self, redis_client: Optional[redis.Redis] = None) -> None:
        self.enabled = _env_bool("CENTRAL_RISK_ENFORCE_DRY_RUN", False)
        # Exact matching central reservation request ID, never a strategy-only
        # release. A restart that loses this handle intentionally fails closed.
        self._dry_reservation_owner_by_strategy: Dict[str, str] = {}
        self.request_stream = os.getenv("CENTRAL_RISK_REQUEST_STREAM", "central:risk:requests:v1")
        self.response_prefix = os.getenv("CENTRAL_RISK_RESPONSE_PREFIX", "central:risk:response:")
        self.timeout_sec = max(0.1, float(os.getenv("CENTRAL_RISK_REQUEST_TIMEOUT_SEC", "2.0") or 2.0))
        self.poll_sec = max(0.01, float(os.getenv("CENTRAL_RISK_RESPONSE_POLL_SEC", "0.05") or 0.05))
        if redis_client is not None:
            self.r = redis_client
        else:
            self.r = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379") or 6379),
                db=int(os.getenv("REDIS_DB", "0") or 0),
                decode_responses=True,
                socket_timeout=1.0,
            )

    def _ask(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return {"state": "BYPASS", "reason": "central_risk_dry_enforcement_disabled"}
        request_id = str(payload.get("request_id") or uuid.uuid4())
        req = dict(payload)
        req["request_id"] = request_id
        try:
            self.r.xadd(self.request_stream, {"json": json.dumps(req, separators=(",", ":"))})
        except Exception as exc:
            return {"state": "REJECT", "reason": "central_risk_request_publish_failed", "detail": str(exc)[:160]}

        key = f"{self.response_prefix}{request_id}"
        deadline = time.monotonic() + self.timeout_sec
        while time.monotonic() < deadline:
            try:
                raw = self.r.get(key)
            except Exception as exc:
                return {"state": "REJECT", "reason": "central_risk_response_read_failed", "detail": str(exc)[:160]}
            if raw:
                try:
                    obj = json.loads(raw)
                    return obj if isinstance(obj, dict) else {"state": "REJECT", "reason": "central_risk_bad_response"}
                except Exception:
                    return {"state": "REJECT", "reason": "central_risk_bad_response"}
            time.sleep(self.poll_sec)
        return {"state": "REJECT", "reason": "central_risk_timeout"}

    def ping(self) -> Dict[str, Any]:
        return self._ask({"action": "PING", "request_mode": "DRY_RUN"})

    def evaluate_open(
        self,
        *,
        strategy_id: str,
        symbol: str,
        side: str,
        market_context_multiplier: float,
        reserve: bool = True,
    ) -> Dict[str, Any]:
        strategy = str(strategy_id or "").upper().strip()
        response = self._ask({
            "action": "EVALUATE_OPEN",
            "request_mode": "DRY_RUN",
            "strategy_id": strategy,
            "symbol": str(symbol or "").upper().strip(),
            "side": str(side or "").lower().strip(),
            "market_context_multiplier": float(market_context_multiplier),
            "reserve": bool(reserve),
        })
        # Persist ownership only after a successful reserved response; never
        # infer it from a caller-supplied request_id or READY without reserve.
        if reserve and response.get("state") == "READY" and response.get("request_id"):
            self._dry_reservation_owner_by_strategy[strategy] = str(response["request_id"])
        return response

    def release(self, *, strategy_id: str) -> Dict[str, Any]:
        strategy = str(strategy_id or "").upper().strip()
        owner = self._dry_reservation_owner_by_strategy.get(strategy)
        if not owner:
            return {"state": "REJECT", "reason": "reservation_owner_unknown",
                    "strategy_id": strategy}
        result = self._ask({
            "action": "RELEASE",
            "request_mode": "DRY_RUN",
            "strategy_id": strategy,
            "reservation_owner_request_id": owner,
        })
        if result.get("state") == "READY" and result.get("reason") == "reservation_released_by_owner":
            self._dry_reservation_owner_by_strategy.pop(strategy, None)
        return result
