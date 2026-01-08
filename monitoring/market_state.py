import threading
from typing import Any, Dict, Optional


class MarketState:
    """
    Thread-safe in-memory market snapshot store.
    OKX WS gibi izleme kaynakları buraya yazar; bot loop / TG komutları buradan okur.
    """
    _lock = threading.Lock()
    _state: Dict[str, Any] = {
        "okx": {
            "ticker": None,
            "ts": None,
        }
    }

    @classmethod
    def set_okx_ticker(cls, ticker: Dict[str, Any], ts: Optional[float] = None) -> None:
        with cls._lock:
            cls._state["okx"]["ticker"] = ticker
            cls._state["okx"]["ts"] = ts

    @classmethod
    def get_snapshot(cls) -> Dict[str, Any]:
        with cls._lock:
            # shallow copy yeterli
            return {
                "okx": dict(cls._state.get("okx", {})),
            }
