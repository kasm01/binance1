import logging
from typing import Any, Dict, Optional


class TradeExecutor:
    """
    Basit bir TradeExecutor iskeleti.
    Şu an sadece log atıyor, gerçek emir gönderimi TODO.
    """

    def __init__(
        self,
        client: Any | None = None,
        risk_manager: Any | None = None,
        position_manager: Any | None = None,
        dry_run: bool = True,
    ) -> None:
        # main.py -> create_trading_objects içinde verilen objeleri tutalım
        self.client = client
        self.risk_manager = risk_manager
        self.position_manager = position_manager
        self.dry_run = dry_run

        # main içindeki logger setup ile uyumlu
        self.logger = logging.getLogger("system")

    async def execute_decision(
        self,
        signal: str,
        symbol: str,
        price: float,
        size: Optional[float] = None,
        interval: Optional[str] = None,
        training_mode: Optional[bool] = None,
        hybrid_mode: Optional[bool] = None,
        probs: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Şimdilik sadece loglayan basit iskelet.
        İleride burada gerçek Binance emir fonksiyonları çağrılabilir.
        """

        self.logger.info(
            "[EXEC] execute_decision çağrıldı | "
            "signal=%s symbol=%s price=%s size=%s interval=%s "
            "training_mode=%s hybrid_mode=%s probs=%s extra=%s",
            signal,
            symbol,
            price,
            size,
            interval,
            training_mode,
            hybrid_mode,
            probs,
            extra,
        )

        if self.dry_run:
            self.logger.info(
                "[EXEC] DRY_RUN=True olduğu için gerçek emir gönderilmeyecek."
            )
            return

        # TODO: Buraya gerçek emir gönderme mantığı eklenecek
