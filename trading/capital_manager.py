import logging

logger = logging.getLogger(__name__)


class CapitalManager:
    """
    Sermaye yönetimi:
      - toplam sermaye
      - kullanılabilir sermaye
      - allocate / release
    """

    def __init__(self, total_capital: float) -> None:
        self.total_capital = float(total_capital)
        self.available_capital = float(total_capital)

    def allocate(self, risk_pct: float) -> float:
        """
        :param risk_pct: Örn: 0.01 => kullanılabilir sermayenin %1'i
        :return: Ayrılan sermaye miktarı
        """
        if risk_pct <= 0:
            logger.warning("[CapitalManager] risk_pct <= 0, allocation skipped.")
            return 0.0

        capital_to_use = self.available_capital * risk_pct
        self.available_capital -= capital_to_use

        logger.info(
            f"[CapitalManager] Allocated {capital_to_use:.4f}, "
            f"remaining available={self.available_capital:.4f}"
        )
        return capital_to_use

    def release(self, amount: float) -> None:
        """
        :param amount: İşlemden dönen (serbest bırakılan) sermaye
        """
        self.available_capital += float(amount)
        logger.info(
            f"[CapitalManager] Released {amount:.4f}, "
            f"available_capital={self.available_capital:.4f}"
        )
