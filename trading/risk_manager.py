import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Temel risk yönetimi:
      - max_risk_per_trade: toplam sermayenin yüzde kaçı tek işleme riske edilebilir.
    """

    def __init__(self, max_risk_per_trade: float = 0.02) -> None:
        """
        :param max_risk_per_trade: Örn: 0.02 => %2
        """
        self.max_risk = max_risk_per_trade

    def check_risk(self, capital: float, position_qty: float) -> bool:
        """
        :param capital: Toplam sermaye
        :param position_qty: İşleme girecek nominal pozisyon büyüklüğü
        :return: Risk limiti aşılmıyorsa True, aşıyorsa False
        """
        if capital <= 0:
            logger.warning("[RiskManager] Capital is non-positive, reject trade.")
            return False

        risk = position_qty / capital
        logger.info(
            f"[RiskManager] risk={risk:.4f}, max_risk={self.max_risk:.4f}"
        )

        if risk > self.max_risk:
            logger.warning(
                f"[RiskManager] Risk limit exceeded: {risk:.4f} > {self.max_risk:.4f}"
            )
            return False
        return True
