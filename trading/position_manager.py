import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class PositionManager:
    """
    Açık pozisyonları hafızada izleyen basit pozisyon yöneticisi.
    Gerçek senaryoda bu bilgiler DB'ye/loglara da yazılmalıdır.
    """

    def __init__(self) -> None:
        # key: position_id, value: dict
        self.active_positions: Dict[str, Dict[str, Any]] = {}

    def open_position(self, symbol: str, qty: float, side: str, price: float) -> str:
        pos_id = f"{symbol}_{len(self.active_positions) + 1}"
        self.active_positions[pos_id] = {
            "symbol": symbol,
            "qty": float(qty),
            "side": side,
            "entry_price": float(price),
        }
        logger.info(f"[PositionManager] Position opened: {pos_id} -> {self.active_positions[pos_id]}")
        return pos_id

    def close_position(self, pos_id: str, exit_price: float) -> Optional[float]:
        """
        Pozisyonu kapatır ve PnL döner.
        Basit hesap: (exit - entry) * qty (side dikkate alınmamış basit versiyon).
        """
        if pos_id not in self.active_positions:
            logger.warning(f"[PositionManager] Tried to close unknown position: {pos_id}")
            return None

        pos = self.active_positions.pop(pos_id)
        entry = float(pos["entry_price"])
        qty = float(pos["qty"])
        side = pos.get("side", "LONG")

        raw_pnl = (float(exit_price) - entry) * qty

        # Side'a göre PnL ayarı (LONG/SHORT)
        if side.upper() == "SHORT":
            raw_pnl = -raw_pnl

        logger.info(
            f"[PositionManager] Position {pos_id} closed at {exit_price}. "
            f"Side={side}, Qty={qty}, Entry={entry}, PnL={raw_pnl:.4f}"
        )
        return raw_pnl

    def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.active_positions)

    def has_open_positions(self) -> bool:
        return len(self.active_positions) > 0

