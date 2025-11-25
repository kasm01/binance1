import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """
    Tek bir pozisyonu temsil eden basit veri sınıfı.
    """
    symbol: str
    side: str          # "LONG" / "SHORT"
    qty: float
    entry_price: float
    leverage: float = 1.0


class PositionManager:
    """
    Açık pozisyonları hafızada izleyen pozisyon yöneticisi.

    Ana API:
      - open_position(symbol, side, qty, entry_price, leverage=1.0) -> Position
      - get_position(symbol, side) -> Optional[Position]
      - close_position(symbol, side, exit_price) -> Optional[float] (PnL)
      - list_open_positions() -> List[Position]
      - has_open_positions() -> bool
    """

    def __init__(self) -> None:
        # key: (symbol, side), value: Position
        self.active_positions: Dict[Tuple[str, str], Position] = {}

    # ───────────────────────── pozisyon açma ─────────────────────────

    def open_position(
        self,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        leverage: float = 1.0,
    ) -> Position:
        side = side.upper()
        key = (symbol, side)

        pos = Position(
            symbol=symbol,
            side=side,
            qty=float(qty),
            entry_price=float(entry_price),
            leverage=float(leverage),
        )
        self.active_positions[key] = pos

        logger.info(
            "[PositionManager] Position opened: %s %s qty=%.6f entry=%.2f lev=%.1f",
            symbol,
            side,
            pos.qty,
            pos.entry_price,
            pos.leverage,
        )
        return pos

    # ───────────────────────── pozisyon sorgulama ─────────────────────────

    def get_position(self, symbol: str, side: str) -> Optional[Position]:
        """
        Sembol + yön (LONG/SHORT) ile pozisyonu getirir.
        """
        side = side.upper()
        return self.active_positions.get((symbol, side))

    # ───────────────────────── pozisyon kapama ─────────────────────────

    def close_position(
        self,
        symbol: str,
        side: str,
        exit_price: float,
    ) -> Optional[float]:
        """
        Pozisyonu kapatır ve PnL döner.
        PnL hesabı:
          LONG  -> (exit - entry) * qty
          SHORT -> (entry - exit) * qty
        """
        side = side.upper()
        key = (symbol, side)

        if key not in self.active_positions:
            logger.warning(
                "[PositionManager] Tried to close unknown position: %s %s",
                symbol,
                side,
            )
            return None

        pos = self.active_positions.pop(key)
        entry = pos.entry_price
        qty = pos.qty
        exit_price = float(exit_price)

        if side == "LONG":
            pnl = (exit_price - entry) * qty
        else:  # SHORT
            pnl = (entry - exit_price) * qty

        logger.info(
            "[PositionManager] Position closed: %s %s qty=%.6f entry=%.2f "
            "exit=%.2f pnl=%.4f",
            symbol,
            side,
            qty,
            entry,
            exit_price,
            pnl,
        )
        return pnl

    # ───────────────────────── yardımcılar ─────────────────────────

    def list_open_positions(self) -> List[Position]:
        """
        Tüm açık pozisyonları liste olarak döner (TradeExecutor.flatten_all_positions için).
        """
        return list(self.active_positions.values())

    def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Eski testleri (tests/test_trading.py) bozmamak için
        backward-compatible bir görünüm döndürüyoruz.
        key: "SYMBOL_SIDE", value: dict(...)
        """
        result: Dict[str, Dict[str, Any]] = {}
        for pos in self.active_positions.values():
            key = f"{pos.symbol}_{pos.side}"
            result[key] = {
                "symbol": pos.symbol,
                "side": pos.side,
                "qty": pos.qty,
                "entry_price": pos.entry_price,
                "leverage": pos.leverage,
            }
        return result

    def has_open_positions(self) -> bool:
        return len(self.active_positions) > 0

