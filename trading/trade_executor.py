# trading/trade_executor.py

from __future__ import annotations

from typing import Any, Dict, Optional, List

from core.logger import system_logger
from core.exceptions import TradeExecutionException
from config.settings import Config
from trading.risk_manager import RiskManager
from trading.position_manager import PositionManager


class TradeExecutor:
    """
    Binance futures emir katmanı.

    - RiskManager: max_risk_per_trade + max_daily_loss
    - PositionManager: açık LONG / SHORT pozisyonları takip eder
    - DRY-RUN / LIVE modu: Config.LIVE_TRADING_ENABLED ile kontrol
    """

    def __init__(
        self,
        client: Any,
        risk_manager: RiskManager,
        position_manager: PositionManager,
    ):
        """
        :param client: python-binance futures client (binance.client.Client)
        :param risk_manager: RiskManager instance
        :param position_manager: PositionManager instance
        """
        self.client = client
        self.risk_manager = risk_manager
        self.position_manager = position_manager

    # ────────────────────────── yardımcı: equity ──────────────────────────

    def get_current_equity(self) -> float:
        """
        Binance futures hesabından USDT cinsinden özsermaye çeker.
        """
        try:
            account = self.client.futures_account()
            total_wallet_balance = float(account["totalWalletBalance"])
            return total_wallet_balance
        except Exception as e:
            system_logger.exception(f"[TRADE] Failed to fetch account equity: {e}")
            # Equity alınamazsa çok agresif olmamak için 0 döndür.
            return 0.0

    # ─────────────────────── pozisyon açma (sinyal) ───────────────────────

    def open_position_from_signal(
        self,
        symbol: str,
        direction: str,  # "LONG" veya "SHORT"
        entry_price: float,
        stop_loss_pct: float,
        leverage: float,
    ) -> Dict[str, Any]:
        """
        Sinyal geldikten sonra risk yönetimi ile birlikte pozisyon açar.

        1) Günlük zarar limiti kontrolü
        2) Pozisyon büyüklüğü hesabı (qty)
        3) DRY-RUN veya gerçek emir
        4) PositionManager'a kaydetme
        """
        direction = direction.upper()
        if direction not in ("LONG", "SHORT"):
            msg = f"Invalid direction: {direction}"
            system_logger.error(f"[TRADE] {msg}")
            return {"status": "failed", "reason": msg}

        # 1) Günlük zarar limiti kontrolü
        equity = self.get_current_equity()
        allowed, reason = self.risk_manager.can_open_new_trade(equity)
        if not allowed:
            system_logger.warning(f"[TRADE] New trade blocked by risk manager: {reason}")
            return {"status": "rejected", "reason": reason}

        # 2) Pozisyon büyüklüğü hesabı
        qty = self.risk_manager.compute_position_size(
            symbol=symbol,
            side=direction,
            entry_price=entry_price,
            equity=equity,
            stop_loss_pct=stop_loss_pct,
            leverage=leverage,
        )
        if qty <= 0:
            msg = "Invalid position size (qty <= 0)"
            system_logger.warning(f"[TRADE] {msg} for {symbol} {direction}")
            return {"status": "rejected", "reason": msg}

        # DRY-RUN (gerçek emir yok, sadece log + PositionManager)
        if not Config.LIVE_TRADING_ENABLED:
            system_logger.info(
                f"[TRADE] DRY-RUN OPEN {symbol} {direction} "
                f"qty={qty:.6f} price={entry_price:.2f} lev={leverage}"
            )
            position = self.position_manager.open_position(
                symbol=symbol,
                side=direction,
                qty=qty,
                entry_price=entry_price,
                leverage=leverage,
            )
            return {"status": "dry_run", "position": position}

        # 3) Gerçek Binance emri
        try:
            side_for_binance = "BUY" if direction == "LONG" else "SELL"

            order = self.client.futures_create_order(
                symbol=symbol,
                side=side_for_binance,
                type="MARKET",
                quantity=qty,
            )

            # Fiyatı order'dan çek (fills veya avgPrice)
            fills = order.get("fills") or []
            if fills:
                fill_price = float(fills[0]["price"])
            else:
                fill_price = float(order.get("avgPrice") or entry_price)

            position = self.position_manager.open_position(
                symbol=symbol,
                side=direction,
                qty=qty,
                entry_price=fill_price,
                leverage=leverage,
            )

            system_logger.info(
                f"[TRADE] OPENED {symbol} {direction} "
                f"qty={qty:.6f} price={fill_price:.2f} lev={leverage}"
            )
            return {"status": "success", "order": order, "position": position}

        except Exception as e:
            system_logger.exception(
                f"[TRADE] Failed to open position {symbol} {direction}: {e}"
            )
            raise TradeExecutionException(str(e)) from e

    # ─────────────────────── pozisyon kapama ───────────────────────

    def close_position(
        self,
        symbol: str,
        direction: str,  # pozisyon yönü: "LONG" veya "SHORT"
        exit_price: float,
    ) -> Dict[str, Any]:
        """
        Mevcut LONG/SHORT pozisyonu kapatır ve realized PnL'i RiskManager'a işler.
        """
        direction = direction.upper()
        if direction not in ("LONG", "SHORT"):
            msg = f"Invalid direction: {direction}"
            system_logger.error(f"[TRADE] {msg}")
            return {"status": "failed", "reason": msg}

        position = self.position_manager.get_position(symbol, direction)
        if not position:
            system_logger.warning(
                f"[TRADE] No open position to close: {symbol} {direction}"
            )
            return {"status": "not_found"}

        qty = position.qty

        # DRY-RUN
        if not Config.LIVE_TRADING_ENABLED:
            pnl = self.position_manager.close_position(
                symbol=symbol, side=direction, exit_price=exit_price
            )
            self.risk_manager.register_closed_trade(pnl)
            system_logger.info(
                f"[TRADE] DRY-RUN CLOSE {symbol} {direction} "
                f"qty={qty:.6f} price={exit_price:.2f} pnl={pnl:.2f}"
            )
            return {"status": "dry_run", "pnl": pnl}

        # Gerçek Binance kapama emri
        try:
            side_for_binance = "SELL" if direction == "LONG" else "BUY"

            close_order = self.client.futures_create_order(
                symbol=symbol,
                side=side_for_binance,
                type="MARKET",
                quantity=qty,
                reduceOnly=True,  # sadece açık pozisyonu kapat
            )

            fills = close_order.get("fills") or []
            if fills:
                fill_price = float(fills[0]["price"])
            else:
                fill_price = float(close_order.get("avgPrice") or exit_price)

            pnl = self.position_manager.close_position(
                symbol=symbol, side=direction, exit_price=fill_price
            )
            self.risk_manager.register_closed_trade(pnl)

            system_logger.info(
                f"[TRADE] CLOSED {symbol} {direction} "
                f"qty={qty:.6f} price={fill_price:.2f} pnl={pnl:.2f}"
            )
            return {"status": "success", "pnl": pnl, "order": close_order}

        except Exception as e:
            system_logger.exception(
                f"[TRADE] Failed to close position {symbol} {direction}: {e}"
            )
            raise TradeExecutionException(str(e)) from e

    # ─────────────────────── tüm pozisyonları kapat ───────────────────────

    def flatten_all_positions(self, current_prices: Dict[str, float]) -> None:
        """
        Günlük zarar limiti aşıldığında TÜM açık pozisyonları kapatmak için kullanılır.

        :param current_prices: {'BTCUSDT': 67000.0, ...}
        """
        open_positions: List[Any] = self.position_manager.list_open_positions()
        if not open_positions:
            system_logger.info("[TRADE] No open positions to flatten.")
            return

        system_logger.warning(
            f"[TRADE] Flatten ALL positions due to daily loss limit. "
            f"open_count={len(open_positions)}"
        )

        for pos in open_positions:
            price = current_prices.get(pos.symbol)
            if price is None:
                system_logger.warning(
                    f"[TRADE] No price for {pos.symbol}, skipping flatten."
                )
                continue

            # pos.side: "LONG" veya "SHORT"
            try:
                self.close_position(
                    symbol=pos.symbol,
                    direction=pos.side,
                    exit_price=price,
                )
            except TradeExecutionException:
                # Hata alınsa bile diğer pozisyonlara devam edilsin
                continue

