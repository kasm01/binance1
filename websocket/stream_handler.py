# websocket/stream_handler.py

import json
from typing import Any, Dict

from core.logger import system_logger
from trading.trade_executor import execute_trade


def handle_message(msg: str) -> None:
    """
    Gelen websocket mesajını ayrıştırır ve loglar.
    İsteğe bağlı olarak trade açmak için execute_trade kullanabilir.

    Binance Futures trade stream örnek mesaj:
      {
        "e": "trade",     // Event type
        "E": 123456789,   // Event time
        "s": "BTCUSDT",   // Symbol
        "t": 12345,       // Trade ID
        "p": "0.001",     // Price
        "q": "100",       // Quantity
        ...
      }

    NOT:
      - Binance trade stream aslında "signal" alanı içermez.
        "signal" anahtarını, ileride kendi sistemimizden gelen özel mesajlar
        için kullanabiliriz. Şimdilik varsa kullan, yoksa sadece logla.
    """
    try:
        data: Dict[str, Any] = json.loads(msg)

        symbol = data.get("s", "UNKNOWN")
        price = float(data.get("p", 0.0))
        qty = float(data.get("q", 0.0))

        # Özel/ekstra alan: kendi sistemin "signal" üretirse
        signal = data.get("signal")  # 'BUY' / 'SELL' vb.

        system_logger.info(
            f"[WebSocket] Message received | symbol={symbol}, price={price}, qty={qty}"
        )

        # Eğer mesajda özel bir "signal" alanı varsa VE miktar pozitifse trade dene
        if signal and isinstance(signal, str) and qty > 0:
            side = signal.upper()
            if side not in ("BUY", "SELL"):
                system_logger.warning(
                    f"[WebSocket] Unknown signal value: {signal}. Skipping trade."
                )
                return

            system_logger.info(
                f"[WebSocket] Executing trade from WS signal | {symbol} {side} qty={qty}"
            )
            order = execute_trade(symbol, side, qty)
            if order:
                system_logger.info(
                    f"[WebSocket] Trade executed successfully from WS: {order}"
                )
            else:
                system_logger.error(
                    f"[WebSocket] Trade execution failed from WS for {symbol} {side} {qty}"
                )

    except Exception as e:
        system_logger.error(f"[WebSocket] Error handling message: {e}", exc_info=True)

