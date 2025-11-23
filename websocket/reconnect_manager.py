# websocket/reconnect_manager.py

import time
from typing import Callable, Optional

from core.logger import system_logger
from .binance_ws import start_ws_in_thread


def reconnect_ws_loop(
    ws_getter: Callable[[], Optional[object]],
    ws_setter: Callable[[object], None],
    retry_interval: int = 5,
) -> None:
    """
    Sürekli çalışan reconnect loop'u.

    Parametreler:
      - ws_getter: Şu anki ws instance'ını döndüren fonksiyon
      - ws_setter: Yeni ws instance'ını set eden fonksiyon
      - retry_interval: saniye cinsinden bekleme süresi

    Örnek kullanım (main.py içinde):
        current_ws = None

        def get_ws():
            return current_ws

        def set_ws(new_ws):
            nonlocal current_ws
            current_ws = new_ws

        # İlk bağlantı:
        current_ws = start_ws_in_thread()

        # Reconnect loop'u ayrı thread'te çalıştır:
        threading.Thread(
            target=reconnect_ws_loop,
            args=(get_ws, set_ws),
            daemon=True,
        ).start()
    """
    system_logger.info(
        f"[ReconnectManager] Reconnect loop started. interval={retry_interval}s"
    )

    while True:
        try:
            ws = ws_getter()
            if ws is None or ws.sock is None or not ws.sock.connected:
                system_logger.warning("[ReconnectManager] WebSocket disconnected. Reconnecting...")
                try:
                    new_ws = start_ws_in_thread()
                    ws_setter(new_ws)
                    system_logger.info("[ReconnectManager] Reconnected successfully.")
                except Exception as e:
                    system_logger.error(f"[ReconnectManager] Reconnect failed: {e}")
        except Exception as e:
            system_logger.error(f"[ReconnectManager] Unexpected error in loop: {e}")

        time.sleep(retry_interval)

