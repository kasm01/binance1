# monitoring/alert_system.py
from dataclasses import dataclass
from typing import Optional, Dict, Any

from core.logger import system_logger
from core.notifier import Notifier


@dataclass
class AlertConfig:
    """
    Alert sistemi için basit konfig.
    """
    enable_telegram: bool = True   # Telegram bildirimleri aktif mi?
    min_level: str = "WARNING"     # INFO < WARNING < ERROR < CRITICAL


class AlertSystem:
    """
    Trade / sistem uyarılarını Telegram (ve ileride başka kanallara) gönderen basit wrapper.

    - core.notifier.Notifier sınıfını kullanır.
    - send_notification fonksiyonu ARTIK yok; bunun yerine AlertSystem.send_alert kullanılır.
    """

    def __init__(
        self,
        notifier: Optional[Notifier] = None,
        config: Optional[AlertConfig] = None,
    ) -> None:
        self.notifier = notifier or Notifier()
        self.config = config or AlertConfig()

    # -----------------------------
    # İç yardımcılar
    # -----------------------------
    def _should_notify(self, level: str) -> bool:
        levels = ["INFO", "WARNING", "ERROR", "CRITICAL"]
        try:
            return levels.index(level) >= levels.index(self.config.min_level)
        except ValueError:
            # Tanımsız level gelirse, güvenli tarafta kal ve gönder
            return True

    # -----------------------------
    # Ana public API
    # -----------------------------
    def send_alert(
        self,
        message: str,
        level: str = "INFO",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Temel alert gönderici.

        - Her zaman system_logger'a yazar.
        - Konfigürasyona göre Telegram'a da gönderir.
        """

        extra = extra or {}
        log_msg = f"[ALERT][{level}] {message} | extra={extra}"

        # Önce log'a bas
        if level == "CRITICAL":
            system_logger.critical(log_msg)
        elif level == "ERROR":
            system_logger.error(log_msg)
        elif level == "WARNING":
            system_logger.warning(log_msg)
        else:
            system_logger.info(log_msg)

        # Telegram devre dışıysa veya level düşükse, burada kes
        if not self.config.enable_telegram:
            return
        if not self._should_notify(level):
            return

        # Telegram notifier'ı çalıştır
        try:
            text = f"[{level}] {message}"
            if extra:
                text += f"\n\nDetails: {extra}"
            self.notifier.send_message(text)
        except Exception as e:
            system_logger.error(f"[AlertSystem] Telegram alert gönderilemedi: {e}")

    # -----------------------------
    # Kolaylık sağlayan helper metodlar
    # -----------------------------
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self.send_alert(message, level="INFO", extra=extra)

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self.send_alert(message, level="WARNING", extra=extra)

    def error(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self.send_alert(message, level="ERROR", extra=extra)

    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self.send_alert(message, level="CRITICAL", extra=extra)

