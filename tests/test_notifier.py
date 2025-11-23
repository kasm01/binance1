# tests/test_notifier.py
import unittest

from core.notifier import Notifier


class FakeTelegramBot:
    """
    Notifier için basit fake Telegram bot.
    Sadece send_message çağrısını sayıyoruz.
    """

    def __init__(self):
        self.messages = []

    def send_message(self, text: str):
        self.messages.append(text)


class TestNotifier(unittest.TestCase):
    def test_notify_system_without_telegram(self):
        """
        Telegram bot verilmediğinde bile notify_system
        hata fırlatmamalı.
        """
        notifier = Notifier(telegram_bot=None)
        # Hata fırlatmıyorsa test geçer
        notifier.notify_system("System test message")

    def test_notify_error_without_telegram(self):
        notifier = Notifier(telegram_bot=None)
        notifier.notify_error("Error test message")

    def test_notify_with_fake_telegram_bot(self):
        fake_bot = FakeTelegramBot()
        notifier = Notifier(telegram_bot=fake_bot)

        notifier.notify_system("Hello system")
        notifier.notify_error("Hello error")

        self.assertGreaterEqual(len(fake_bot.messages), 2)
        self.assertTrue(any("ERROR" in msg or "error" in msg.lower() for msg in fake_bot.messages))


if __name__ == "__main__":
    unittest.main()

