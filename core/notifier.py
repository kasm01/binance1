from core.logger import system_logger, error_logger

class Notifier:
    def __init__(self, telegram_bot=None):
        self.telegram_bot = telegram_bot

    def notify_system(self, message):
        system_logger.info(message)
        if self.telegram_bot:
            self.telegram_bot.send_message(message)

    def notify_error(self, message):
        error_logger.error(message)
        if self.telegram_bot:
            self.telegram_bot.send_message(f"ERROR: {message}")
