from telegram import Update, Bot
from telegram.ext import Updater, CommandHandler, CallbackContext
from .commands import start_command, status_command, trades_command
from config.credentials import TELEGRAM_TOKEN

class TelegramBot:
    def __init__(self):
        self.bot = Bot(token=TELEGRAM_TOKEN)
        self.updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
        self.dispatcher = self.updater.dispatcher
        self.register_commands()

    def register_commands(self):
        self.dispatcher.add_handler(CommandHandler('start', start_command))
        self.dispatcher.add_handler(CommandHandler('status', status_command))
        self.dispatcher.add_handler(CommandHandler('trades', trades_command))

    def start(self):
        self.updater.start_polling()
        self.updater.idle()
