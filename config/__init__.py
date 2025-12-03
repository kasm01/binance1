"""
Config package initializer.
This file exposes settings.py globals and optional credential / redis helpers.
"""

# Ortak ayarları (SYMBOL, INTERVAL, API ayarları vs.) dışarı aç
from .settings import *

# İhtiyaç olursa kullanılabilen yardımcı sınıflar
from .credentials import Credentials
from .redis_config import RedisConfig

__all__ = [*dir()]  # settings.py içindeki bütün global ayarları export eder

