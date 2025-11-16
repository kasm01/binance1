# core/cache_manager.py
import redis

from core.utils import retry
from core.logger import error_logger
from config.redis_config import RedisConfig


class CacheManager:
    def __init__(self):
        self._redis_config = RedisConfig()
        self.redis_client = self._redis_config.client

    @retry(max_attempts=5, delay=1, exceptions=(redis.ConnectionError,))
    def set(self, key: str, value: str, ex: int = 60) -> None:
        if not self.redis_client:
            error_logger.warning("Redis client not available, skipping cache set.")
            return
        self.redis_client.set(key, value, ex=ex)

    @retry(max_attempts=5, delay=1, exceptions=(redis.ConnectionError,))
    def get(self, key: str):
        if not self.redis_client:
            error_logger.warning("Redis client not available, skipping cache get.")
            return None
        return self.redis_client.get(key)
