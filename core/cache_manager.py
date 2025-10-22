import redis
from core.utils import retry
from config.settings import Settings

class CacheManager:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=Settings.REDIS_HOST,
            port=Settings.REDIS_PORT,
            password=Settings.REDIS_PASSWORD,
            decode_responses=True
        )

    @retry(max_attempts=5, delay=1, exceptions=(redis.ConnectionError,))
    def set(self, key, value, ex=60):
        """Cache'e veri ekle."""
        self.redis_client.set(key, value, ex=ex)

    @retry(max_attempts=5, delay=1, exceptions=(redis.ConnectionError,))
    def get(self, key):
        """Cache'den veri al."""
        return self.redis_client.get(key)
