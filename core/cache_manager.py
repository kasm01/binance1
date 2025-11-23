import redis

from core.utils import retry
from config.settings import Settings
from config.credentials import Credentials


class CacheManager:
    """
    Basit Redis cache yöneticisi.

    - Kısa süreli veriler için TTL ile set/get
    - Redis bağlantı sorunlarında retry devreye girer
    """

    def __init__(self) -> None:
        self.redis_client = redis.Redis(
            host=Settings.REDIS_HOST,
            port=Settings.REDIS_PORT,
            db=Settings.REDIS_DB,
            password=Credentials.REDIS_PASSWORD,
            decode_responses=True,
        )

    @retry(max_attempts=5, delay=1, exceptions=(redis.ConnectionError,))
    def set(self, key: str, value: str, ex: int = 60) -> None:
        """Cache'e veri ekle (TTL: ex saniye)."""
        self.redis_client.set(key, value, ex=ex)

    @retry(max_attempts=5, delay=1, exceptions=(redis.ConnectionError,))
    def get(self, key: str) -> str | None:
        """Cache'den veri al. Bulunamazsa None döner."""
        return self.redis_client.get(key)
