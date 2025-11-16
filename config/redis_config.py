# config/redis_config.py
import redis

from .settings import Settings
from .credentials import Credentials


class RedisConfig:
    """
    Redis bağlantı yöneticisi.
    Market verisi, sinyal ve model tahmin cache için kullanılacak.
    """

    def __init__(self):
        self.host = Settings.REDIS_HOST
        self.port = Settings.REDIS_PORT
        self.db = Settings.REDIS_DB
        self.password = Credentials.REDIS_PASSWORD
        self._client: redis.Redis | None = None

    def connect(self) -> redis.Redis | None:
        if self._client is not None:
            return self._client

        try:
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
            )
            self._client.ping()
            print("[INFO] Redis connection established successfully.")
        except Exception as e:
            print(f"[ERROR] Redis connection failed: {e}")
            self._client = None

        return self._client

    @property
    def client(self) -> redis.Redis | None:
        if self._client is None:
            return self.connect()
        return self._client

