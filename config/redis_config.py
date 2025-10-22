import redis
from .settings import Settings
from .credentials import Credentials

class RedisConfig:
    """
    Redis connection handler for caching market data and model predictions.
    """

    def __init__(self):
        self.host = Settings.REDIS_HOST
        self.port = Settings.REDIS_PORT
        self.db = Settings.REDIS_DB
        self.password = Credentials.REDIS_PASSWORD
        self.client = None

    def connect(self):
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True
            )
            self.client.ping()
            print("[INFO] Redis connection established successfully.")
        except Exception as e:
            print(f"[ERROR] Redis connection failed: {e}")

    def get_client(self):
        if not self.client:
            self.connect()
        return self.client
