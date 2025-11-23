# tests/test_redis.py
import unittest
import redis

from core.cache_manager import CacheManager


class TestRedis(unittest.TestCase):
    def setUp(self) -> None:
        self.cache = CacheManager()

    def test_set_get(self):
        """
        Redis erişilemezse (örn. test ortamında Redis yoksa),
        testi graceful şekilde atlarız.
        """
        try:
            self.cache.set("test_key", "test_value")
            value = self.cache.get("test_key")
            self.assertEqual(value, "test_value")
        except (redis.ConnectionError, redis.TimeoutError):
            self.skipTest("Redis not available in this environment")


if __name__ == "__main__":
    unittest.main()

