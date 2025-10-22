import unittest
from core.cache_manager import CacheManager

class TestRedis(unittest.TestCase):
    def setUp(self):
        self.cache = CacheManager()

    def test_set_get(self):
        self.cache.set("test_key", "test_value")
        value = self.cache.get("test_key")
        self.assertEqual(value, "test_value")

if __name__ == "__main__":
    unittest.main()
