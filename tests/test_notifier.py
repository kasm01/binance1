import unittest
from core.notifier import Notifier

class TestNotifier(unittest.TestCase):
    def setUp(self):
        self.notifier = Notifier()

    def test_send_message(self):
        result = self.notifier.send_message("Test message")
        self.assertTrue(result)

if __name__ == "__main__":
    unittest.main()
