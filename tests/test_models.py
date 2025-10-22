import unittest
from models.lstm_model import LSTMModel
from models.lightgbm_model import LightGBMModel
from models.ensemble_model import EnsembleModel

class TestModels(unittest.TestCase):
    def setUp(self):
        self.lstm = LSTMModel()
        self.lgbm = LightGBMModel()
        self.ensemble = EnsembleModel([self.lstm, self.lgbm])

    def test_lstm_predict(self):
        data = [[0.1, 0.2, 0.3]]  # örnek veri
        result = self.lstm.predict(data)
        self.assertIsNotNone(result)

    def test_lgbm_predict(self):
        data = [[0.1, 0.2, 0.3]]
        result = self.lgbm.predict(data)
        self.assertIsNotNone(result)

    def test_ensemble_predict(self):
        data = [[0.1, 0.2, 0.3]]
        result = self.ensemble.predict(data)
        self.assertIsNotNone(result)

if __name__ == "__main__":
    unittest.main()
