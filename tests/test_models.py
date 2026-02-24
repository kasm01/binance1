# tests/test_models.py
import unittest
import numpy as np

from models.lstm_model import LSTMModel
from lightgbm_model import LightGBMModel
from ensemble_model import EnsembleModel


class TestModels(unittest.TestCase):
    def setUp(self) -> None:
        # LSTM: (timesteps=5, features=3) gibi küçük bir input shape
        self.lstm = LSTMModel(input_shape=(5, 3))

        # LightGBM wrapper
        self.lgbm_wrapper = LightGBMModel()
        self.lgbm = self.lgbm_wrapper.model  # sklearn-style estimator

        # EnsembleModel: LightGBM'i kullanarak basit bir ensemble
        self.ensemble = EnsembleModel(
            estimators=[("lgbm", self.lgbm)]
        )

    def test_lstm_predict(self):
        # 4 örnek, 5 timestep, 3 feature
        X = np.random.rand(4, 5, 3)
        y = np.random.randint(0, 2, size=(4,))

        # Hızlı ve küçük bir eğitim
        self.lstm.fit(X, y, epochs=1, batch_size=2, validation_split=0.5)
        preds = self.lstm.predict(X)

        self.assertIsNotNone(preds)
        self.assertEqual(len(preds), 4)

    def test_lightgbm_predict(self):
        # 20 örnek, 4 feature
        X = np.random.rand(20, 4)
        y = np.random.randint(0, 2, size=(20,))

        self.lgbm_wrapper.fit(X, y)
        preds = self.lgbm_wrapper.predict(X)

        self.assertIsNotNone(preds)
        self.assertEqual(len(preds), 20)

    def test_ensemble_predict(self):
        # Ensemble için de basit bir fit
        X = np.random.rand(20, 4)
        y = np.random.randint(0, 2, size=(20,))

        self.ensemble.fit(X, y)
        preds = self.ensemble.predict(X)

        self.assertIsNotNone(preds)
        self.assertEqual(len(preds), 20)


if __name__ == "__main__":
    unittest.main()

