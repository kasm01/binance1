from models.ensemble_model import EnsembleModel

class StrategyEngine:
    def __init__(self):
        self.model = EnsembleModel()

    def generate_signal(self, features):
        signal = self.model.predict(features)
        return signal
