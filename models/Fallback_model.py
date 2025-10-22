class FallbackModel:
    def __init__(self, primary_model, secondary_model):
        self.primary_model = primary_model
        self.secondary_model = secondary_model

    def predict(self, X):
        try:
            return self.primary_model.predict(X)
        except Exception:
            return self.secondary_model.predict(X)

    def predict_proba(self, X):
        try:
            return self.primary_model.predict_proba(X)
        except Exception:
            return self.secondary_model.predict_proba(X)
