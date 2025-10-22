from catboost import CatBoostClassifier

class CatBoostModel:
    def __init__(self, params=None):
        default_params = {
            'iterations': 500,
            'learning_rate': 0.05,
            'depth': 6,
            'verbose': False
        }
        self.params = params or default_params
        self.model = CatBoostClassifier(**self.params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:,1]
