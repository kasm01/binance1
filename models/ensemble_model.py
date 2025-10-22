from sklearn.ensemble import VotingClassifier

class EnsembleModel:
    def __init__(self, models):
        """
        models: [('lgbm', lgbm_model), ('catboost', cat_model), ('lstm', lstm_wrapper)]
        """
        self.models = models
        self.ensemble = VotingClassifier(estimators=models, voting='soft')

    def fit(self, X, y):
        self.ensemble.fit(X, y)

    def predict(self, X):
        return self.ensemble.predict(X)

    def predict_proba(self, X):
        return self.ensemble.predict_proba(X)
