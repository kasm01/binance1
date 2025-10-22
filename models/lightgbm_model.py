import lightgbm as lgb

class LightGBMModel:
    def __init__(self, params=None):
        default_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'n_estimators': 500,
            'verbose': -1
        }
        self.params = params or default_params
        self.model = lgb.LGBMClassifier(**self.params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:,1]
