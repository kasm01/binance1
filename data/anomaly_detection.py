import numpy as np
from sklearn.ensemble import IsolationForest

class AnomalyDetection:
    def __init__(self, contamination=0.01):
        self.model = IsolationForest(contamination=contamination, random_state=42)

    def fit_predict(self, df, features):
        """
        Verilen özellikler üzerinde anomali tespiti yapar.
        """
        X = df[features].values
        df['anomaly'] = self.model.fit_predict(X)
        df['anomaly'] = df['anomaly'].apply(lambda x: 1 if x == -1 else 0)
        return df
