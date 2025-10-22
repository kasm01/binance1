from sklearn.linear_model import SGDClassifier
import pandas as pd

class OnlineLearning:
    def __init__(self, features, target='label'):
        self.features = features
        self.target = target
        self.model = SGDClassifier(loss='log', max_iter=1000, tol=1e-3)

    def partial_fit(self, df):
        """
        Gelen yeni veri ile modeli kısmi olarak günceller.
        """
        X = df[self.features].values
        y = df[self.target].values
        self.model.partial_fit(X, y, classes=[0,1])
        return self.model
