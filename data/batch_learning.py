from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class BatchLearning:
    def __init__(self, features, target='label'):
        self.features = features
        self.target = target
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, df):
        X = df[self.features]
        y = df[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return self.model, acc
