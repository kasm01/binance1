from sklearn.model_selection import GridSearchCV

class HyperparameterTuner:
    def __init__(self, model, param_grid, cv=3, scoring='accuracy'):
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.best_model = None

    def tune(self, X, y):
        grid = GridSearchCV(self.model, self.param_grid, cv=self.cv, scoring=self.scoring, n_jobs=-1)
        grid.fit(X, y)
        self.best_model = grid.best_estimator_
        return self.best_model, grid.best_params_
