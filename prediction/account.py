import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor
from utils import rolling_window
from tqdm import tqdm

class AverageModel:
    def predict(self, X):
        return np.average(X, axis=1)

class MLPModel:
    def __init__(self, max_memory=1<<36):
        self.model = MLPRegressor()
        self.max_memory = max_memory
    def fit(self, data, window=5):
        sample_size = data.shape[0]*window*8
        step_size = max(self.max_memory // sample_size, window)
        print('Partial fit with step:', step_size, data.shape)
        for step in tqdm(range(0, data.shape[1]-step_size, step_size-window+1)):
            fit_data = rolling_window(data[:,step:step+step_size], window).reshape((-1,window))
            fit_data = fit_data[~np.all(fit_data==0, axis=1)]
            fit_X = fit_data[:,:-1]
            fit_y = fit_data[:,-1]
            self.model.partial_fit(fit_X, fit_y)
            score = self.score(fit_X, fit_y)
        return score
    def predict(self, X):
        return self.model.predict(X)
    def score(self, X, y):
        return self.model.score(X, y)

class LRModel:
    def __init__(self, max_memory=1<<36):
        self.model = LogisticRegression(fit_intercept=False)
        self.max_memory = max_memory
    def fit(self, data, window=5):
        sample_size = data.shape[0]*window*8
        step_size = max(self.max_memory // sample_size, window)
        print('Partial fit with step:', step_size, data.shape)
        for step in tqdm(range(0, data.shape[1], step_size-window+1)):
            fit_data = rolling_window(data[:,step:step+step_size], window).reshape((-1,window))
            fit_data = fit_data[~np.all(fit_data==0, axis=1)]
            fit_X = fit_data[:,:-1]
            fit_y = fit_data[:,-1]
            self.model.fit(self.fit_X, self.fit_y)
            score = self.model.score(self.fit_X, self.fit_y)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        return score

class LinearModel:
    def __init__(self):
        self.model = LinearRegression(fit_intercept=False, positive=True)
    def fit(self, data, window=5):
        data = rolling_window(data, window).reshape((-1,window)) # rolling window
        data = data[~np.all(data==0, axis=1)] # remove zero samples
        self.fit_X = data[:,:-1]
        self.fit_y = data[:,-1]
        self.model.fit(self.fit_X, self.fit_y)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        return self.score(self.fit_X, self.fit_y)
    def predict(self, X):
        return self.model.predict(X)
    def score(self, X, y):
        return self.model.score(X, y)

class ARIMAModel:
    pass
