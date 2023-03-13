import numpy as np
from sklearn.linear_model import LinearRegression

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

class AverageModel:
    def predict(self, X):
        return np.average(X, axis=1)

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
