import numpy as np
from sklearn.linear_model import LinearRegression

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

class LinearModel:
    def __init__(self):
        self.model = LinearRegression(fit_intercept=False, positive=True)
    def fit(self, data, window=5):
        data = rolling_window(data, window).reshape((-1,window)) # rolling window
        data = data[~np.all(data==0, axis=0)] # remove zero samples
        self.model.fit(data[:,:-1], data[:,-1])
    def predict(self, x):
        return self.model.predict(x)

class ARIMAModel:
    pass
