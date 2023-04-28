import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor
from .utils import rolling_window
from tqdm import tqdm

class AverageModel:
    def predict(self, X, **kwargs):
        return np.average(X, axis=1)

class MLPModel:
    def __init__(self, max_memory=1<<35, base_model=None, window=5, min_value=0., normalize=False, extra_features=False, n_iters=200, early_stop=True, validation_rate=0.1, tol=1e-4, tol_iters=10):
        self.model = MLPRegressor()
        self.max_memory = max_memory
        self.base_model = base_model
        self.n_iters = n_iters
        self.early_stop = early_stop
        self.validation_rate = validation_rate
        self.tol = tol
        self.tol_iters = tol_iters
        self.window = window
        self.min_value = min_value
        self.normalize = normalize
        self.extra_features = extra_features

    def _process_data(self, data, window, min_value, normalize, extra_features):
        fit_data = rolling_window(data, window).reshape((-1,window))
        X_mean = np.average(fit_data[:,:-1], axis=1, keepdims=True)
        X_cond = X_mean[:,0]>=min_value
        fit_data = fit_data[X_cond,:]
        fit_X = fit_data[:,:-1]
        X_mean = X_mean[X_cond,:]
        X_std = np.std(fit_X, axis=1, keepdims=True)
        fit_y = fit_data[:,-1]
        if normalize:
            fit_X = (fit_X - X_mean)/X_std
            fit_y = (fit_y - X_mean)/X_std
        if extra_features:
            fit_X = np.concatenate((fit_X, X_mean, X_std), axis=1)
        return fit_X, fit_y

    def fit(self, data, window=None, min_value=None, normalize=False, extra_features=False):
        if window is None: window = self.window
        if min_value is None: min_value = self.min_value
        if normalize is None: normalize = self.normalize
        if extra_features is None: extra_features = self.extra_features
        # data : weight data with shape (n_accounts, n_epochs)
        print('Fit data:', data.shape)
        if self.early_stop:
            val_size = int(data.shape[0]*self.validation_rate)
            val_X, val_y = self._process_data(data[:,-val_size:], window=window, min_value=min_value, normalize=normalize, extra_features=extra_features)
            print('Val data:', val_X.shape, val_y.shape)
            data = data[:,:-val_size]
        sample_size = data.shape[0]*window*8
        step_size = max(self.max_memory // sample_size, window)
        if step_size>data.shape[1]:
            step_size = data.shape[1]
        print('Partial fit with step:', step_size, data.shape)
        self.loss_curves = []
        partial_fit_X = None
        for step in range(0, data.shape[1]-step_size+1, step_size-window+1):
            fit_X, fit_y = self._process_data(data[:,step:step+step_size], window=window, min_value=min_value, normalize=normalize, extra_features=extra_features)
            if partial_fit_X is None:
                partial_fit_X = fit_X
                partial_fit_y = fit_y
            else:
                partial_fit_X = np.concatenate((partial_fit_X, fit_X), axis=0)
                partial_fit_y = np.concatenate((partial_fit_y, fit_y), axis=0)
            if np.prod(partial_fit_X.shape)*8+np.prod(fit_X.shape)*8<self.max_memory and step+2*step_size-window<data.shape[1]:
                continue
            best_score = -np.inf
            n_iter_no_change = 0
            loss_curve = []
            for iter in tqdm(range(self.n_iters), desc=f'Step {step}'):
                self.model.partial_fit(partial_fit_X, partial_fit_y)
                loss_curve.append(self.model.loss_)
                if self.early_stop:
                    fit_score = self.model.score(val_X, val_y)
                else:
                    fit_score = self.model.score(partial_fit_X, partial_fit_y)
                n_iter_no_change += 1
                if fit_score - best_score>self.tol:
                    n_iter_no_change = 0
                    best_score = fit_score
                if n_iter_no_change>self.tol_iters:
                    break
            partial_fit_X = None
            partial_fit_y = None
            self.loss_curves.append(loss_curve)
        return best_score
    def predict(self, X, min_value=None, normalize=None, extra_features=None, base_model=None):
        if min_value is None: min_value = self.min_value
        if normalize is None: normalize = self.normalize
        if extra_features is None: extra_features = self.extra_features
        if base_model is None: base_model = self.base_model
        X_mean = np.average(X, axis=1, keepdims=True)
        X_cond = X_mean[:,0]>=min_value
        X_std = np.std(X, axis=1, keepdims=True)
        if normalize:
            X = (X - X_mean)/X_std
        if extra_features:
            X = np.concatenate((X, X_mean, X_std), axis=1)
        pred = np.zeros(shape=X.shape[0], dtype=X.dtype)
        hot_X = X[X_cond,:]
        rest_X = X[~X_cond,:]
        model_pred = self.model.predict(hot_X)[:,0]
        pred[X_cond] = model_pred
        if len(rest_X)>0:
            base_pred = base_model.predict(rest_X)
            pred[~X_cond] = base_pred
        if normalize:
            pred = pred*X_std+X_mean
        return pred
    def score(self, X, y):
        return self.model.score(X, y)

class LRModel:
    def __init__(self, max_memory=1<<35):
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
            self.model.fit(fit_X, fit_y)
            score = self.model.score(fit_X, fit_y)
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
