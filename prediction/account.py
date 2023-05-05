import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor
from .utils import rolling_window
from tqdm import tqdm

def process_data(data, window, min_value, normalize, extra_features):
    fit_data = rolling_window(data, window).reshape((-1,window))
    X_mean = np.average(fit_data[:,:-1], axis=1, keepdims=True)
    X_cond = X_mean[:,0]>=min_value
    fit_data = fit_data[X_cond,:]
    fit_X = fit_data[:,:-1]
    X_mean = X_mean[X_cond,:]
    X_std = np.std(fit_X, axis=1, keepdims=True)
    fit_y = fit_data[:,-1:]
    if normalize:
        fit_X = (fit_X - X_mean)/X_std
        fit_y = (fit_y - X_mean)/X_std
    if extra_features:
        fit_X = np.concatenate((fit_X, X_mean, X_std), axis=1)
    return fit_X, fit_y

class AverageModel:
    def predict(self, X, **kwargs):
        return np.average(X, axis=1)

class MLPModel:
    def __init__(self, max_memory=1<<35, base_model=None, window=5, min_value=0., normalize=False, extra_features=False, n_iters=200, early_stop=True, validation_rate=0.1, tol=1e-4, tol_iters=10, shuffle=False, val_on_time=False):
        self.model = MLPRegressor(learning_rate='adaptive', learning_rate_init=0.01)
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
        self.shuffle = shuffle
        self.val_on_time = val_on_time

    def fit(self, data, window=None, min_value=None, normalize=None, extra_features=None, shuffle=None):
        if window is None: window = self.window
        if min_value is None: min_value = self.min_value
        if normalize is None: normalize = self.normalize
        if extra_features is None: extra_features = self.extra_features
        if shuffle is None: shuffle = self.shuffle
        # data : weight data with shape (n_accounts, n_epochs)
        print('Fit data:', data.shape)
        if shuffle:
            data = data[np.random.permutation(data.shape[0]),:]
        if self.early_stop:
            if self.val_on_time:
                val_size = max(window, int(data.shape[1]*self.validation_rate))
                val_X, val_y = process_data(data[:,-val_size:], window=window, min_value=min_value, normalize=normalize, extra_features=extra_features)
                data = data[:,:-val_size]
            else:
                val_size = max(1, int(data.shape[0]*self.validation_rate))
                val_X, val_y = process_data(data[-val_size:,:], window=window, min_value=min_value, normalize=normalize, extra_features=extra_features)
                data = data[:-val_size,:]
            print('Val data:', val_size, val_X.shape, val_y.shape)
        sample_size = data.shape[0]*window*8
        step_size = min(data.shape[1], max(self.max_memory // sample_size, window))
        print('Partial fit with step:', step_size, data.shape)
        self.loss_curves = []
        self.score_curves = []
        partial_fit_X = partial_fit_y = None
        best_score = -np.inf
        for step in range(0, data.shape[1]-step_size+1, step_size-window+1):
            fit_X, fit_y = process_data(data[:,step:step+step_size], window=window, min_value=min_value, normalize=normalize, extra_features=extra_features)
            partial_fit_X = fit_X if partial_fit_X is None else np.concatenate((partial_fit_X, fit_X), axis=0)
            partial_fit_y = fit_y if partial_fit_y is None else np.concatenate((partial_fit_y, fit_y), axis=0)
            if np.prod(partial_fit_X.shape)*8+np.prod(fit_X.shape)*8<self.max_memory and step+2*step_size-window<data.shape[1]:
                continue
            n_iter_no_change = 0
            loss_curve = []
            score_curve = []
            for iter in tqdm(range(self.n_iters), desc=f'Step {step}:{partial_fit_X.shape}'):
                self.model.partial_fit(partial_fit_X, partial_fit_y[:,0])
                loss_curve.append(self.model.loss_)
                if self.early_stop:
                    fit_score = self.model.score(val_X, val_y[:,0])
                else:
                    fit_score = self.model.score(partial_fit_X, partial_fit_y[:,0])
                score_curve.append(fit_score)
                n_iter_no_change += 1
                if fit_score - best_score>self.tol:
                    n_iter_no_change = 0
                    best_score = fit_score
                if n_iter_no_change>self.tol_iters:
                    break
            partial_fit_X = None
            partial_fit_y = None
            self.loss_curves.append(loss_curve)
            self.score_curves.append(score_curve)
        return best_score
    def predict(self, X, min_value=None, normalize=None, extra_features=None, base_model=None):
        if min_value is None: min_value = self.min_value
        if normalize is None: normalize = self.normalize
        if extra_features is None: extra_features = self.extra_features
        if base_model is None: base_model = self.base_model
        X_mean = np.average(X, axis=1, keepdims=True)
        X_cond = X_mean[:,0]>=min_value
        pred = np.zeros(shape=X.shape[0], dtype=X.dtype)
        hot_X = X[X_cond,:]
        rest_X = X[~X_cond,:]
        hot_X_mean = X_mean[X_cond,:]
        hot_X_std = np.maximum(1e-4,np.std(hot_X, axis=0, keepdims=True))
        if normalize:
            hot_X = (hot_X - hot_X_mean)/hot_X_std
        if extra_features:
            hot_X = np.concatenate((hot_X, hot_X_mean, hot_X_std), axis=1)
        model_pred = self.model.predict(hot_X)
        if normalize:
            model_pred = model_pred*hot_X_std[:,0]+hot_X_mean[:,0]
        pred[X_cond] = model_pred
        if len(rest_X)>0:
            base_pred = base_model.predict(rest_X)
            pred[~X_cond] = base_pred
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
