from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestRegressor
from pykrige.rk import RegressionKriging
from sklearn import svm
from sklearn.linear_model import LinearRegression
import numpy as np
from tqdm import tqdm


def gaussian_process_inference(X_train, y_train, X_test):
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=10)
    gpr.fit(X_train, y_train)
    
    batch_size = 10000
    batch_num = int(np.ceil(len(X_test) / batch_size))
    y_pred = np.zeros(len(X_test))
    for i in tqdm(range(batch_num), desc='Gaussian Process', mininterval=8):
        if i < batch_num - 1:
            X_batch = X_test[(i * batch_size):((i + 1) * batch_size)]
            y_pred[(i * batch_size):((i + 1) * batch_size)] = gpr.predict(X_batch)
        elif i == batch_num - 1:
            X_batch = X_test[(i * batch_size):]
            y_pred[(i * batch_size):] = gpr.predict(X_batch)
        
    return y_pred


def regression_kriging(X_train, y_train, X_test):
    x_train = X_train[:, 0:2]
    p_train = X_train[:, 2:3]
    x_test = X_test[:, 0:2]
    p_test = X_test[:, 2:3]
    target_train = y_train
    
    rf_model = RandomForestRegressor(n_estimators=100)
    lr_model = LinearRegression(copy_X=True, fit_intercept=False)
    rk = RegressionKriging(regression_model=lr_model, n_closest_points=10)
    rk.fit(p_train, x_train, target_train)
    test_pre = rk.predict(p_test, x_test)
    
    return test_pre


