import math
import numpy as np
import numpy.linalg as la
from sklearn.metrics import mean_squared_error,mean_absolute_error
import math

def SMAPE(actual, predicted):
    """
    Calculates the SMAPE metric
    Parameters:
        actual - target values
        predicted - output values predicted by model
    Returns:
        smape - returns smape metric
    """

    return np.mean(abs(predicted - actual) / ((abs(predicted) + abs(actual)) / 2)) * 100

def MSE(target, pred):
    """
    Calculates the MSE metric
    Parameters:
        actual - target values
        predicted - output values predicted by model
    Returns:
        mse - returns MSE metric
    """

    return mean_squared_error(target, pred, squared=True)

def RMSE(target, pred):
    """
    Calculates the RMSE metric
    Parameters:
        actual - target values
        predicted - output values predicted by model
    Returns:
        root - returns RMSE metric
    """

    root = math.sqrt(mean_squared_error(target, pred))
    return root

def MAE(target, pred):
    """
    Calculates the MAE metric
    Parameters:
        actual - target values
        predicted - output values predicted by model
    Returns:
        mae - returns MAE metric
    """
    return mean_absolute_error(target, pred)


# Old metrics file used
# def metrics(a, b):
#     """
#     Calculate evaluation metrics for comparing actual values 'a' and predicted values 'b'.
#     a: actual values
#     b: predicted values
#     """
#     # RMSE and MAE calculations
#     rmse = math.sqrt(mean_squared_error(a, b))
#     mae = mean_absolute_error(a, b)
    
#     # MAPE calculation
#     mape = np.zeros_like(a)
#     idx = a != 0
#     mape[idx] = np.abs((a[idx] - b[idx]) / a[idx]) * 100
    
#     # SMAPE calculation
#     smape = np.zeros_like(a)
#     idx = (a != 0) & (b != 0)
#     smape[idx] = np.abs(a[idx] - b[idx]) / ((np.abs(a[idx]) + np.abs(b[idx])) / 2) * 100
    
#     F_norm = la.norm(a - b, 'fro') / la.norm(a, 'fro')
#     r2 = 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()
#     var = 1 - (np.var(a - b)) / np.var(a)
    
#     return rmse, mae, np.mean(mape), np.mean(smape), 1 - F_norm, r2, var

### Perturbation Analysis
def MaxMinNormalization(x,Max,Min):
    x = (x-Min)/(Max-Min)
    return x