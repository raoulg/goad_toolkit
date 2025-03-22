import numpy as np
from scipy.optimize import minimize
from typing import Callable

def linear_model(x: np.ndarray, params: tuple) -> np.ndarray:
    """a basic linear model"""
    a, b = params
    yhat = a * x + b
    return yhat

def mse(y: np.ndarray, yhat: np.ndarray) -> float:
    """mean squared error loss function"""
    squared_diff = (y - yhat) ** 2
    return np.mean(squared_diff)

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    loss_fn: Callable,
    params: list[float],
    bounds=None
) -> list[float]:
    def objective(params):
        yhat = model_fn(X, params)
        return loss_fn(y, yhat)

    result = minimize(fun=objective, x0=params, bounds=bounds)
    return result.x