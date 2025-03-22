from typing import Callable

import numpy as np
from scipy.optimize import minimize


def linear_model(x: np.ndarray, params: tuple | list[float]) -> np.ndarray:
    """a basic linear model"""
    a, b = params
    yhat = a * x + b
    return yhat


def logistic(x: np.ndarray, k, x0, L=1.0) -> np.ndarray:
    """
    Parameters:
    x: Independent variable
    k: Growth rate
    x0: Midpoint (inflection point)
    L: Upper limit (default 1)
    """
    return L / (1 + np.exp(-k * (x - x0)))


def mse(y: np.ndarray, yhat: np.ndarray):
    """mean squared error loss function"""
    squared_diff = (y - yhat) ** 2
    return np.mean(squared_diff)


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    loss_fn: Callable,
    params: list[float],
    bounds=None,
) -> list[float]:
    def objective(params):
        yhat = model_fn(X, params)
        return loss_fn(y, yhat)

    result = minimize(fun=objective, x0=params, bounds=bounds)
    return result.x

