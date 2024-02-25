from typing import Optional

import numpy as np


def mape_single(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def rmse_single(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae_single(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmsse_single(
    y_true: np.ndarray, y_pred: np.ndarray, y_hist: np.ndarray
) -> float | np.ndarray:
    n = y_hist.shape[0]
    h = y_true.shape[0]
    numerator = 1 / h * np.mean((y_true - y_pred) ** 2, axis=-1)
    denominator = 1 / (n - 1) * np.sum((y_hist[1:] - y_hist[:-1]) ** 2, axis=-1)
    return np.sqrt(numerator / denominator)


def wrmsse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_hist: np.ndarray,
    weights: Optional[np.ndarray] = None,
    last: int = 28,
) -> float:
    if weights is None:
        last_vals = y_hist[-last:]  # N x T
        weights = np.sum(last_vals, axis=-1)  # N
        weights = weights / np.sum(weights)  # N
    return weights * rmsse_single(y_true, y_pred, y_hist)
