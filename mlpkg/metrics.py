"""Evaluation metrics for regression and classification."""

import numpy as np


def mean_squared_error(y_true, y_pred):
    """Mean squared error: mean((y_true - y_pred)^2)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true, y_pred):
    """Root mean squared error: sqrt(MSE)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(y_true, y_pred):
    """
    R^2 (coefficient of determination).

    R^2 = 1 - SS_res / SS_tot
        where SS_res = sum((y_true - y_pred)^2)
              SS_tot = sum((y_true - mean(y_true))^2)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        # Degenerate case: y_true is constant
        return 0.0 if ss_res > 0 else 1.0
    return float(1.0 - ss_res / ss_tot)


def accuracy_score(y_true, y_pred):
    """Mean classification accuracy."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    return float(np.mean(y_true == y_pred))
