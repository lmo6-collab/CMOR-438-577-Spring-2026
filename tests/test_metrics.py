"""Tests for the metrics module."""

import numpy as np
import pytest
from sklearn.metrics import (
    mean_squared_error as sk_mse,
    r2_score as sk_r2,
    accuracy_score as sk_acc,
)

from mlpkg.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
)


def test_mse_zero_on_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert mean_squared_error(y, y) == 0.0


def test_mse_matches_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=100)
    y_pred = rng.normal(size=100)
    assert np.isclose(mean_squared_error(y_true, y_pred), sk_mse(y_true, y_pred))


def test_rmse_is_sqrt_mse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    assert np.isclose(
        root_mean_squared_error(y_true, y_pred),
        np.sqrt(mean_squared_error(y_true, y_pred)),
    )


def test_r2_perfect_is_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_score(y, y) == 1.0


def test_r2_matches_sklearn():
    rng = np.random.default_rng(1)
    y_true = rng.normal(size=200)
    y_pred = y_true + rng.normal(scale=0.1, size=200)
    assert np.isclose(r2_score(y_true, y_pred), sk_r2(y_true, y_pred))


def test_accuracy_basic():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    assert accuracy_score(y_true, y_pred) == 0.75


def test_accuracy_matches_sklearn():
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 3, size=100)
    y_pred = rng.integers(0, 3, size=100)
    assert np.isclose(accuracy_score(y_true, y_pred), sk_acc(y_true, y_pred))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mean_squared_error(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
