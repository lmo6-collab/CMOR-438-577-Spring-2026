"""Tests for preprocessing utilities."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler as SklearnScaler

from mlpkg.preprocessing import train_test_split, StandardScaler


def test_split_sizes():
    X = np.arange(100).reshape(-1, 1)
    y = np.arange(100)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    assert X_tr.shape[0] == 80
    assert X_te.shape[0] == 20
    assert y_tr.shape[0] == 80
    assert y_te.shape[0] == 20


def test_split_reproducible():
    X = np.arange(50).reshape(-1, 1)
    y = np.arange(50)
    a = train_test_split(X, y, test_size=0.3, random_state=42)
    b = train_test_split(X, y, test_size=0.3, random_state=42)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[2], b[2])


def test_split_no_overlap():
    X = np.arange(40).reshape(-1, 1)
    y = np.arange(40)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=7)
    train_set = set(X_tr.ravel().tolist())
    test_set = set(X_te.ravel().tolist())
    assert train_set.isdisjoint(test_set)
    assert len(train_set | test_set) == 40


def test_invalid_test_size():
    X = np.array([[1.0]])
    y = np.array([1.0])
    with pytest.raises(ValueError):
        train_test_split(X, y, test_size=1.5)


def test_scaler_zero_mean_unit_var():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=3.0, size=(200, 4))
    Xs = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(Xs.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(Xs.std(axis=0), 1.0, atol=1e-10)


def test_scaler_matches_sklearn():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 5))
    ours = StandardScaler().fit_transform(X)
    theirs = SklearnScaler().fit_transform(X)
    np.testing.assert_allclose(ours, theirs, atol=1e-10)


def test_scaler_handles_constant_column():
    """A constant column should be passed through without producing NaN."""
    X = np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]])
    Xs = StandardScaler().fit_transform(X)
    assert not np.any(np.isnan(Xs))
