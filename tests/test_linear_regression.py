"""Tests for LinearRegressionScratch."""

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression as SklearnLR

from mlpkg.supervised import LinearRegressionScratch


def test_perfect_line():
    """A perfectly linear y = 2x + 1 should be recovered exactly."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])  # y = 2x + 1
    model = LinearRegressionScratch().fit(X, y)
    assert np.isclose(model.coef_[0], 2.0, atol=1e-6)
    assert np.isclose(model.intercept_, 1.0, atol=1e-6)


def test_predict_shape():
    X, y = make_regression(n_samples=50, n_features=3, random_state=0)
    model = LinearRegressionScratch().fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (50,)


def test_matches_sklearn():
    """Our coefficients should match sklearn's to within numerical tolerance."""
    X, y = make_regression(n_samples=200, n_features=5, noise=0.5, random_state=42)
    ours = LinearRegressionScratch().fit(X, y)
    theirs = SklearnLR().fit(X, y)
    np.testing.assert_allclose(ours.coef_, theirs.coef_, atol=1e-6)
    assert np.isclose(ours.intercept_, theirs.intercept_, atol=1e-6)


def test_score_on_perfect_fit():
    """R^2 should be ~1.0 on noise-free data."""
    X, y = make_regression(n_samples=100, n_features=4, noise=0, random_state=1)
    model = LinearRegressionScratch().fit(X, y)
    assert model.score(X, y) > 0.9999


def test_predict_before_fit_raises():
    model = LinearRegressionScratch()
    with pytest.raises(RuntimeError):
        model.predict(np.array([[1.0]]))


def test_mismatched_shapes_raise():
    model = LinearRegressionScratch()
    X = np.array([[1.0], [2.0]])
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        model.fit(X, y)
