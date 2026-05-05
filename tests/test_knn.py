"""Tests for KNNClassifierScratch."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN

from mlpkg.supervised import KNNClassifierScratch
from mlpkg.preprocessing import train_test_split


def test_easy_two_clusters():
    """Two well-separated clusters: should achieve perfect accuracy."""
    X = np.array([
        [0.0, 0.0], [0.1, 0.1], [0.0, 0.1],   # class 0
        [10.0, 10.0], [10.1, 10.0], [10.0, 10.1],  # class 1
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = KNNClassifierScratch(k=3).fit(X, y)

    test_pts = np.array([[0.05, 0.05], [10.05, 10.05]])
    preds = model.predict(test_pts)
    assert preds[0] == 0
    assert preds[1] == 1


def test_predict_shape():
    iris = load_iris()
    model = KNNClassifierScratch(k=5).fit(iris.data, iris.target)
    preds = model.predict(iris.data[:10])
    assert preds.shape == (10,)


def test_matches_sklearn_on_iris():
    """Should agree with sklearn's KNeighborsClassifier on Iris."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=0
    )

    ours = KNNClassifierScratch(k=5).fit(X_train, y_train)
    theirs = SklearnKNN(n_neighbors=5).fit(X_train, y_train)

    # Predictions can differ slightly when ties are broken differently,
    # but accuracy should be essentially identical.
    our_acc = ours.score(X_test, y_test)
    their_acc = theirs.score(X_test, y_test)
    assert abs(our_acc - their_acc) < 0.05
    assert our_acc > 0.85  # Iris is easy


def test_invalid_k_raises():
    with pytest.raises(ValueError):
        KNNClassifierScratch(k=0)


def test_predict_before_fit_raises():
    model = KNNClassifierScratch()
    with pytest.raises(RuntimeError):
        model.predict(np.array([[1.0, 2.0]]))


def test_k_too_large_raises():
    X = np.array([[1.0], [2.0]])
    y = np.array([0, 1])
    with pytest.raises(ValueError):
        KNNClassifierScratch(k=5).fit(X, y)
