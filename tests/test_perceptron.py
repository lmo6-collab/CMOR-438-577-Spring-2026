"""Tests for PerceptronScratch."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from mlpkg.supervised import PerceptronScratch


def test_linearly_separable_converges():
    """On well-separated data, perceptron should converge to 100% accuracy."""
    X = np.array([
        [0.0, 0.0], [0.5, 0.5], [1.0, 1.0],   # negative class
        [5.0, 5.0], [5.5, 5.5], [6.0, 6.0],   # positive class
    ])
    y = np.array([-1, -1, -1, 1, 1, 1])
    model = PerceptronScratch(learning_rate=0.1, n_epochs=50).fit(X, y)
    assert model.score(X, y) == 1.0


def test_predict_shape():
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3,
        n_redundant=0, n_classes=2, random_state=42,
    )
    model = PerceptronScratch(n_epochs=50).fit(X, y)
    preds = model.predict(X[:10])
    assert preds.shape == (10,)


def test_string_labels_work():
    """Should accept arbitrary 2-class labels, not just integers."""
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    y = np.array(['cat', 'cat', 'dog', 'dog'])
    model = PerceptronScratch(learning_rate=0.5, n_epochs=20).fit(X, y)
    preds = model.predict(np.array([[0.5], [10.5]]))
    assert preds[0] == 'cat'
    assert preds[1] == 'dog'


def test_records_errors_per_epoch():
    """The errors_per_epoch_ list should be populated and (eventually) drop."""
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    y = np.array([0, 0, 1, 1])
    model = PerceptronScratch(learning_rate=0.5, n_epochs=20).fit(X, y)
    assert len(model.errors_per_epoch_) >= 1
    # On separable data, last epoch should have zero or very few errors
    assert model.errors_per_epoch_[-1] <= model.errors_per_epoch_[0]


def test_invalid_lr_raises():
    with pytest.raises(ValueError):
        PerceptronScratch(learning_rate=-0.1)


def test_invalid_epochs_raises():
    with pytest.raises(ValueError):
        PerceptronScratch(n_epochs=0)


def test_three_classes_raises():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1, 2])
    with pytest.raises(ValueError):
        PerceptronScratch().fit(X, y)


def test_predict_before_fit_raises():
    model = PerceptronScratch()
    with pytest.raises(RuntimeError):
        model.predict(np.array([[1.0, 2.0]]))


def test_high_accuracy_on_easy_classification():
    """On an easy synthetic classification dataset, accuracy should be >85%."""
    X, y = make_classification(
        n_samples=200, n_features=4, n_informative=4,
        n_redundant=0, n_classes=2, class_sep=2.0, random_state=0,
    )
    model = PerceptronScratch(learning_rate=0.01, n_epochs=200).fit(X, y)
    assert model.score(X, y) > 0.80
