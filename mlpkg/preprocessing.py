"""Data preprocessing utilities: train/test split and feature scaling."""

import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split arrays into random train and test subsets.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    test_size : float in (0, 1), default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int or None, default=None
        Seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if not (0 < test_size < 1):
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Inconsistent samples: X has {X.shape[0]}, y has {y.shape[0]}"
        )

    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    indices = rng.permutation(n)
    n_test = int(round(n * test_size))

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.

    z = (x - mean) / std

    Equivalent to sklearn.preprocessing.StandardScaler.
    """

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0, ddof=0)
        # Avoid division by zero on constant columns
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise RuntimeError("Scaler not fitted. Call .fit() first.")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
