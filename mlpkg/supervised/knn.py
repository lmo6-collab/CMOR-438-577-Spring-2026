"""
k-Nearest Neighbors classifier from scratch.

For each query point, find the k closest training points (by Euclidean
distance) and assign the most common class label among them.
"""

import numpy as np
from collections import Counter


class KNNClassifierScratch:
    """
    k-Nearest Neighbors classifier.

    A non-parametric, instance-based classifier. There is no real "training"
    phase — fit() just stores the training data. All work happens at predict
    time.

    Parameters
    ----------
    k : int, default=5
        Number of neighbors to consider when making a prediction.

    Attributes
    ----------
    X_train_ : np.ndarray of shape (n_samples, n_features)
    y_train_ : np.ndarray of shape (n_samples,)
    """

    def __init__(self, k: int = 5):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X, y):
        """Memorize the training data."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have inconsistent number of samples: "
                f"{X.shape[0]} vs {y.shape[0]}"
            )
        if X.shape[0] < self.k:
            raise ValueError(
                f"Need at least k={self.k} training samples, got {X.shape[0]}"
            )

        self.X_train_ = X
        self.y_train_ = y
        return self

    def _predict_one(self, x):
        """Predict the class label for a single sample x."""
        # Compute Euclidean distance from x to every training point
        # (vectorized: broadcasting subtracts x from each row, then we square,
        # sum across features, and take the square root)
        distances = np.sqrt(np.sum((self.X_train_ - x) ** 2, axis=1))

        # Find the indices of the k smallest distances
        k_nearest_idx = np.argsort(distances)[: self.k]

        # Get their labels and return the most common
        k_nearest_labels = self.y_train_[k_nearest_idx]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
        """
        if self.X_train_ is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x) for x in X])

    def score(self, X, y):
        """Return mean classification accuracy on (X, y)."""
        from mlpkg.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
