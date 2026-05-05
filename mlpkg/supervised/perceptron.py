"""
Perceptron — Rosenblatt's classic linear classifier from scratch.

The perceptron is one of the foundational algorithms in machine learning,
introduced by Frank Rosenblatt in 1958. It is a single-layer neural network
that learns a linear decision boundary by iteratively updating its weights
on misclassified examples.

For each sample, the perceptron computes:
    z = w · x + b
    y_hat = sign(z)   (i.e., +1 if z >= 0 else -1)

The learning rule (only updates on errors):
    if y_true != y_hat:
        w := w + lr * y_true * x
        b := b + lr * y_true

This algorithm is guaranteed to converge if and only if the data is
linearly separable — a result known as the Perceptron Convergence Theorem.
"""

import numpy as np


class PerceptronScratch:
    """
    Rosenblatt's perceptron for binary classification.

    Labels are mapped to {-1, +1} internally. The user can supply labels
    of any two distinct values (e.g., {0, 1} or {'cat', 'dog'}).

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for weight updates.
    n_epochs : int, default=100
        Maximum number of passes over the training data.
    random_state : int or None, default=42
        Seed for weight initialization and sample shuffling.

    Attributes
    ----------
    weights_ : np.ndarray of shape (n_features,)
        Learned weight vector.
    bias_ : float
        Learned bias term.
    classes_ : np.ndarray of shape (2,)
        The two unique class labels seen during fit.
    errors_per_epoch_ : list of int
        Number of misclassifications at each epoch (useful for plotting
        convergence).
    """

    def __init__(self, learning_rate=0.01, n_epochs=100, random_state=42):
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        if n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1, got {n_epochs}")
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state

    @staticmethod
    def _sign(z):
        """Sign activation: returns +1 if z >= 0 else -1."""
        return np.where(z >= 0, 1, -1)

    def fit(self, X, y):
        """Fit the perceptron on the training data."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Inconsistent samples: X={X.shape[0]} vs y={y.shape[0]}"
            )

        # Map class labels to {-1, +1}
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                f"Perceptron is binary; got {len(self.classes_)} classes"
            )
        y_signed = np.where(y == self.classes_[1], 1, -1)

        # Initialize weights small and random
        rng = np.random.default_rng(self.random_state)
        n_features = X.shape[1]
        self.weights_ = rng.normal(scale=0.01, size=n_features)
        self.bias_ = 0.0
        self.errors_per_epoch_ = []

        for _ in range(self.n_epochs):
            errors = 0
            # Shuffle for stochastic updates
            indices = rng.permutation(X.shape[0])
            for i in indices:
                xi, yi = X[i], y_signed[i]
                pred = self._sign(np.dot(self.weights_, xi) + self.bias_)
                if pred != yi:
                    # Perceptron update rule
                    self.weights_ += self.learning_rate * yi * xi
                    self.bias_ += self.learning_rate * yi
                    errors += 1
            self.errors_per_epoch_.append(errors)
            # Early stop on convergence
            if errors == 0:
                break

        return self

    def predict(self, X):
        """Predict class labels for X."""
        if not hasattr(self, "weights_"):
            raise RuntimeError("Model not fitted. Call .fit() first.")
        X = np.asarray(X, dtype=float)
        z = X @ self.weights_ + self.bias_
        signed = self._sign(z)
        # Map {-1, +1} back to original class labels
        return np.where(signed == 1, self.classes_[1], self.classes_[0])

    def score(self, X, y):
        """Mean classification accuracy on (X, y)."""
        from mlpkg.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
