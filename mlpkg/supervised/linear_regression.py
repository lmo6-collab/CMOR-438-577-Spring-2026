"""
Linear Regression — Ordinary Least Squares (OLS) from scratch.

Solves the normal equation: w = (X^T X)^(-1) X^T y

This closed-form solution minimizes the sum of squared residuals:
    L(w) = || y - X w ||^2
"""

import numpy as np


class LinearRegressionScratch:
    """
    Ordinary Least Squares linear regression.

    Fits a linear model y = X @ w + b by solving the normal equations
    directly. Equivalent to sklearn.linear_model.LinearRegression.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Estimated coefficients for the linear regression problem.
    intercept_ : float
        Independent term (bias).
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        """
        Fit the linear model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have inconsistent number of samples: "
                f"{X.shape[0]} vs {y.shape[0]}"
            )

        if self.fit_intercept:
            # Add a column of ones for the bias term
            X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        else:
            X_aug = X

        # Solve normal equations using lstsq for numerical stability
        # (handles ill-conditioned X^T X better than direct inversion)
        weights, *_ = np.linalg.lstsq(X_aug, y, rcond=None)

        if self.fit_intercept:
            self.intercept_ = float(weights[0])
            self.coef_ = weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = weights

        return self

    def predict(self, X):
        """
        Predict target values for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
        """
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.

        R^2 = 1 - SS_res / SS_tot
        """
        from mlpkg.metrics import r2_score
        return r2_score(y, self.predict(X))
