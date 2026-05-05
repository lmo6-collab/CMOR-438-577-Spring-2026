"""
mlpkg — a small machine learning package built for INDE 577 (Spring 2026).

Provides from-scratch implementations of classical ML algorithms with a
scikit-learn-style fit/predict API.
"""

from mlpkg.supervised.linear_regression import LinearRegressionScratch
from mlpkg.supervised.knn import KNNClassifierScratch
from mlpkg.supervised.perceptron import PerceptronScratch
from mlpkg.preprocessing import train_test_split, StandardScaler
from mlpkg.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
)

__version__ = "0.1.0"

__all__ = [
    "LinearRegressionScratch",
    "KNNClassifierScratch",
    "PerceptronScratch",
    "train_test_split",
    "StandardScaler",
    "mean_squared_error",
    "root_mean_squared_error",
    "r2_score",
    "accuracy_score",
]
