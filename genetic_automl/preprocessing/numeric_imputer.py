"""
NumericImputer
--------------
Handles missing values in numeric columns.

Strategies
----------
mean        : sklearn SimpleImputer, strategy='mean'
median      : sklearn SimpleImputer, strategy='median'
knn         : sklearn KNNImputer (k=5)
iterative   : sklearn IterativeImputer (experimental)
constant    : fill with 0

Categorical columns are handled by CategoricalEncoder, not here.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer

from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)

_VALID = {"mean", "median", "knn", "iterative", "constant"}


class NumericImputer:
    """
    Fit on numeric columns of training data, transform any split.

    Parameters
    ----------
    strategy : str
        One of 'mean', 'median', 'knn', 'iterative', 'constant'.
    """

    def __init__(self, strategy: str = "median") -> None:
        if strategy not in _VALID:
            raise ValueError(f"strategy must be one of {_VALID}, got '{strategy}'")
        self.strategy = strategy
        self._imputer = None
        self._num_cols: List[str] = []

    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "NumericImputer":
        self._num_cols = X.select_dtypes(include="number").columns.tolist()

        if not self._num_cols:
            return self

        missing_counts = X[self._num_cols].isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]
        log.info(
            "NumericImputer(strategy=%s): %d numeric cols, %d with missing",
            self.strategy,
            len(self._num_cols),
            len(cols_with_missing),
        )

        self._imputer = self._build_imputer()
        self._imputer.fit(X[self._num_cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._num_cols or self._imputer is None:
            return X
        X = X.copy()
        cols_present = [c for c in self._num_cols if c in X.columns]
        imputed = self._imputer.transform(X[cols_present])
        X[cols_present] = imputed
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------

    def _build_imputer(self):
        if self.strategy in ("mean", "median", "constant"):
            fill = 0 if self.strategy == "constant" else self.strategy
            return SimpleImputer(strategy=fill if self.strategy != "constant" else "constant",
                                 fill_value=0 if self.strategy == "constant" else None)
        if self.strategy == "knn":
            return KNNImputer(n_neighbors=5)
        if self.strategy == "iterative":
            try:
                from sklearn.experimental import enable_iterative_imputer  # noqa
                from sklearn.impute import IterativeImputer
                return IterativeImputer(max_iter=10, random_state=0)
            except Exception:
                log.warning("IterativeImputer unavailable, falling back to median.")
                return SimpleImputer(strategy="median")
        raise ValueError(f"Unknown strategy: {self.strategy}")
