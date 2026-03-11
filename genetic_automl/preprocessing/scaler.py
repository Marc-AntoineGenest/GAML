"""
Scaler
------
Scales numeric features. Fit on training data only.

Strategies
----------
none        : no-op
standard    : zero mean, unit variance (StandardScaler)
minmax      : scale to [0, 1] (MinMaxScaler)
robust      : median + IQR scaling — outlier resistant (RobustScaler)

Implementation note
-------------------
Each column gets its own fitted scaler instance.  A single joint scaler would
fail at transform time whenever the test/val set is missing a column that was
present at fit time (e.g. after CorrelationFilter removes a feature on a
different data slice), because sklearn validates n_features_in_ against the
number of columns passed.  Per-column scalers sidestep this entirely: only the
columns that are present in both fit and transform are scaled; absent columns
are left untouched.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)


class Scaler:
    """
    Parameters
    ----------
    method : str
        'none' | 'standard' | 'minmax' | 'robust'
    """

    def __init__(self, method: str = "standard") -> None:
        self.method = method
        # Per-column scaler instances: {col_name: fitted_sklearn_scaler}
        self._col_scalers: Dict[str, object] = {}
        self._num_cols: List[str] = []

    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "Scaler":
        self._num_cols = X.select_dtypes(include="number").columns.tolist()
        self._col_scalers = {}

        if self.method == "none" or not self._num_cols:
            log.info("Scaler(method=none): skipped")
            return self

        for col in self._num_cols:
            sc = self._build_scaler()
            sc.fit(X[[col]])
            self._col_scalers[col] = sc

        log.info("Scaler(method=%s): fitted on %d numeric columns", self.method, len(self._num_cols))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.method == "none" or not self._col_scalers:
            return X
        X = X.copy()
        for col, sc in self._col_scalers.items():
            if col not in X.columns:
                continue  # column absent in this split — skip gracefully
            X[col] = sc.transform(X[[col]]).ravel()
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------

    def _build_scaler(self):
        if self.method == "standard":
            from sklearn.preprocessing import StandardScaler
            return StandardScaler()
        if self.method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            return MinMaxScaler()
        if self.method == "robust":
            from sklearn.preprocessing import RobustScaler
            return RobustScaler()
        raise ValueError(f"Unknown scaler method: '{self.method}'")
