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
"""

from __future__ import annotations

from typing import List, Optional

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
        self._scaler = None
        self._num_cols: List[str] = []

    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "Scaler":
        self._num_cols = X.select_dtypes(include="number").columns.tolist()

        if self.method == "none" or not self._num_cols:
            log.info("Scaler(method=none): skipped")
            return self

        self._scaler = self._build_scaler()
        self._scaler.fit(X[self._num_cols])
        log.info("Scaler(method=%s): fitted on %d numeric columns", self.method, len(self._num_cols))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.method == "none" or self._scaler is None or not self._num_cols:
            return X
        X = X.copy()
        cols_present = [c for c in self._num_cols if c in X.columns]
        X[cols_present] = self._scaler.transform(X[cols_present])
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
