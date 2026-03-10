"""
OutlierHandler
--------------
Detects and handles outliers in numeric features.

Methods
-------
none             : no-op
iqr              : flag/clip beyond Q1 - k*IQR / Q3 + k*IQR
zscore           : flag/clip beyond ±threshold standard deviations
isolation_forest : sklearn IsolationForest

Action
------
clip : replace outlier values with boundary (or training median for IsolationForest)
flag : add a binary __outlier__ indicator column
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)


class OutlierHandler:
    """
    Parameters
    ----------
    method : str
        'none' | 'iqr' | 'zscore' | 'isolation_forest'
    threshold : float
        k multiplier for IQR, or standard-deviation count for zscore.
    action : str
        'clip' or 'flag'.
    contamination : float
        Expected outlier fraction for IsolationForest.
    """

    def __init__(
        self,
        method: str = "none",
        threshold: float = 1.5,
        action: str = "clip",
        contamination: float = 0.05,
    ) -> None:
        self.method = method
        self.threshold = threshold
        self.action = action
        self.contamination = contamination

        self._num_cols: List[str] = []
        self._bounds: Dict[str, Tuple[float, float]] = {}
        self._means: Dict[str, float] = {}
        self._stds: Dict[str, float] = {}
        self._medians: Dict[str, float] = {}
        self._iso_forest = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "OutlierHandler":
        self._num_cols = X.select_dtypes(include="number").columns.tolist()
        if self.method == "none" or not self._num_cols:
            return self

        if self.method == "iqr":
            for col in self._num_cols:
                q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
                iqr = q3 - q1
                self._bounds[col] = (q1 - self.threshold * iqr, q3 + self.threshold * iqr)

        elif self.method == "zscore":
            for col in self._num_cols:
                self._means[col] = X[col].mean()
                self._stds[col] = X[col].std(ddof=0) + 1e-9
                self._bounds[col] = (
                    self._means[col] - self.threshold * self._stds[col],
                    self._means[col] + self.threshold * self._stds[col],
                )

        elif self.method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            self._iso_forest = IsolationForest(
                contamination=self.contamination, random_state=42, n_jobs=-1,
            )
            self._iso_forest.fit(X[self._num_cols].fillna(0))
            # Store training medians; used at transform time to avoid val/test leakage
            for col in self._num_cols:
                self._medians[col] = float(X[col].median())

        log.info(
            "OutlierHandler(method=%s, threshold=%.1f): ~%d outlier rows in train",
            self.method, self.threshold, self._count_outliers(X),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.method == "none" or not self._num_cols:
            return X
        X = X.copy()
        cols = [c for c in self._num_cols if c in X.columns]

        if self.method in ("iqr", "zscore"):
            if self.action == "clip":
                for col in cols:
                    lo, hi = self._bounds[col]
                    X[col] = X[col].clip(lower=lo, upper=hi)
            else:
                mask = pd.Series(False, index=X.index)
                for col in cols:
                    lo, hi = self._bounds[col]
                    mask |= (X[col] < lo) | (X[col] > hi)
                X["__outlier__"] = mask.astype(int)

        elif self.method == "isolation_forest" and self._iso_forest is not None:
            preds = self._iso_forest.predict(X[cols].fillna(0))
            if self.action == "flag":
                X["__outlier__"] = (preds == -1).astype(int)
            else:
                mask = preds == -1
                for col in cols:
                    X.loc[mask, col] = self._medians.get(col, 0.0)

        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def _count_outliers(self, X: pd.DataFrame) -> int:
        if self.method in ("iqr", "zscore") and self._bounds:
            mask = pd.Series(False, index=X.index)
            for col in [c for c in self._num_cols if c in X.columns]:
                lo, hi = self._bounds[col]
                mask |= (X[col] < lo) | (X[col] > hi)
            return int(mask.sum())
        if self.method == "isolation_forest" and self._iso_forest is not None:
            return int((self._iso_forest.predict(X[self._num_cols].fillna(0)) == -1).sum())
        return 0
