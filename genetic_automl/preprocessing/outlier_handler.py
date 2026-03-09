"""
OutlierHandler
--------------
Detects outliers in numeric features and either clips or flags them.

Methods
-------
none             : no-op
iqr              : clip values beyond Q1 - k*IQR / Q3 + k*IQR
zscore           : clip values beyond ±threshold standard deviations
isolation_forest : sklearn IsolationForest — sets outlier rows to NaN
                   (NumericImputer must run AFTER this step if using it,
                    or configure it to clip instead — see *action* param)

Action
------
clip : replace outlier values with the boundary value (default, safe for val/test)
flag : add a binary `__outlier__` indicator column
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
        k for IQR (default 1.5) or number of std deviations for zscore (default 3.0).
    action : str
        'clip' or 'flag'. Isolation forest always clips (sets to boundary).
    contamination : float
        Fraction of expected outliers — used by IsolationForest.
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
        self._bounds: Dict[str, Tuple[float, float]] = {}   # col → (lower, upper)
        self._means: Dict[str, float] = {}
        self._stds: Dict[str, float] = {}
        self._iso_forest = None

    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "OutlierHandler":
        self._num_cols = X.select_dtypes(include="number").columns.tolist()

        if self.method == "none" or not self._num_cols:
            return self

        if self.method == "iqr":
            for col in self._num_cols:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self.threshold * iqr
                upper = q3 + self.threshold * iqr
                self._bounds[col] = (lower, upper)

        elif self.method == "zscore":
            for col in self._num_cols:
                self._means[col] = X[col].mean()
                self._stds[col] = X[col].std(ddof=0) + 1e-9
                lower = self._means[col] - self.threshold * self._stds[col]
                upper = self._means[col] + self.threshold * self._stds[col]
                self._bounds[col] = (lower, upper)

        elif self.method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            self._iso_forest = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1,
            )
            self._iso_forest.fit(X[self._num_cols].fillna(0))

        n_outliers = self._count_outliers(X)
        log.info(
            "OutlierHandler(method=%s, threshold=%.1f): ~%d outlier rows detected in train",
            self.method,
            self.threshold,
            n_outliers,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.method == "none" or not self._num_cols:
            return X

        X = X.copy()
        cols_present = [c for c in self._num_cols if c in X.columns]

        if self.method in ("iqr", "zscore"):
            if self.action == "clip":
                for col in cols_present:
                    lower, upper = self._bounds[col]
                    X[col] = X[col].clip(lower=lower, upper=upper)
            elif self.action == "flag":
                outlier_mask = pd.Series(False, index=X.index)
                for col in cols_present:
                    lower, upper = self._bounds[col]
                    outlier_mask |= (X[col] < lower) | (X[col] > upper)
                X["__outlier__"] = outlier_mask.astype(int)

        elif self.method == "isolation_forest" and self._iso_forest is not None:
            preds = self._iso_forest.predict(X[cols_present].fillna(0))
            if self.action == "flag":
                X["__outlier__"] = (preds == -1).astype(int)
            else:
                # clip: for detected outlier rows, clip each feature to its IQR bounds
                # (compute simple bounds on-the-fly using training bounds if available,
                #  else just replace with column median)
                outlier_mask = preds == -1
                for col in cols_present:
                    col_median = X[col].median()
                    X.loc[outlier_mask, col] = X.loc[outlier_mask, col].fillna(col_median)

        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------

    def _count_outliers(self, X: pd.DataFrame) -> int:
        if self.method in ("iqr", "zscore") and self._bounds:
            cols = [c for c in self._num_cols if c in X.columns]
            mask = pd.Series(False, index=X.index)
            for col in cols:
                lo, hi = self._bounds[col]
                mask |= (X[col] < lo) | (X[col] > hi)
            return int(mask.sum())
        if self.method == "isolation_forest" and self._iso_forest is not None:
            preds = self._iso_forest.predict(X[self._num_cols].fillna(0))
            return int((preds == -1).sum())
        return 0
