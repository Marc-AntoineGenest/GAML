"""
DistributionTransform
---------------------
Applies a power transform to numeric features to reduce skewness and
make distributions more Gaussian before scaling.

Many ML algorithms (linear models, SVMs, KNN, neural nets) perform better
when features are approximately normally distributed. Even tree-based methods
can benefit when combined with proper scaling.

Methods
-------
none        : no-op
yeo-johnson : Yeo-Johnson transform — handles negative values, most versatile
box-cox     : Box-Cox transform — requires strictly positive values (auto-shifts)
log1p       : log(x+1) — fast, interpretable, good for count data and right skew

Only applied to numeric columns with significant skewness (|skew| > threshold).
Columns below the skew threshold are left untouched to avoid unnecessary distortion.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)

_SKEW_THRESHOLD = 0.5   # only transform columns with |skewness| above this


class DistributionTransform:
    """
    Parameters
    ----------
    method : str
        'none' | 'yeo-johnson' | 'box-cox' | 'log1p'
    skew_threshold : float
        Only transform columns whose absolute skewness exceeds this value.
        Avoids distorting already-symmetric features.
    """

    def __init__(
        self,
        method: str = "none",
        skew_threshold: float = _SKEW_THRESHOLD,
    ) -> None:
        self.method = method
        self.skew_threshold = skew_threshold

        self._transformers: Dict[str, object] = {}  # col → fitted transformer
        self._log1p_cols: List[str] = []            # for log1p: cols to transform
        self._shifts: Dict[str, float] = {}         # for box-cox: shift to make positive
        self._skewed_cols: List[str] = []           # cols that exceeded threshold

    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None) -> "DistributionTransform":
        if self.method == "none":
            return self

        num_cols = X.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            return self

        # Identify skewed columns
        skewness = X[num_cols].skew()
        self._skewed_cols = [
            c for c in num_cols if abs(skewness.get(c, 0)) > self.skew_threshold
        ]

        if not self._skewed_cols:
            log.info("DistributionTransform(%s): no skewed columns found (threshold=%.1f)", self.method, self.skew_threshold)
            return self

        log.info(
            "DistributionTransform(%s): fitting on %d/%d skewed columns",
            self.method, len(self._skewed_cols), len(num_cols),
        )

        if self.method == "yeo-johnson":
            for col in self._skewed_cols:
                pt = PowerTransformer(method="yeo-johnson", standardize=False)
                pt.fit(X[[col]].fillna(0))
                self._transformers[col] = pt

        elif self.method == "box-cox":
            for col in self._skewed_cols:
                # Box-Cox requires strictly positive values — shift if needed
                col_min = X[col].min()
                shift = max(0.0, -col_min + 1e-6)  # shift so all values > 0
                self._shifts[col] = shift
                pt = PowerTransformer(method="box-cox", standardize=False)
                pt.fit((X[[col]] + shift).fillna(1e-6))
                self._transformers[col] = pt

        elif self.method == "log1p":
            for col in self._skewed_cols:
                # log1p requires non-negative — record shift
                col_min = X[col].min()
                shift = max(0.0, -col_min)
                self._shifts[col] = shift
            self._log1p_cols = self._skewed_cols

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.method == "none" or not self._skewed_cols:
            return X
        X = X.copy()

        if self.method in ("yeo-johnson", "box-cox"):
            for col, pt in self._transformers.items():
                if col not in X.columns:
                    continue
                shift = self._shifts.get(col, 0.0)
                vals = (X[[col]] + shift).fillna(0)
                if self.method == "box-cox":
                    vals = vals.clip(lower=1e-6)
                try:
                    X[col] = pt.transform(vals).ravel()
                except Exception as e:
                    log.warning("DistributionTransform: col '%s' failed transform: %s", col, e)

        elif self.method == "log1p":
            for col in self._log1p_cols:
                if col not in X.columns:
                    continue
                shift = self._shifts.get(col, 0.0)
                X[col] = np.log1p((X[col] + shift).clip(lower=0).fillna(0))

        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    @property
    def transformed_columns(self) -> List[str]:
        return list(self._skewed_cols)
