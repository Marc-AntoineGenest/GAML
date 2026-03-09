"""
CorrelationFilter
-----------------
Drops features whose Pearson correlation with another feature exceeds
*threshold*. Always runs first in the pipeline (cheapest step, reduces
dimensionality for all downstream steps).

Fit/transform contract:
  - fit()           : learn which columns to drop (on train only)
  - transform()     : apply the same column mask to any split
  - No target leakage: correlation is computed on X only
"""

from __future__ import annotations

from typing import List, Optional, Set

import numpy as np
import pandas as pd

from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)


class CorrelationFilter:
    """
    Remove features with pairwise Pearson correlation above *threshold*.

    Parameters
    ----------
    threshold : float | None
        Correlation cutoff [0, 1]. None disables the step entirely.
    """

    def __init__(self, threshold: Optional[float] = 0.95) -> None:
        self.threshold = threshold
        self._cols_to_drop: List[str] = []
        self._feature_names_in: List[str] = []

    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "CorrelationFilter":
        self._feature_names_in = list(X.columns)
        self._cols_to_drop = []

        if self.threshold is None:
            return self

        # Only numeric columns are considered for correlation
        num_X = X.select_dtypes(include="number")
        if num_X.shape[1] < 2:
            return self

        corr = num_X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop: Set[str] = set()
        for col in upper.columns:
            if col in to_drop:
                continue
            correlated = upper.index[upper[col] > self.threshold].tolist()
            for c in correlated:
                if c not in to_drop:
                    to_drop.add(c)
                    log.debug("CorrelationFilter: drop '%s' (corr with '%s' > %.2f)", c, col, self.threshold)

        self._cols_to_drop = list(to_drop)
        log.info(
            "CorrelationFilter(threshold=%.2f): dropping %d / %d features",
            self.threshold,
            len(self._cols_to_drop),
            X.shape[1],
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._cols_to_drop:
            return X
        cols_present = [c for c in self._cols_to_drop if c in X.columns]
        return X.drop(columns=cols_present)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    @property
    def dropped_features(self) -> List[str]:
        return list(self._cols_to_drop)
