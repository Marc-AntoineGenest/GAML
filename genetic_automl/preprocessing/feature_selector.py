"""
FeatureSelector
---------------
Selects the most informative features after encoding + scaling.

Methods
-------
none               : keep all features
variance_threshold : drop near-zero-variance features
mutual_info        : keep top-k by mutual information with target
rfe                : recursive feature elimination with ExtraTrees

keep_k controls retention:
  float in (0, 1] → fraction of features
  int > 1         → absolute count
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import pandas as pd

from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)


class FeatureSelector:
    """
    Parameters
    ----------
    method : str
        'none' | 'variance_threshold' | 'mutual_info' | 'rfe'
    keep_k : float | int
        Fraction or absolute count of features to retain.
    variance_threshold : float
        Minimum variance for 'variance_threshold' method.
    problem_type_str : str
        'classification' or 'regression'.
    random_seed : int
    """

    def __init__(
        self,
        method: str = "none",
        keep_k: Union[float, int] = 0.75,
        variance_threshold: float = 0.01,
        problem_type_str: str = "classification",
        random_seed: int = 42,
    ) -> None:
        self.method = method
        self.keep_k = keep_k
        self.variance_threshold = variance_threshold
        self.problem_type_str = problem_type_str
        self.random_seed = random_seed

        self._selected_cols: List[str] = []
        self._selector = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelector":
        n_total = X.shape[1]

        if self.method == "none" or n_total == 0:
            self._selected_cols = list(X.columns)
            log.info("FeatureSelector(method=none): keeping all %d features", n_total)
            return self

        k = self._resolve_k(n_total)

        if self.method == "variance_threshold":
            from sklearn.feature_selection import VarianceThreshold
            self._selector = VarianceThreshold(threshold=self.variance_threshold)
            try:
                self._selector.fit(X.fillna(0))
                mask = self._selector.get_support()
                self._selected_cols = [c for c, m in zip(X.columns, mask) if m]
            except Exception as e:
                log.warning("VarianceThreshold failed: %s — keeping all", e)
                self._selected_cols = list(X.columns)

        elif self.method == "mutual_info":
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
            fn = mutual_info_classif if self.problem_type_str == "classification" else mutual_info_regression
            try:
                scores = fn(X.fillna(0), y, random_state=self.random_seed)
                top_idx = np.argsort(scores)[::-1][:k]
                self._selected_cols = [X.columns[i] for i in sorted(top_idx)]
            except Exception as e:
                log.warning("mutual_info failed: %s — keeping all", e)
                self._selected_cols = list(X.columns)

        elif self.method == "rfe":
            from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
            from sklearn.feature_selection import RFE
            estimator = (
                ExtraTreesClassifier(n_estimators=50, random_state=self.random_seed, n_jobs=-1)
                if self.problem_type_str == "classification"
                else ExtraTreesRegressor(n_estimators=50, random_state=self.random_seed, n_jobs=-1)
            )
            try:
                self._selector = RFE(estimator=estimator, n_features_to_select=k)
                self._selector.fit(X.fillna(0), y)
                mask = self._selector.get_support()
                self._selected_cols = [c for c, m in zip(X.columns, mask) if m]
            except Exception as e:
                log.warning("RFE failed: %s — keeping all", e)
                self._selected_cols = list(X.columns)

        log.info("FeatureSelector(method=%s): %d → %d features", self.method, n_total, len(self._selected_cols))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._selected_cols:
            return X

        missing = [c for c in self._selected_cols if c not in X.columns]
        if missing:
            raise ValueError(
                f"FeatureSelector.transform(): {len(missing)} selected column(s) missing from input: "
                f"{missing}. This usually means an upstream step behaved differently at transform "
                f"time than at fit time. Present columns: {list(X.columns)}"
            )
        return X[[c for c in self._selected_cols if c in X.columns]]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    @property
    def selected_features(self) -> List[str]:
        return list(self._selected_cols)

    def _resolve_k(self, n_total: int) -> int:
        if isinstance(self.keep_k, float) and 0 < self.keep_k <= 1.0:
            return max(1, int(n_total * self.keep_k))
        if isinstance(self.keep_k, int) and self.keep_k > 0:
            return min(self.keep_k, n_total)
        return n_total
