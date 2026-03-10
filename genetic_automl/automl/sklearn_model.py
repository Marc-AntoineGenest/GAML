"""
Sklearn GBM backend — wraps GradientBoosting{Classifier,Regressor}.

All preprocessing is handled upstream by PreprocessingPipeline.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from genetic_automl.core.base_automl import BaseAutoML
from genetic_automl.core.problem import ProblemType
from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)


class SklearnModel(BaseAutoML):
    """
    Lightweight sklearn GBM wrapper with no internal preprocessing.

    Parameters
    ----------
    n_estimators : int
    max_depth : int
    learning_rate : float
    """

    def __init__(
        self,
        problem_type: ProblemType,
        target_column: str,
        time_limit: int = 60,
        random_seed: int = 42,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(problem_type, target_column, time_limit, random_seed, **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._estimator = None

    def _build_estimator(self):
        params = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_seed,
        )
        if self.problem_type in (ProblemType.CLASSIFICATION, ProblemType.MULTI_OBJECTIVE):
            return GradientBoostingClassifier(**params)
        return GradientBoostingRegressor(**params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "SklearnModel":
        log.info(
            "SklearnModel fit | estimators=%d | depth=%d | samples=%d | features=%d",
            self.n_estimators, self.max_depth, len(y_train), X_train.shape[1],
        )
        start = self._start_timer()
        self._estimator = self._build_estimator()
        self._estimator.fit(X_train.values, y_train.values)
        self._is_fitted = True
        log.info("SklearnModel fit complete in %.2fs", self._stop_timer(start))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._estimator.predict(X.values)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        self._check_fitted()
        if self.problem_type == ProblemType.REGRESSION:
            return None
        return self._estimator.predict_proba(X.values) if hasattr(self._estimator, "predict_proba") else None

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            "backend": "sklearn",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
        }
