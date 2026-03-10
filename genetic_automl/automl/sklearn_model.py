"""
Sklearn fallback backend.

B4 fix: the internal ColumnTransformer (imputer + scaler + one-hot encoder) has been
removed. By the time data reaches SklearnModel it has already been fully preprocessed
by PreprocessingPipeline (imputation, scaling, encoding, feature selection, etc.).
Applying a second StandardScaler on top of already-scaled data biased the model away
from what the GA evaluated during CV.

The model now wraps only the estimator — no internal preprocessing.
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
    Lightweight sklearn GBM wrapper — no internal preprocessing.

    All preprocessing (imputation, scaling, encoding, feature selection)
    is handled upstream by PreprocessingPipeline before this class is called.

    Parameters
    ----------
    n_estimators : int
        Number of boosting stages.
    max_depth : int
        Maximum tree depth.
    learning_rate : float
        Shrinkage parameter.
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

    # ------------------------------------------------------------------

    def _build_estimator(self):
        if self.problem_type in (
            ProblemType.CLASSIFICATION,
            ProblemType.MULTI_OBJECTIVE,
        ):
            return GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_seed,
            )
        return GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_seed,
        )

    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "SklearnModel":
        log.info(
            "SklearnModel fit | estimators=%d | depth=%d | samples=%d | features=%d",
            self.n_estimators,
            self.max_depth,
            len(y_train),
            X_train.shape[1],
        )
        start = self._start_timer()
        self._estimator = self._build_estimator()
        # Data already preprocessed upstream — fit estimator directly
        self._estimator.fit(X_train.values, y_train.values)
        elapsed = self._stop_timer(start)
        self._is_fitted = True
        log.info("SklearnModel fit complete in %.2fs", elapsed)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._estimator.predict(X.values)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        self._check_fitted()
        if self.problem_type == ProblemType.REGRESSION:
            return None
        if hasattr(self._estimator, "predict_proba"):
            return self._estimator.predict_proba(X.values)
        return None

    def get_params(self) -> dict:
        base = super().get_params()
        base.update(
            {
                "backend": "sklearn",
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
            }
        )
        return base

