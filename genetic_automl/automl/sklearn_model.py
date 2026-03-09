"""
Sklearn fallback backend.

Uses a simple but solid pipeline:
  - Preprocessing: median imputation + standard scaling (numeric),
                   most-frequent imputation + one-hot encoding (categorical)
  - Model: RandomForest or GradientBoosting depending on problem type

No AutoGluon required. Useful for quick tests and CI.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from genetic_automl.core.base_automl import BaseAutoML
from genetic_automl.core.problem import ProblemType
from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)


class SklearnModel(BaseAutoML):
    """
    Lightweight sklearn-based AutoML fallback.

    Parameters
    ----------
    n_estimators : int
        Number of boosting stages.
    max_depth : int
        Maximum tree depth.
    learning_rate : float
        Shrinkage parameter (GBM only).
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
        self._pipeline: Optional[Pipeline] = None

    # ------------------------------------------------------------------

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        num_cols = X.select_dtypes(include="number").columns.tolist()
        cat_cols = X.select_dtypes(exclude="number").columns.tolist()

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers = []
        if num_cols:
            transformers.append(("num", numeric_transformer, num_cols))
        if cat_cols:
            transformers.append(("cat", categorical_transformer, cat_cols))
        return ColumnTransformer(transformers=transformers, remainder="drop")

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
            "SklearnModel fit | estimators=%d | depth=%d",
            self.n_estimators,
            self.max_depth,
        )
        start = self._start_timer()
        preprocessor = self._build_preprocessor(X_train)
        estimator = self._build_estimator()
        self._pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("estimator", estimator)]
        )
        self._pipeline.fit(X_train, y_train)
        elapsed = self._stop_timer(start)
        self._is_fitted = True
        log.info("SklearnModel fit complete in %.2fs", elapsed)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        self._check_fitted()
        if self.problem_type == ProblemType.REGRESSION:
            return None
        if hasattr(self._pipeline, "predict_proba"):
            return self._pipeline.predict_proba(X)
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
