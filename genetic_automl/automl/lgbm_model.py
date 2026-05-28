"""
LightGBM backend — wraps LGBMClassifier / LGBMRegressor.

Why LightGBM?
- Leaf-wise tree growth → better accuracy than level-wise (sklearn GBM) at
  the same n_estimators.
- Native early stopping on a validation set — avoids overfitting without a
  separate regularisation search.
- 5-10× faster than sklearn GradientBoosting on most tabular datasets.

All preprocessing is handled upstream by PreprocessingPipeline.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from genetic_automl.core.base_automl import BaseAutoML
from genetic_automl.core.problem import ProblemType
from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)

# How many rounds without improvement before early stopping fires.
_EARLY_STOPPING_ROUNDS = 50


class LGBMModel(BaseAutoML):
    """
    LightGBM wrapper with optional early stopping.

    Parameters
    ----------
    n_estimators : int
        Maximum number of boosting rounds.
    max_depth : int
        Maximum tree depth. -1 = unlimited (LightGBM default).
    learning_rate : float
    num_leaves : int
        Main capacity control for leaf-wise trees.  LightGBM default is 31.
    subsample : float
        Row subsampling fraction per tree.
    colsample_bytree : float
        Column subsampling fraction per tree.
    """

    def __init__(
        self,
        problem_type: ProblemType,
        target_column: str,
        time_limit: int = 60,
        random_seed: int = 42,
        n_estimators: int = 300,
        max_depth: int = -1,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        **kwargs: Any,
    ) -> None:
        super().__init__(problem_type, target_column, time_limit, random_seed, **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self._estimator = None

    # ------------------------------------------------------------------

    def _build_estimator(self):
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as exc:
            raise ImportError(
                "lightgbm is required for the 'lgbm' model type. "
                "Install it with:  pip install lightgbm"
            ) from exc

        params = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_seed,
            verbose=-1,
            n_jobs=-1,
        )
        if self.problem_type in (ProblemType.CLASSIFICATION, ProblemType.MULTI_OBJECTIVE):
            return LGBMClassifier(**params)
        return LGBMRegressor(**params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LGBMModel":
        log.info(
            "LGBMModel fit | n_estimators=%d | lr=%.3f | leaves=%d | "
            "samples=%d | features=%d | early_stopping=%s",
            self.n_estimators, self.learning_rate, self.num_leaves,
            len(y_train), X_train.shape[1],
            "yes" if X_val is not None else "no",
        )
        start = self._start_timer()
        self._estimator = self._build_estimator()

        fit_kwargs: dict = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val.values, y_val.values)]
            fit_kwargs["callbacks"] = [
                _make_early_stopping_callback(_EARLY_STOPPING_ROUNDS)
            ]

        self._estimator.fit(X_train.values, y_train.values, **fit_kwargs)
        self._feature_names = list(X_train.columns)
        self._is_fitted = True
        log.info("LGBMModel fit complete in %.2fs", self._stop_timer(start))
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

    @property
    def feature_importances_(self):
        """Return gain-based feature importances, or None."""
        self._check_fitted()
        if hasattr(self._estimator, "feature_importances_"):
            return self._estimator.feature_importances_
        return None

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            "model_type": "lgbm",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
        }


# ------------------------------------------------------------------
# Helper: build a LightGBM early-stopping callback compatible with
# both lgbm 3.x (EarlyStopping class) and lgbm 4.x (same API, but
# the import path changed slightly in some builds).
# ------------------------------------------------------------------

def _make_early_stopping_callback(rounds: int):
    """Return a LightGBM early-stopping callback, handling version differences."""
    try:
        from lightgbm import early_stopping
        return early_stopping(stopping_rounds=rounds, verbose=False)
    except ImportError:
        # lgbm < 3.3 used the legacy string-based API — return None to skip.
        return None
