"""
AutoGluon backend — wraps TabularPredictor behind the BaseAutoML interface.

Install:  pip install autogluon.tabular
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Optional

import numpy as np
import pandas as pd

from genetic_automl.core.base_automl import BaseAutoML
from genetic_automl.core.problem import ProblemType
from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)

# Map our ProblemType to AutoGluon's problem_type string
_AG_PROBLEM_TYPE = {
    ProblemType.CLASSIFICATION: "multiclass",
    ProblemType.REGRESSION: "regression",
    ProblemType.MULTI_OBJECTIVE: "regression",  # handled externally
}

# Map our metric names to AutoGluon eval_metric strings
_AG_METRIC_MAP = {
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
    "mse": "mean_squared_error",
    "mae": "mean_absolute_error",
    "r2": "r2",
}


class AutoGluonModel(BaseAutoML):
    """
    AutoGluon TabularPredictor wrapped as a BaseAutoML.

    Parameters
    ----------
    problem_type, target_column, time_limit, random_seed
        Inherited from BaseAutoML.
    presets : str
        AutoGluon presets ('best_quality', 'high_quality', 'medium_quality',
        'optimize_for_deployment', 'good_quality', 'ignore_text').
    ag_metric : str | None
        AutoGluon eval_metric. Inferred from problem_type when None.
    model_dir : str | None
        Where to save AutoGluon artifacts. Uses a temp dir when None.
    keep_model_dir : bool
        Retain the model directory after fitting (useful for later prediction).
    """

    def __init__(
        self,
        problem_type: ProblemType,
        target_column: str,
        time_limit: int = 60,
        random_seed: int = 42,
        presets: str = "medium_quality",
        ag_metric: Optional[str] = None,
        model_dir: Optional[str] = None,
        keep_model_dir: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(problem_type, target_column, time_limit, random_seed, **kwargs)
        self.presets = presets
        self.ag_metric = ag_metric
        self.model_dir = model_dir
        self.keep_model_dir = keep_model_dir
        self._predictor = None
        self._tmp_dir: Optional[str] = None

    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "AutoGluonModel":
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError as e:
            raise ImportError(
                "AutoGluon is not installed. "
                "Run: pip install autogluon.tabular"
            ) from e

        # Build a single DataFrame with the label column included
        train_df = X_train.copy()
        train_df[self.target_column] = y_train.values

        tuning_data = None
        if X_val is not None and y_val is not None:
            tuning_data = X_val.copy()
            tuning_data[self.target_column] = y_val.values

        # Determine model output directory
        if self.model_dir is not None:
            path = self.model_dir
        else:
            self._tmp_dir = tempfile.mkdtemp(prefix="ag_automl_")
            path = self._tmp_dir

        ag_problem = _AG_PROBLEM_TYPE.get(self.problem_type, "multiclass")
        metric = self.ag_metric

        log.info(
            "AutoGluon fit | problem=%s | presets=%s | time_limit=%ds",
            ag_problem,
            self.presets,
            self.time_limit,
        )

        start = self._start_timer()
        self._predictor = TabularPredictor(
            label=self.target_column,
            problem_type=ag_problem,
            eval_metric=metric,
            path=path,
        ).fit(
            train_data=train_df,
            tuning_data=tuning_data,
            time_limit=self.time_limit,
            presets=self.presets,
            verbosity=0,
        )
        elapsed = self._stop_timer(start)
        self._is_fitted = True
        log.info("AutoGluon fit complete in %.1fs", elapsed)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._predictor.predict(X).to_numpy()

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        self._check_fitted()
        if self.problem_type == ProblemType.REGRESSION:
            return None
        try:
            return self._predictor.predict_proba(X).to_numpy()
        except Exception:
            return None

    def get_params(self) -> dict:
        base = super().get_params()
        base.update({"presets": self.presets, "backend": "autogluon"})
        return base

    def leaderboard(self) -> Optional[pd.DataFrame]:
        """Return the AutoGluon model leaderboard (if available)."""
        if self._predictor is None:
            return None
        try:
            return self._predictor.leaderboard(silent=True)
        except Exception:
            return None

    def __del__(self) -> None:
        """Clean up temp directory if not keeping model."""
        if not self.keep_model_dir and self._tmp_dir and os.path.exists(self._tmp_dir):
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
