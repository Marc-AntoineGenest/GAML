"""
Random Forest backend — wraps RandomForestClassifier / RandomForestRegressor.

Why RandomForest in the model zoo?
- Independently strong: competitive on many tabular tasks, especially with
  moderate feature counts and noisy data.
- Parallel tree construction — faster wall-clock training than serial GBM
  on multi-core machines.
- Naturally provides prediction variance via per-tree disagreement, which
  the surrogate module exploits for uncertainty-aware fitness prediction.
- No learning_rate hyperparameter — the GA search space is slightly
  different from GBM families, which helps population diversity.

As surrogate:
  RandomForest is the default surrogate model for the GA because:
  - Trains in milliseconds on the small fitness-history dataset.
  - std_pred = std of per-tree predictions gives a free uncertainty signal,
    allowing the surrogate to hedge on chromosomes it hasn't seen before.
  - No hyperparameter tuning needed (n_estimators=100 is fine for surrogacy).
  - Zero additional dependencies — already in scikit-learn.

All preprocessing is handled upstream by PreprocessingPipeline.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from genetic_automl.core.base_automl import BaseAutoML
from genetic_automl.core.problem import ProblemType
from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)


class RandomForestModel(BaseAutoML):
    """
    Random Forest wrapper.

    Parameters
    ----------
    n_estimators : int
        Number of trees.
    max_depth : int | None
        Maximum tree depth. None = grow until leaves are pure.
    min_samples_leaf : int
        Minimum samples per leaf — key regularisation knob for RF.
    max_features : str | float
        Feature subset per split: 'sqrt' (default clf), 'log2', float fraction.
    """

    def __init__(
        self,
        problem_type: ProblemType,
        target_column: str,
        time_limit: int = 60,
        random_seed: int = 42,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        # Accept (and ignore) learning_rate so the gene schema stays uniform
        # across all sklearn model_types — the GA may send it during crossover.
        learning_rate: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(problem_type, target_column, time_limit, random_seed, **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self._estimator = None


    def _build_estimator(self):
        params = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_seed,
            n_jobs=-1,
        )
        if self.problem_type in (ProblemType.CLASSIFICATION, ProblemType.MULTI_OBJECTIVE):
            return RandomForestClassifier(**params)
        return RandomForestRegressor(**params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "RandomForestModel":
        # X_val / y_val accepted but ignored — RF has no native early stopping.
        log.info(
            "RandomForestModel fit | trees=%d | depth=%s | samples=%d | features=%d",
            self.n_estimators,
            str(self.max_depth) if self.max_depth else "unlimited",
            len(y_train),
            X_train.shape[1],
        )
        start = self._start_timer()
        self._estimator = self._build_estimator()
        self._estimator.fit(X_train.values, y_train.values)
        self._feature_names = list(X_train.columns)
        self._is_fitted = True
        log.info("RandomForestModel fit complete in %.2fs", self._stop_timer(start))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._estimator.predict(X.values)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        self._check_fitted()
        if self.problem_type == ProblemType.REGRESSION:
            return None
        return self._estimator.predict_proba(X.values)

    def predict_with_std(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (mean_prediction, std_prediction) using per-tree disagreement.

        This is used by the surrogate module to quantify prediction uncertainty:
        high std → the surrogate has seen few similar chromosomes → don't skip.

        For classification, returns mean and std of the positive-class probability
        across trees. For regression, returns mean and std of the raw predictions.
        """
        self._check_fitted()
        # Collect per-tree predictions for each sample
        if self.problem_type == ProblemType.REGRESSION:
            tree_preds = np.stack(
                [tree.predict(X.values) for tree in self._estimator.estimators_],
                axis=0,
            )  # (n_trees, n_samples)
        else:
            # Use positive-class probability (index 1 for binary; mean across
            # classes for multiclass — surrogate only needs a scalar signal)
            tree_preds = np.stack(
                [tree.predict_proba(X.values)[:, -1]
                 for tree in self._estimator.estimators_],
                axis=0,
            )  # (n_trees, n_samples)

        return tree_preds.mean(axis=0), tree_preds.std(axis=0)

    @property
    def feature_importances_(self):
        """Return MDI feature importances, or None."""
        self._check_fitted()
        if hasattr(self._estimator, "feature_importances_"):
            return self._estimator.feature_importances_
        return None

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            "model_type": "rf",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
        }
