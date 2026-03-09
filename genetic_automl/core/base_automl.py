"""
BaseAutoML — abstract contract every AutoML backend must implement.

Adding a new backend:
1. Subclass BaseAutoML
2. Implement fit(), predict(), predict_proba() (optional), score()
3. Register it in automl/__init__.py
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from genetic_automl.core.problem import ProblemType, compute_metric, get_default_metric
from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)


class BaseAutoML(ABC):
    """
    Abstract base class for AutoML backends.

    Every implementation must provide at minimum:
        fit(X_train, y_train, **kwargs)
        predict(X) -> np.ndarray
        score(X, y, metric=None) -> float
    """

    def __init__(
        self,
        problem_type: ProblemType,
        target_column: str,
        time_limit: int = 60,
        random_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        self.problem_type = problem_type
        self.target_column = target_column
        self.time_limit = time_limit
        self.random_seed = random_seed
        self.extra_kwargs = kwargs

        self._is_fitted: bool = False
        self._fit_duration: float = 0.0
        self._feature_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Core interface (must override)
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "BaseAutoML":
        """Train the model. Returns self for chaining."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return hard predictions (class labels or regression values)."""
        ...

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Return class probabilities. Not mandatory; returns None by default.
        Override for classification backends that support it.
        """
        return None

    # ------------------------------------------------------------------
    # Scoring (concrete — calls predict + metric registry)
    # ------------------------------------------------------------------

    def score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metric: Optional[str] = None,
    ) -> float:
        """Evaluate the model on *X* / *y* using *metric*."""
        self._check_fitted()
        if metric is None:
            metric = get_default_metric(self.problem_type)
        y_pred = self.predict(X)
        value = compute_metric(metric, y.values, y_pred)
        log.debug("score | metric=%s | value=%.6f", metric, value)
        return value

    # ------------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def fit_duration(self) -> float:
        return self._fit_duration

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

    def _start_timer(self) -> float:
        return time.perf_counter()

    def _stop_timer(self, start: float) -> float:
        elapsed = time.perf_counter() - start
        self._fit_duration = elapsed
        return elapsed

    # ------------------------------------------------------------------
    # Serialization hooks (optional override)
    # ------------------------------------------------------------------

    def get_params(self) -> Dict[str, Any]:
        """Return hyper-parameters for logging / chromosome encoding."""
        return {
            "problem_type": self.problem_type.value,
            "time_limit": self.time_limit,
            "random_seed": self.random_seed,
            **self.extra_kwargs,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"problem={self.problem_type.value}, "
            f"fitted={self._is_fitted})"
        )
