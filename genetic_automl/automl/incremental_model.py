"""
IncrementalModel — Online / incremental learning via sklearn partial_fit.

Problem solved
--------------
Standard GAML trains once on a fixed dataset.  In production, new labelled
batches arrive continuously (daily transactions, sensor readings, user events).
Retraining from scratch on every batch is expensive; fine-tuning the existing
model on new data is far cheaper and often nearly as accurate.

sklearn provides `partial_fit()` on several estimators — this module wraps
them in the same BaseAutoML interface so the rest of GAML (preprocessing,
reporting, calibration) is unaffected.

Supported estimators
--------------------
Classification:
  SGDClassifier (logistic / hinge / modified_huber loss)
  PassiveAggressiveClassifier
  MultinomialNB  (non-negative features only)
  BernoulliNB

Regression:
  SGDRegressor
  PassiveAggressiveRegressor

Usage via pipeline
------------------
  pipeline.partial_fit(new_df)           # update model on new batch
  pipeline.partial_fit(new_df, epochs=3) # multiple passes over new batch

Standalone
----------
  from genetic_automl.automl.incremental_model import IncrementalModel
  model = IncrementalModel(
      problem_type=ProblemType.CLASSIFICATION,
      target_column="label",
      model_type="sgd",
      loss="log_loss",   # logistic regression
  )
  model.fit(X_train, y_train)           # initial fit
  model.partial_fit(X_new, y_new)       # online update
  preds = model.predict(X_test)

Design notes
------------
- `fit()` calls `partial_fit()` in minibatches — no need for special first-call
  handling; sklearn's partial_fit initialises on first call.
- `classes_` is stored from the first fit and reused on all subsequent updates
  (required by sklearn's partial_fit API for classifiers).
- The wrapper is fully compatible with the pipeline's save/load (joblib), SHAP
  (TreeExplainer won't work but KernelExplainer will — skipped gracefully), and
  drift detection.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from genetic_automl.core.base_automl import BaseAutoML
from genetic_automl.core.problem import ProblemType, compute_metric
from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)

# Mapping of model_type → (ClassificationClass, RegressionClass, default_kwargs)
_INCREMENTAL_REGISTRY: Dict[str, dict] = {
    "sgd": {
        "clf": "sklearn.linear_model.SGDClassifier",
        "reg": "sklearn.linear_model.SGDRegressor",
        "clf_kwargs": {"loss": "log_loss", "max_iter": 1, "tol": None, "warm_start": True},
        "reg_kwargs": {"loss": "squared_error", "max_iter": 1, "tol": None, "warm_start": True},
    },
    "passive_aggressive": {
        # PassiveAggressiveClassifier deprecated in sklearn 1.8;
        # use SGDClassifier with PA hyperparameters instead.
        "clf": "sklearn.linear_model.SGDClassifier",
        "reg": "sklearn.linear_model.SGDRegressor",
        "clf_kwargs": {"loss": "hinge", "penalty": None,
                       "learning_rate": "pa1", "eta0": 1.0,
                       "max_iter": 1, "tol": None, "warm_start": True},
        "reg_kwargs": {"loss": "squared_error",
                       "max_iter": 1, "tol": None, "warm_start": True},
    },
    "bernoulli_nb": {
        "clf": "sklearn.naive_bayes.BernoulliNB",
        "reg": None,   # no regression variant
        "clf_kwargs": {},
        "reg_kwargs": {},
    },
}


def _import_class(dotted_path: str) -> type:
    parts = dotted_path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(parts[0])
    return getattr(mod, parts[1])


class IncrementalModel(BaseAutoML):
    """
    Online-learning model that supports incremental updates via partial_fit().

    Parameters
    ----------
    problem_type : ProblemType
    target_column : str
    model_type : str
        One of: "sgd" (default), "passive_aggressive", "bernoulli_nb".
    random_seed : int
    **model_kwargs
        Forwarded to the underlying sklearn estimator constructor.
        E.g. loss="modified_huber" for SGDClassifier.
    """

    def __init__(
        self,
        problem_type: ProblemType,
        target_column: str,
        model_type: str = "sgd",
        random_seed: int = 42,
        **model_kwargs,
    ) -> None:
        self.problem_type = problem_type
        self.target_column = target_column
        self.model_type = model_type
        self.random_seed = random_seed
        self.model_kwargs = model_kwargs

        self._estimator = None
        self._classes: Optional[np.ndarray] = None
        self._feature_names: Optional[List[str]] = None
        self._n_batches_seen: int = 0
        self._fit_start: float = 0.0
        self._fit_duration: float = 0.0


    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        batch_size: int = 1024,
    ) -> "IncrementalModel":
        """
        Initial fit via minibatch partial_fit passes.

        Parameters
        ----------
        batch_size : int
            Rows per minibatch (default 1024).  Smaller = lower memory,
            more gradient steps; larger = faster epoch.
        """
        self._estimator = self._build_estimator()
        self._feature_names = list(X_train.columns)
        self._fit_start = time.perf_counter()

        X_arr = X_train.values.astype(float)
        y_arr = y_train.values

        if self.problem_type != ProblemType.REGRESSION:
            self._classes = np.unique(y_arr)

        n = len(X_arr)
        for start in range(0, n, batch_size):
            X_batch = X_arr[start: start + batch_size]
            y_batch = y_arr[start: start + batch_size]
            self._partial_fit_one(X_batch, y_batch)

        self._n_batches_seen += 1
        self._fit_duration = time.perf_counter() - self._fit_start
        log.info(
            "IncrementalModel fit | model=%s | rows=%d | batches=%d | %.2fs",
            self.model_type, n, -(-n // batch_size), self._fit_duration,
        )
        return self

    def partial_fit(
        self,
        X_new: pd.DataFrame,
        y_new: pd.Series,
        epochs: int = 1,
    ) -> "IncrementalModel":
        """
        Update the model on a new data batch without full retraining.

        Parameters
        ----------
        X_new : pd.DataFrame
        y_new : pd.Series
        epochs : int
            Number of passes over the new batch (default 1).
            More epochs = stronger adaptation, but risk of catastrophic
            forgetting of old patterns.  1-3 is typical.
        """
        if self._estimator is None:
            raise RuntimeError("Call fit() before partial_fit().")

        X_arr = X_new.values.astype(float)
        y_arr = y_new.values

        for epoch in range(epochs):
            self._partial_fit_one(X_arr, y_arr)

        self._n_batches_seen += 1
        log.info(
            "IncrementalModel partial_fit | batch=%d | rows=%d | epochs=%d",
            self._n_batches_seen, len(X_new), epochs,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._estimator.predict(X.values.astype(float))

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        self._check_fitted()
        if hasattr(self._estimator, "predict_proba"):
            return self._estimator.predict_proba(X.values.astype(float))
        if hasattr(self._estimator, "decision_function"):
            # Calibrate decision scores into [0,1] via sigmoid
            scores = self._estimator.decision_function(X.values.astype(float))
            proba_pos = 1 / (1 + np.exp(-scores))
            return np.column_stack([1 - proba_pos, proba_pos])
        return None

    def score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metric: Optional[str] = None,
    ) -> float:
        preds = self.predict(X)
        proba = self.predict_proba(X) if metric in ("roc_auc",) else None
        y_score = proba[:, 1] if (proba is not None and proba.ndim == 2) else preds
        return compute_metric(metric or "accuracy", y.values, y_score)

    @property
    def fit_duration(self) -> float:
        return self._fit_duration

    @property
    def n_batches_seen(self) -> int:
        """Number of fit/partial_fit calls completed."""
        return self._n_batches_seen


    def _build_estimator(self) -> Any:
        """Instantiate the underlying sklearn estimator."""
        spec = _INCREMENTAL_REGISTRY.get(self.model_type)
        if spec is None:
            raise ValueError(
                f"Unknown model_type={self.model_type!r}. "
                f"Choose from: {list(_INCREMENTAL_REGISTRY)}"
            )

        if self.problem_type == ProblemType.REGRESSION:
            if spec["reg"] is None:
                raise ValueError(
                    f"model_type={self.model_type!r} does not support regression."
                )
            cls = _import_class(spec["reg"])
            kwargs = {**spec["reg_kwargs"], **self.model_kwargs}
        else:
            cls = _import_class(spec["clf"])
            kwargs = {**spec["clf_kwargs"], **self.model_kwargs}

        if "random_state" in cls.__init__.__code__.co_varnames:
            kwargs.setdefault("random_state", self.random_seed)

        return cls(**kwargs)

    def _partial_fit_one(self, X: np.ndarray, y: np.ndarray) -> None:
        """Single partial_fit call with correct classes argument."""
        if self.problem_type != ProblemType.REGRESSION and self._classes is not None:
            self._estimator.partial_fit(X, y, classes=self._classes)
        else:
            self._estimator.partial_fit(X, y)

    def _check_fitted(self) -> None:
        if self._estimator is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
