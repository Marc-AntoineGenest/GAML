"""
XGBoost backend — wraps XGBClassifier / XGBRegressor.

Why XGBoost?
- Level-wise tree growth with regularisation (L1 + L2) — often more robust
  than LightGBM on smaller datasets.
- Native early stopping when a validation set is supplied.
- Excellent GPU support via tree_method='hist' (CPU histogram also fast).

All preprocessing is handled upstream by PreprocessingPipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from genetic_automl.core.base_automl import BaseAutoML
from genetic_automl.core.problem import ProblemType
from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)

_EARLY_STOPPING_ROUNDS = 50


class XGBModel(BaseAutoML):
    """
    XGBoost wrapper with optional early stopping.

    Parameters
    ----------
    n_estimators : int
        Maximum boosting rounds.
    max_depth : int
        Maximum tree depth.
    learning_rate : float
    subsample : float
        Row subsampling fraction per tree.
    colsample_bytree : float
        Column subsampling fraction per tree.
    reg_alpha : float
        L1 regularisation term.
    reg_lambda : float
        L2 regularisation term.
    """

    def __init__(
        self,
        problem_type: ProblemType,
        target_column: str,
        time_limit: int = 60,
        random_seed: int = 42,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(problem_type, target_column, time_limit, random_seed, **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self._estimator = None
        self._label_encoder = None  # needed when y contains non-integer class labels


    def _build_estimator(self):
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for the 'xgb' model type. "
                "Install it with:  pip install xgboost"
            ) from exc

        params = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_seed,
            tree_method="hist",  # fast on CPU; auto-selects GPU when available
            verbosity=0,
            n_jobs=-1,
        )
        if self.problem_type in (ProblemType.CLASSIFICATION, ProblemType.MULTI_OBJECTIVE):
            return XGBClassifier(**params)
        return XGBRegressor(**params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> XGBModel:
        log.info(
            "XGBModel fit | n_estimators=%d | depth=%d | lr=%.3f | "
            "samples=%d | features=%d | early_stopping=%s",
            self.n_estimators, self.max_depth, self.learning_rate,
            len(y_train), X_train.shape[1],
            "yes" if X_val is not None else "no",
        )
        start = self._start_timer()
        self._estimator = self._build_estimator()

        # XGBoost requires contiguous integer class labels starting from 0.
        # Encode y and store the mapping so predict() can decode it back.
        y_train_enc, y_val_enc = self._encode_labels(y_train, y_val)

        fit_kwargs: dict = {}
        if X_val is not None and y_val_enc is not None:
            fit_kwargs["eval_set"] = [(X_val.values, y_val_enc)]
            fit_kwargs["verbose"] = False
            self._estimator.set_params(early_stopping_rounds=_EARLY_STOPPING_ROUNDS)

        self._estimator.fit(X_train.values, y_train_enc, **fit_kwargs)
        self._feature_names = list(X_train.columns)
        self._is_fitted = True
        log.info("XGBModel fit complete in %.2fs", self._stop_timer(start))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        raw = self._estimator.predict(X.values)
        return self._decode_labels(raw)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
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
            "model_type": "xgb",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
        }


    def _encode_labels(self, y_train: pd.Series, y_val=None):
        """
        XGBoost requires 0-indexed integer labels.  Encode once at fit time
        and store the classes array for decode at predict time.
        """
        if self.problem_type == ProblemType.REGRESSION:
            self._label_encoder = None
            return y_train.values, (y_val.values if y_val is not None else None)

        classes = np.unique(y_train)
        # Already 0-indexed integers — no encoding needed
        if np.array_equal(classes, np.arange(len(classes))):
            self._label_encoder = None
            return y_train.values, (y_val.values if y_val is not None else None)

        self._label_encoder = {cls: idx for idx, cls in enumerate(classes)}
        self._label_decoder = {idx: cls for cls, idx in self._label_encoder.items()}
        y_enc = np.array([self._label_encoder[v] for v in y_train])
        y_val_enc = (
            np.array([self._label_encoder.get(v, 0) for v in y_val])
            if y_val is not None else None
        )
        return y_enc, y_val_enc

    def _decode_labels(self, y_pred: np.ndarray) -> np.ndarray:
        if self._label_encoder is None:
            return y_pred
        return np.array([self._label_decoder.get(v, v) for v in y_pred])
