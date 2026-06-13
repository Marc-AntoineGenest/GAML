"""
EnsembleModel — combines multiple fitted AutoML models into one predictor.

Classification  : soft voting (average predicted probabilities, then argmax).
Regression      : simple average of predictions.

This is the final model returned by AutoMLPipeline when ensemble mode is
enabled.  It satisfies the full BaseAutoML interface, so the rest of the
pipeline (save/load, score, predict_proba, feature_importances_) works without
any changes.

Design notes
------------
- Each member has already been independently fitted on the dev set (train+val)
  by AutoMLPipeline before being passed here.
- Weights default to equal (1/N per model).  Callers may supply custom weights,
  e.g. proportional to each member's CV fitness.
- The ensemble is intentionally lightweight — no meta-learner training here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from genetic_automl.core.base_automl import BaseAutoML
from genetic_automl.core.problem import ProblemType
from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)


class EnsembleModel(BaseAutoML):
    """
    Soft-voting / averaging ensemble of pre-fitted AutoML models.

    Parameters
    ----------
    members : list[BaseAutoML]
        Already-fitted model instances.
    weights : list[float] | None
        Per-model weights.  None = uniform weights.
    problem_type : ProblemType
    target_column : str
    """

    def __init__(
        self,
        members: list[BaseAutoML],
        problem_type: ProblemType,
        target_column: str,
        weights: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        if not members:
            raise ValueError("EnsembleModel requires at least one member model.")

        super().__init__(problem_type, target_column, **kwargs)
        self.members = members
        self.weights = self._normalise_weights(weights, len(members))
        # The ensemble is ready to predict immediately — members are pre-fitted.
        self._is_fitted = True


    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> EnsembleModel:
        """
        No-op: members were fitted before being passed to this class.
        Satisfies the BaseAutoML contract.
        """
        log.info(
            "EnsembleModel: %d pre-fitted members (weights=%s) — no refit needed.",
            len(self.members),
            [f"{w:.3f}" for w in self.weights],
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return hard predictions (argmax of averaged probabilities / averaged values)."""
        self._check_fitted()
        if self.problem_type == ProblemType.REGRESSION:
            return self._averaged_predictions(X)
        proba = self.predict_proba(X)
        if proba is not None:
            return self._classes_from_proba(proba)
        return self._majority_vote(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        """
        Return weighted average of per-member class probability matrices.
        Returns None if no member supports predict_proba.
        """
        self._check_fitted()
        if self.problem_type == ProblemType.REGRESSION:
            return None

        proba_list = []
        used_weights = []
        for model, w in zip(self.members, self.weights):
            p = model.predict_proba(X)
            if p is not None:
                proba_list.append(p * w)
                used_weights.append(w)

        if not proba_list:
            return None

        # Re-normalise weights in case some members returned None
        total_w = sum(used_weights)
        return sum(proba_list) / total_w  # type: ignore[return-value]

    @property
    def feature_importances_(self):
        """
        Return weighted-average feature importances across members that expose them.
        Returns None if no member supports feature_importances_.
        """
        importances = []
        weights_used = []
        for model, w in zip(self.members, self.weights):
            fi = getattr(model, "feature_importances_", None)
            if fi is not None:
                importances.append(np.asarray(fi) * w)
                weights_used.append(w)

        if not importances:
            return None

        total_w = sum(weights_used)
        averaged = sum(importances) / total_w  # type: ignore[return-value]
        # Normalise so they sum to 1
        total = averaged.sum()
        return averaged / total if total > 0 else averaged

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            "n_members": len(self.members),
            "weights": self.weights,
            "member_types": [type(m).__name__ for m in self.members],
        }

    def __repr__(self) -> str:
        types = [type(m).__name__ for m in self.members]
        return (
            f"EnsembleModel(members={len(self.members)}, "
            f"types={types}, "
            f"problem={self.problem_type.value})"
        )


    def _averaged_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted average of regression predictions."""
        preds = np.stack(
            [m.predict(X) * w for m, w in zip(self.members, self.weights)],
            axis=0,
        )
        return preds.sum(axis=0)

    def _majority_vote(self, X: pd.DataFrame) -> np.ndarray:
        """Hard majority vote fallback when predict_proba is unavailable."""
        all_preds = np.stack(
            [m.predict(X) for m in self.members], axis=0
        )  # (n_members, n_samples)
        # Weighted majority: most common weighted label per sample
        results = []
        for sample_preds in all_preds.T:
            votes: dict = {}
            for pred, w in zip(sample_preds, self.weights):
                votes[pred] = votes.get(pred, 0.0) + w
            results.append(max(votes, key=votes.__getitem__))
        return np.array(results)

    def _classes_from_proba(self, proba: np.ndarray) -> np.ndarray:
        """Convert a probability matrix to hard class labels."""
        indices = np.argmax(proba, axis=1)
        # Recover original class labels from the first member that exposes them
        for member in self.members:
            classes = getattr(getattr(member, "_estimator", None), "classes_", None)
            if classes is not None:
                return classes[indices]
        return indices

    @staticmethod
    def _normalise_weights(weights: list[float] | None, n: int) -> list[float]:
        if weights is None:
            return [1.0 / n] * n
        if len(weights) != n:
            raise ValueError(
                f"len(weights)={len(weights)} does not match n_members={n}."
            )
        total = sum(weights)
        if total <= 0:
            # All weights non-positive (e.g. regression negated-MSE fitness).
            # Fall back to uniform weights instead of crashing.
            return [1.0 / n] * n
        return [w / total for w in weights]
