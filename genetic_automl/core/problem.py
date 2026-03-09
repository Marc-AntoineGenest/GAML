"""
Problem type enumeration and metric helpers.

Fitness direction convention
-----------------------------
- Classification : maximize  (F1-macro by default)
- Regression     : minimize  (MSE by default, stored as negative for GA)
- MultiObjective : depends on per-objective direction
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


class ProblemType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    MULTI_OBJECTIVE = "multi_objective"


class MetricDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

_METRIC_REGISTRY: Dict[str, Tuple[Any, MetricDirection]] = {
    # Classification
    "f1_macro": (
        lambda y, yp: f1_score(y, yp, average="macro", zero_division=0),
        MetricDirection.MAXIMIZE,
    ),
    "f1_weighted": (
        lambda y, yp: f1_score(y, yp, average="weighted", zero_division=0),
        MetricDirection.MAXIMIZE,
    ),
    "accuracy": (accuracy_score, MetricDirection.MAXIMIZE),
    "roc_auc": (
        lambda y, yp: roc_auc_score(y, yp, multi_class="ovr", average="macro"),
        MetricDirection.MAXIMIZE,
    ),
    # Regression
    "mse": (mean_squared_error, MetricDirection.MINIMIZE),
    "mae": (mean_absolute_error, MetricDirection.MINIMIZE),
    "r2": (r2_score, MetricDirection.MAXIMIZE),
}

_DEFAULT_METRIC: Dict[ProblemType, str] = {
    ProblemType.CLASSIFICATION: "f1_macro",
    ProblemType.REGRESSION: "mse",
    ProblemType.MULTI_OBJECTIVE: "mse",  # overridden per objective
}


def get_default_metric(problem_type: ProblemType) -> str:
    return _DEFAULT_METRIC[problem_type]


def compute_metric(
    metric_name: str,
    y_true: Any,
    y_pred: Any,
) -> float:
    """Compute a registered metric. Returns a scalar float."""
    if metric_name not in _METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric '{metric_name}'. "
            f"Available: {list(_METRIC_REGISTRY.keys())}"
        )
    fn, _ = _METRIC_REGISTRY[metric_name]
    return float(fn(y_true, y_pred))


def metric_direction(metric_name: str) -> MetricDirection:
    """Return whether this metric should be maximized or minimized."""
    _, direction = _METRIC_REGISTRY[metric_name]
    return direction


def fitness_sign(metric_name: str) -> int:
    """
    Return +1 if higher is better, -1 if lower is better.
    The GA always *maximizes* fitness, so callers multiply raw metric by this sign.
    """
    return 1 if metric_direction(metric_name) == MetricDirection.MAXIMIZE else -1


# ---------------------------------------------------------------------------
# Multi-objective helpers (Pareto front)
# ---------------------------------------------------------------------------

def pareto_front(scores: List[List[float]]) -> List[int]:
    """
    Return indices of non-dominated solutions.
    All scores are assumed to be in the *maximize* direction
    (negate minimization metrics before calling this).
    """
    scores_arr = np.array(scores)
    n = len(scores_arr)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(scores_arr[j] >= scores_arr[i]) and np.any(
                scores_arr[j] > scores_arr[i]
            ):
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]
