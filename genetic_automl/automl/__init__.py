"""AutoML backend factory."""

from __future__ import annotations

from typing import Any

from genetic_automl.core.base_automl import BaseAutoML
from genetic_automl.core.problem import ProblemType


def build_automl(
    backend: str,
    problem_type: ProblemType,
    target_column: str,
    time_limit: int = 60,
    random_seed: int = 42,
    **kwargs: Any,
) -> BaseAutoML:
    """
    Instantiate an AutoML backend by name.

    Parameters
    ----------
    backend : str
        'autogluon' | 'sklearn'
    """
    backend = backend.lower()
    if backend == "autogluon":
        from genetic_automl.automl.autogluon_model import AutoGluonModel
        return AutoGluonModel(
            problem_type=problem_type,
            target_column=target_column,
            time_limit=time_limit,
            random_seed=random_seed,
            **kwargs,
        )
    elif backend == "sklearn":
        from genetic_automl.automl.sklearn_model import SklearnModel
        return SklearnModel(
            problem_type=problem_type,
            target_column=target_column,
            time_limit=time_limit,
            random_seed=random_seed,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown AutoML backend '{backend}'. Choose from: autogluon, sklearn"
        )
