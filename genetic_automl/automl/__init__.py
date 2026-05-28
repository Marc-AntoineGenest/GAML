"""AutoML backend factory."""

from __future__ import annotations

from typing import Any

from genetic_automl.core.base_automl import BaseAutoML
from genetic_automl.core.problem import ProblemType

# Models that map to model_type gene values within the 'sklearn' backend.
# Keeping them under one backend avoids proliferating top-level backend names
# while still letting the GA freely search across model families.
_SKLEARN_MODEL_TYPE_MAP = {
    "gbm": "_build_sklearn_gbm",
    "lgbm": "_build_lgbm",
    "xgb": "_build_xgb",
}


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
    **kwargs
        Forwarded to the backend constructor.  For the 'sklearn' backend,
        an optional ``model_type`` kwarg selects the underlying algorithm:
          'gbm'   — sklearn GradientBoosting (original default)
          'lgbm'  — LightGBM  (recommended; requires lightgbm)
          'xgb'   — XGBoost   (recommended; requires xgboost)
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
        model_type = kwargs.pop("model_type", "gbm")
        if model_type == "lgbm":
            from genetic_automl.automl.lgbm_model import LGBMModel
            return LGBMModel(
                problem_type=problem_type,
                target_column=target_column,
                time_limit=time_limit,
                random_seed=random_seed,
                **kwargs,
            )
        elif model_type == "xgb":
            from genetic_automl.automl.xgb_model import XGBModel
            return XGBModel(
                problem_type=problem_type,
                target_column=target_column,
                time_limit=time_limit,
                random_seed=random_seed,
                **kwargs,
            )
        else:
            # Default: original sklearn GradientBoosting
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
