"""
PreprocessingPipeline
---------------------
Orchestrates all preprocessing steps in the correct order:

  1. NumericImputer         — fill NaN before outlier detection
  2. OutlierHandler         — clean numeric data, before scaling
  3. CorrelationFilter      — reliable stats after imputation + outlier treatment
  4. CategoricalEncoder     — encode before scaling
  5. DistributionTransform  — normalize skewness before scaling
  6. Scaler                 — all columns numeric, distributions shaped
  7. MissingIndicator       — add binary flags using pre-imputation mask
  8. FeatureSelector        — on fully preprocessed data
  9. ImbalanceHandler       — train only, on final feature matrix

Zero-leakage guarantee: all steps fit on training data only.
ImbalanceHandler is never applied to val/test data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from genetic_automl.core.problem import ProblemType
from genetic_automl.preprocessing.categorical_encoder import CategoricalEncoder
from genetic_automl.preprocessing.correlation_filter import CorrelationFilter
from genetic_automl.preprocessing.distribution_transform import DistributionTransform
from genetic_automl.preprocessing.feature_selector import FeatureSelector
from genetic_automl.preprocessing.imbalance_handler import ImbalanceHandler
from genetic_automl.preprocessing.missing_indicator import MissingIndicator
from genetic_automl.preprocessing.numeric_imputer import NumericImputer
from genetic_automl.preprocessing.outlier_handler import OutlierHandler
from genetic_automl.preprocessing.scaler import Scaler
from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class PreprocessingConfig:
    numeric_imputer: str = "median"
    outlier_method: str = "none"
    outlier_threshold: float = 1.5
    outlier_action: str = "clip"
    correlation_threshold: Optional[float] = 0.95
    categorical_encoder: str = "onehot"
    distribution_transform: str = "none"
    scaler: str = "standard"
    missing_indicator: bool = False
    feature_selection_method: str = "none"
    feature_selection_k: float = 0.75
    imbalance_method: str = "none"

    @classmethod
    def from_genes(cls, genes: Dict[str, Any]) -> "PreprocessingConfig":
        """Build from a chromosome gene dict (keys may be a superset)."""
        fields = set(cls.__dataclass_fields__.keys())  # type: ignore
        return cls(**{k: v for k, v in genes.items() if k in fields})


class PreprocessingPipeline:
    """
    End-to-end preprocessing pipeline with correct step ordering.

    Parameters
    ----------
    config : PreprocessingConfig
    problem_type : ProblemType
    random_seed : int
    """

    def __init__(
        self,
        config: PreprocessingConfig,
        problem_type: ProblemType = ProblemType.CLASSIFICATION,
        random_seed: int = 42,
    ) -> None:
        self.config = config
        self.problem_type = problem_type
        self.random_seed = random_seed
        self._is_fitted = False

        self._numeric_imputer       = NumericImputer(strategy=config.numeric_imputer)
        self._outlier_handler       = OutlierHandler(
            method=config.outlier_method,
            threshold=config.outlier_threshold,
            action=config.outlier_action,
        )
        self._correlation_filter    = CorrelationFilter(threshold=config.correlation_threshold)
        self._categorical_encoder   = CategoricalEncoder(strategy=config.categorical_encoder)
        self._distribution_transform = DistributionTransform(method=config.distribution_transform)
        self._scaler                = Scaler(method=config.scaler)
        self._missing_indicator     = MissingIndicator(enabled=config.missing_indicator)
        self._feature_selector      = FeatureSelector(
            method=config.feature_selection_method,
            keep_k=config.feature_selection_k,
            problem_type_str=problem_type.value,
            random_seed=random_seed,
        )
        self._imbalance_handler     = ImbalanceHandler(
            method=config.imbalance_method,
            random_seed=random_seed,
        )

    # ------------------------------------------------------------------

    def fit_transform_train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit all steps on training data and return (X_transformed, y_transformed).
        ImbalanceHandler resampling is only applied here — never on val/test.
        """
        log.info("PreprocessingPipeline: fitting on %d × %d", *X_train.shape)
        X, y = X_train.copy(), y_train.copy()

        # Capture missingness mask before imputation fills NaNs
        self._missing_indicator.fit(X)
        raw_mask = X.isnull()

        X = self._numeric_imputer.fit_transform(X, y)
        X = self._outlier_handler.fit_transform(X, y)
        X = self._correlation_filter.fit_transform(X, y)
        X = self._categorical_encoder.fit_transform(X, y if self.config.categorical_encoder == "target" else None)
        X = self._distribution_transform.fit_transform(X, y)
        X = self._scaler.fit_transform(X, y)
        X = self._missing_indicator.transform_with_mask(X, raw_mask)
        self._feature_selector.fit(X, y)
        X = self._feature_selector.transform(X)

        if self.problem_type != ProblemType.REGRESSION:
            X, y = self._imbalance_handler.fit_resample(X, y)
        else:
            self._imbalance_handler.fit(X, y)

        self._is_fitted = True
        log.info("PreprocessingPipeline: output %d × %d", *X.shape)
        return X, y

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted transformations to val/test data. No fitting occurs here."""
        if not self._is_fitted:
            raise RuntimeError("Call fit_transform_train() before transform().")
        X = X.copy()
        raw_mask = X.isnull()
        X = self._numeric_imputer.transform(X)
        X = self._outlier_handler.transform(X)
        X = self._correlation_filter.transform(X)
        X = self._categorical_encoder.transform(X)
        X = self._distribution_transform.transform(X)
        X = self._scaler.transform(X)
        X = self._missing_indicator.transform_with_mask(X, raw_mask)
        X = self._feature_selector.transform(X)
        return X

    def sample_weights(self, y: pd.Series):
        """Return per-sample weights if imbalance_method='class_weight', else None."""
        return self._imbalance_handler.sample_weights(y)

    def summary(self) -> Dict[str, Any]:
        return {
            "correlation_dropped": self._correlation_filter.dropped_features,
            "selected_features": self._feature_selector.selected_features,
            "missing_indicator_cols": self._missing_indicator.indicator_columns,
            "config": {
                "numeric_imputer": self.config.numeric_imputer,
                "outlier_method": self.config.outlier_method,
                "outlier_threshold": self.config.outlier_threshold,
                "correlation_threshold": self.config.correlation_threshold,
                "categorical_encoder": self.config.categorical_encoder,
                "distribution_transform": self.config.distribution_transform,
                "scaler": self.config.scaler,
                "missing_indicator": self.config.missing_indicator,
                "feature_selection_method": self.config.feature_selection_method,
                "feature_selection_k": self.config.feature_selection_k,
                "imbalance_method": self.config.imbalance_method,
            },
        }
