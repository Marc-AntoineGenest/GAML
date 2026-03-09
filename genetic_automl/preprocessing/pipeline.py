"""
PreprocessingPipeline
---------------------
Orchestrates all preprocessing steps in the scientifically correct order.

CORRECT Step Order (enforced here):
  1. NumericImputer         — FIRST: outlier detection needs clean data, NaN breaks IQR/IsolationForest
  2. OutlierHandler         — SECOND: on clean numeric data, before scaling distorts distances
  3. CorrelationFilter      — THIRD: after imputation+outlier treatment, correlation stats are reliable
  4. CategoricalEncoder     — FOURTH: encode before scaling (scaling strings is nonsensical)
  5. DistributionTransform  — FIFTH: Yeo-Johnson/Box-Cox to normalize skewed distributions (NEW)
  6. Scaler                 — SIXTH: scale all numeric columns uniformly after encoding
  7. MissingIndicator       — creates binary flags for originally-missing columns (NEW)
  8. FeatureSelector        — LAST before imbalance: mutual_info/RFE need valid numeric input
  9. ImbalanceHandler       — ALWAYS LAST, train only, on final preprocessed X

Zero-leakage guarantee:
  - All steps fit on training data only, transform is applied to val/test
  - ImbalanceHandler.fit_resample() is called only on training data
  - TargetEncoder uses cross-val encoding internally
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Config dataclass (mirrors chromosome genes)
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingConfig:
    # Step 1 — Numeric imputation (FIRST — needed before outlier detection)
    numeric_imputer: str = "median"          # mean|median|knn|iterative|constant

    # Step 2 — Outlier (on clean numeric data)
    outlier_method: str = "none"             # none|iqr|zscore|isolation_forest
    outlier_threshold: float = 1.5           # k for IQR, std for zscore
    outlier_action: str = "clip"             # clip|flag

    # Step 3 — Correlation (on imputed+cleaned data, stats are now reliable)
    correlation_threshold: Optional[float] = 0.95   # None = disabled

    # Step 4 — Categorical encoding (before scaling)
    categorical_encoder: str = "onehot"     # onehot|ordinal|target|binary

    # Step 5 — Distribution transform (NEW: normalize skewness before scaling)
    distribution_transform: str = "none"    # none|yeo-johnson|box-cox|log1p

    # Step 6 — Scaling
    scaler: str = "standard"                # none|standard|minmax|robust

    # Step 7 — Missing indicator flags (NEW: binary flags for originally-missing cols)
    missing_indicator: bool = False         # True = add __missing_X__ columns

    # Step 8 — Feature selection
    feature_selection_method: str = "none"  # none|variance_threshold|mutual_info|rfe
    feature_selection_k: float = 0.75       # fraction or int of features to keep

    # Step 9 — Imbalance (classification only, train only)
    imbalance_method: str = "none"          # none|smote|borderline_smote|adasyn|class_weight

    @classmethod
    def from_genes(cls, genes: Dict[str, Any]) -> "PreprocessingConfig":
        """Build from a chromosome gene dict (keys may be a superset)."""
        fields = set(cls.__dataclass_fields__.keys())  # type: ignore
        filtered = {k: v for k, v in genes.items() if k in fields}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PreprocessingPipeline:
    """
    End-to-end preprocessing pipeline with scientifically correct step ordering.

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

        # Instantiate steps in execution order
        self._numeric_imputer = NumericImputer(strategy=config.numeric_imputer)
        self._outlier_handler = OutlierHandler(
            method=config.outlier_method,
            threshold=config.outlier_threshold,
            action=config.outlier_action,
        )
        self._correlation_filter = CorrelationFilter(threshold=config.correlation_threshold)
        self._categorical_encoder = CategoricalEncoder(strategy=config.categorical_encoder)
        self._distribution_transform = DistributionTransform(method=config.distribution_transform)
        self._scaler = Scaler(method=config.scaler)
        self._missing_indicator = MissingIndicator(enabled=config.missing_indicator)
        self._feature_selector = FeatureSelector(
            method=config.feature_selection_method,
            keep_k=config.feature_selection_k,
            problem_type_str=problem_type.value,
            random_seed=random_seed,
        )
        self._imbalance_handler = ImbalanceHandler(
            method=config.imbalance_method,
            random_seed=random_seed,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform_train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit all steps on training data and return (X_transformed, y_transformed).
        ImbalanceHandler resampling is applied here — never on val/test.
        """
        log.info("PreprocessingPipeline: fitting on %d samples × %d features", *X_train.shape)
        X = X_train.copy()
        y = y_train.copy()

        # Fit missing indicator on raw data BEFORE imputation (captures true missingness)
        # We store the raw missing mask to apply as binary columns later
        self._missing_indicator.fit(X)
        X_raw_missing_mask = X.isnull()  # save mask before imputation fills NaNs

        # Step 1: Numeric imputation — FIRST (outlier detection needs clean values)
        X = self._numeric_imputer.fit_transform(X, y)

        # Step 2: Outlier handling — on clean numeric data, before scaling
        X = self._outlier_handler.fit_transform(X, y)

        # Step 3: Correlation filter — after imputation+outlier treatment, stats are reliable
        X = self._correlation_filter.fit_transform(X, y)

        # Step 4: Categorical encoding — before scaling (encoding uses string values)
        needs_y = (self.config.categorical_encoder == "target")
        X = self._categorical_encoder.fit_transform(X, y if needs_y else None)

        # Step 5: Distribution transform — normalize skewness before scaling
        X = self._distribution_transform.fit_transform(X, y)

        # Step 6: Scaling — after all columns are numeric
        X = self._scaler.fit_transform(X, y)

        # Step 7: Missing indicator — add binary flags using PRE-IMPUTATION mask
        X = self._missing_indicator.transform_with_mask(X, X_raw_missing_mask)

        # Step 8: Feature selection — on fully preprocessed data
        self._feature_selector.fit(X, y)
        X = self._feature_selector.transform(X)

        # Step 9: Imbalance handling — LAST, train only
        if self.problem_type != ProblemType.REGRESSION:
            X, y = self._imbalance_handler.fit_resample(X, y)
        else:
            self._imbalance_handler.fit(X, y)

        self._is_fitted = True
        log.info(
            "PreprocessingPipeline: output %d samples × %d features",
            X.shape[0], X.shape[1],
        )
        return X, y

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted transformations to val/test data.
        NO fitting occurs here — zero leakage guaranteed.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit_transform_train() before transform().")
        X = X.copy()
        X_raw_missing_mask = X.isnull()   # capture before imputation
        X = self._numeric_imputer.transform(X)
        X = self._outlier_handler.transform(X)
        X = self._correlation_filter.transform(X)
        X = self._categorical_encoder.transform(X)
        X = self._distribution_transform.transform(X)
        X = self._scaler.transform(X)
        X = self._missing_indicator.transform_with_mask(X, X_raw_missing_mask)
        X = self._feature_selector.transform(X)
        # ImbalanceHandler is NOT applied to val/test
        return X

    def sample_weights(self, y: pd.Series):
        """Return per-sample weights if imbalance_method='class_weight', else None."""
        return self._imbalance_handler.sample_weights(y)

    # ------------------------------------------------------------------
    # Introspection helpers (used by HTML reporter)
    # ------------------------------------------------------------------

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
