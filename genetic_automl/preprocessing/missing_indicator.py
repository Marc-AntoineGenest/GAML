"""
MissingIndicator
----------------
Creates binary flag columns for each feature that had missing values in training.

Why this matters:
  - Missingness is often NOT random — the absence of a value can be highly
    predictive (e.g., a patient not having a lab test result vs. having it)
  - After imputation, the information "this was missing" is lost
  - Adding a binary __missing_{col}__ column preserves that signal

Applied BEFORE imputation conceptually (fit on raw data), but the transform
is applied AFTER imputation in the pipeline order. This way:
  - The indicator captures original missingness
  - The numeric column itself is filled cleanly by the imputer
  - No NaN values are introduced by this step

Only adds indicator columns for features where at least *min_missing_frac*
of training samples were missing (avoids noise from near-zero missingness).
"""

from __future__ import annotations

from typing import List

import pandas as pd

from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)


class MissingIndicator:
    """
    Parameters
    ----------
    enabled : bool
        Whether to add indicator columns. When False, this is a no-op.
    min_missing_frac : float
        Minimum fraction of training samples that must be missing for a
        column to get an indicator. Default 0.01 (at least 1% missing).
    """

    def __init__(
        self,
        enabled: bool = False,
        min_missing_frac: float = 0.01,
    ) -> None:
        self.enabled = enabled
        self.min_missing_frac = min_missing_frac
        self.indicator_columns: List[str] = []   # original cols that get indicators

    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame) -> "MissingIndicator":
        """Fit on raw (pre-imputation) data to detect originally-missing columns."""
        self.indicator_columns = []
        if not self.enabled:
            return self

        missing_frac = X.isnull().mean()
        self.indicator_columns = [
            col for col, frac in missing_frac.items()
            if frac >= self.min_missing_frac
        ]

        log.info(
            "MissingIndicator: %d columns will get binary indicator flags: %s",
            len(self.indicator_columns),
            self.indicator_columns,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add __missing_{col}__ binary columns for each tracked column.
        Called AFTER imputation — uses the original pre-imputation state
        if passed as-is, but since imputation already happened we use
        the indicator built from the fitted missing pattern.

        Note: At transform time on val/test, we add indicator columns based
        on the FITTED column list (from training). A val/test row missing a
        column that wasn't missing in training gets 0 (correct — we never saw
        that pattern in training). A column that was always missing in training
        but present in test gets 0 (also correct).
        """
        if not self.enabled or not self.indicator_columns:
            return X

        X = X.copy()
        for col in self.indicator_columns:
            indicator_name = f"__missing_{col}__"
            if col in X.columns:
                X[indicator_name] = X[col].isnull().astype(int)
            else:
                # Column was dropped by correlation filter or similar — no indicator
                X[indicator_name] = 0

        return X

    def transform_with_mask(self, X: pd.DataFrame, raw_mask: pd.DataFrame) -> pd.DataFrame:
        """
        Add indicator columns using a pre-captured raw missing mask.
        This is the correct call path: mask is captured before imputation,
        applied after — so indicators reflect true original missingness.
        """
        if not self.enabled or not self.indicator_columns:
            return X
        X = X.copy()
        for col in self.indicator_columns:
            indicator_name = f"__missing_{col}__"
            if col in raw_mask.columns:
                X[indicator_name] = raw_mask[col].astype(int).values
            else:
                X[indicator_name] = 0
        return X

    def fit_transform_raw(self, X_raw: pd.DataFrame) -> "MissingIndicator":
        """Convenience: fit on raw data and return self."""
        return self.fit(X_raw)
