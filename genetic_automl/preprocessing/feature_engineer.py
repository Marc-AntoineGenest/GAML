"""
FeatureEngineer
---------------
Generates new features from existing numeric columns before scaling.

Placed at step 1.5 in the pipeline — after imputation and outlier handling
(so inputs are clean), but before scaling (so raw numeric values are available
for ratio and log transforms).

Why this matters
----------------
GBMs already capture non-linear relationships, but explicit feature engineering
still helps by:
  - Reducing the depth of trees needed to capture a relationship (faster, less
    overfit).
  - Enabling features that GBMs can't easily build, like exact ratios and
    cross-column products.
  - Letting the GA discover *which* engineered features are worth keeping —
    the FeatureSelector downstream removes the ones that don't help.

Strategies (all controlled by gene values)
-------------------------------------------
none         No-op. Default for small/fast runs.

poly2        Degree-2 polynomial features for the top-N most-variant numeric
             columns (capped by max_interaction_features to avoid combinatorial
             explosion). Generates col_i^2 and col_i * col_j terms.
             N is controlled by the interaction_features gene.

ratio        Pairwise ratios (col_i / (col_j + eps)) for the top-N column
             pairs with highest absolute correlation to each other — the pairs
             most likely to cancel or amplify. Prevents division by zero via
             a small epsilon offset.

log1p        log(x + 1) for numeric columns that are right-skewed
             (skewness > skew_threshold). Produces a copy of the column
             rather than replacing it, so the original is still available.

all          Apply poly2 + ratio + log1p in sequence.

Design decisions
----------------
- All operations use only the columns available at fit time. New columns are
  named deterministically (e.g. `feat_poly_a_b`, `feat_ratio_a_b`,
  `feat_log1p_a`) so transform() can reconstruct the exact same schema.
- The step is safe to skip: setting strategy='none' returns X unchanged.
- Works correctly with downstream FeatureSelector — engineered columns that
  don't improve the model's CV score get dropped automatically.
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)

_EPS = 1e-6          # denominator offset for ratio features
_SKEW_THRESHOLD = 1.0  # |skewness| above which log1p is applied


class FeatureEngineer:
    """
    Generates engineered features from numeric columns.

    Parameters
    ----------
    strategy : str
        'none' | 'poly2' | 'ratio' | 'log1p' | 'all'
    max_interaction_features : int
        Maximum number of source columns used to build poly2 or ratio pairs.
        Caps the quadratic blow-up: k columns → k*(k+1)/2 new features.
        Defaults to 8 → at most 36 new poly features.
    """

    def __init__(
        self,
        strategy: str = "none",
        max_interaction_features: int = 8,
    ) -> None:
        self.strategy = strategy
        self.max_interaction_features = max_interaction_features

        # Set at fit time
        self._poly_cols: list[str] = []            # source cols for poly2
        self._ratio_pairs: list[tuple[str, str]] = []  # (num_col, denom_col)
        self._log1p_cols: list[str] = []           # cols with high skewness
        self._is_fitted = False


    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> FeatureEngineer:
        if self.strategy == "none":
            self._is_fitted = True
            return self

        num_cols = X.select_dtypes(include="number").columns.tolist()

        if self.strategy in ("poly2", "all"):
            self._poly_cols = self._select_top_variant(X, num_cols)

        if self.strategy in ("ratio", "all"):
            self._ratio_pairs = self._select_ratio_pairs(X, num_cols)

        if self.strategy in ("log1p", "all"):
            self._log1p_cols = self._select_skewed(X, num_cols)

        log.info(
            "FeatureEngineer(strategy=%s) fit | poly_cols=%d | ratio_pairs=%d | log1p_cols=%d",
            self.strategy,
            len(self._poly_cols),
            len(self._ratio_pairs),
            len(self._log1p_cols),
        )
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")
        if self.strategy == "none":
            return X

        X = X.copy()

        if self.strategy in ("poly2", "all"):
            X = self._apply_poly2(X)

        if self.strategy in ("ratio", "all"):
            X = self._apply_ratios(X)

        if self.strategy in ("log1p", "all"):
            X = self._apply_log1p(X)

        log.debug(
            "FeatureEngineer transform | input_cols=%d | output_cols=%d",
            len(X.columns) - self._n_new_cols,
            len(X.columns),
        )
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


    def _apply_poly2(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self._poly_cols if c in X.columns]
        new: dict = {}
        for i, c1 in enumerate(cols):
            # Squared term
            new[f"feat_poly_{c1}_sq"] = X[c1].values ** 2
            # Cross terms
            for c2 in cols[i + 1:]:
                new[f"feat_poly_{c1}_{c2}"] = X[c1].values * X[c2].values
        if new:
            X = pd.concat([X, pd.DataFrame(new, index=X.index)], axis=1)
        return X

    def _apply_ratios(self, X: pd.DataFrame) -> pd.DataFrame:
        new: dict = {}
        for num, denom in self._ratio_pairs:
            if num in X.columns and denom in X.columns:
                new[f"feat_ratio_{num}_{denom}"] = (
                    X[num].values / (X[denom].values + _EPS)
                )
        if new:
            X = pd.concat([X, pd.DataFrame(new, index=X.index)], axis=1)
        return X

    def _apply_log1p(self, X: pd.DataFrame) -> pd.DataFrame:
        new: dict = {}
        for col in self._log1p_cols:
            if col in X.columns:
                # Shift negative values so log1p is always valid
                vals = X[col].values.astype(float)
                min_val = vals.min()
                shifted = vals - min(min_val, 0)
                new[f"feat_log1p_{col}"] = np.log1p(shifted)
        if new:
            X = pd.concat([X, pd.DataFrame(new, index=X.index)], axis=1)
        return X


    def _select_top_variant(self, X: pd.DataFrame, num_cols: list[str]) -> list[str]:
        """Select top-N numeric columns by variance (most information-dense)."""
        if not num_cols:
            return []
        variances = X[num_cols].var()
        top = variances.nlargest(self.max_interaction_features).index.tolist()
        return top

    def _select_ratio_pairs(
        self, X: pd.DataFrame, num_cols: list[str]
    ) -> list[tuple[str, str]]:
        """
        Select pairs (a, b) where a and b are highly correlated — the pairs
        where a ratio is most likely to reveal a meaningful relationship.
        Cap total pairs at max_interaction_features to avoid explosion.
        """
        if len(num_cols) < 2:
            return []
        top_cols = self._select_top_variant(X, num_cols)
        pairs = []
        seen: set = set()
        try:
            corr = X[top_cols].corr().abs()
        except Exception:
            return []

        # Sort all pairs by absolute correlation, highest first
        for i, c1 in enumerate(top_cols):
            for c2 in top_cols[i + 1:]:
                key = (c1, c2)
                if key not in seen:
                    seen.add(key)
                    val = corr.at[c1, c2] if (c1 in corr.index and c2 in corr.columns) else 0.0
                    pairs.append((val, c1, c2))

        pairs.sort(reverse=True)
        return [(c1, c2) for _, c1, c2 in pairs[: self.max_interaction_features]]

    def _select_skewed(self, X: pd.DataFrame, num_cols: list[str]) -> list[str]:
        """Select columns whose skewness exceeds the threshold."""
        skewed = []
        for col in num_cols:
            try:
                sk = float(X[col].skew())
                if abs(sk) > _SKEW_THRESHOLD:
                    skewed.append(col)
            except Exception:
                continue
        return skewed

    @property
    def _n_new_cols(self) -> int:
        """Count of columns added by this step (approximate, for logging)."""
        n_poly = len(self._poly_cols)
        poly_count = n_poly + (n_poly * (n_poly - 1)) // 2
        return poly_count + len(self._ratio_pairs) + len(self._log1p_cols)
