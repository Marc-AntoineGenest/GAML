"""
CategoricalEncoder
------------------
Encodes categorical (object / category dtype) columns.

Strategies
----------
onehot      : OneHotEncoder — creates binary dummy columns
ordinal     : OrdinalEncoder — integer codes (preserves memory, good for trees)
target      : Target encoding with cross-val (no leakage) — requires y on fit
binary      : Binary encoding (category_encoders style, implemented here)

Categorical columns with missing values are filled with '__MISSING__'
before encoding.

Unknown categories at transform time are handled gracefully:
  - onehot : handle_unknown='ignore' → all zeros
  - ordinal : unknown → -1
  - target  : unknown → global mean
  - binary  : unknown → all zeros
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder

from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)


class CategoricalEncoder:
    """
    Parameters
    ----------
    strategy : str
        'onehot' | 'ordinal' | 'target' | 'binary'
    handle_missing : str
        'fill'  — replace NaN with '__MISSING__' (default)
        'drop'  — drop rows with NaN categorical (not recommended)
    """

    def __init__(
        self,
        strategy: str = "onehot",
        handle_missing: str = "fill",
        n_folds: int = 5,
    ) -> None:
        self.strategy = strategy
        self.handle_missing = handle_missing
        self.n_folds = n_folds

        self._cat_cols: List[str] = []
        self._encoder = None
        # target encoding: col → (category → mean target)
        self._target_map: Dict[str, Dict] = {}
        self._global_mean: float = 0.0
        # binary: col → sorted categories list
        self._binary_cats: Dict[str, List] = {}
        # onehot output column names (for transform alignment)
        self._ohe_cols: List[str] = []

    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CategoricalEncoder":
        self._cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        if not self._cat_cols:
            log.info("CategoricalEncoder: no categorical columns found")
            return self

        log.info(
            "CategoricalEncoder(strategy=%s): %d categorical columns",
            self.strategy,
            len(self._cat_cols),
        )

        X_cat = self._fill_missing(X[self._cat_cols])

        if self.strategy == "onehot":
            from sklearn.preprocessing import OneHotEncoder
            self._encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.float32,
            )
            self._encoder.fit(X_cat)
            # Pre-compute output column names
            self._ohe_cols = list(self._encoder.get_feature_names_out(self._cat_cols))

        elif self.strategy == "ordinal":
            # B8 fix: unknown_value=-1 mapped unseen categories to a value below index 0,
            # which distance-based models (KNN, SVM, linear) interpret as a phantom
            # category "less than all known ones". Use NaN as the sentinel and then fill
            # with the mean ordinal index (mid-range), which is a neutral, non-misleading
            # imputation for an unknown category.
            self._encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,   # B8: NaN sentinel — filled post-transform
                dtype=np.float32,
            )
            self._encoder.fit(X_cat)
            # Store per-column mean ordinal index for neutral fill
            self._ordinal_fill: Dict[str, float] = {}
            for col, cats in zip(self._cat_cols, self._encoder.categories_):
                self._ordinal_fill[col] = float(len(cats) - 1) / 2.0

        elif self.strategy == "target":
            if y is None:
                raise ValueError("strategy='target' requires y to be passed to fit().")
            self._global_mean = float(y.mean())
            self._fit_target_encoding(X_cat, y)

        elif self.strategy == "binary":
            for col in self._cat_cols:
                cats = sorted(X_cat[col].unique().tolist())
                self._binary_cats[col] = cats

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._cat_cols:
            return X

        X = X.copy()
        cat_cols_present = [c for c in self._cat_cols if c in X.columns]
        X_cat = self._fill_missing(X[cat_cols_present])
        non_cat = X.drop(columns=cat_cols_present)

        if self.strategy == "onehot":
            encoded = self._encoder.transform(X_cat)
            enc_df = pd.DataFrame(encoded, columns=self._ohe_cols, index=X.index)
            return pd.concat([non_cat, enc_df], axis=1)

        elif self.strategy == "ordinal":
            encoded = self._encoder.transform(X_cat)
            enc_df = pd.DataFrame(encoded, columns=cat_cols_present, index=X.index)
            # B8 fix: fill NaN sentinels (unseen categories) with the neutral
            # per-column mid-range ordinal value stored at fit time.
            for col in cat_cols_present:
                fill_val = getattr(self, "_ordinal_fill", {}).get(col, 0.0)
                enc_df[col] = enc_df[col].fillna(fill_val)
            return pd.concat([non_cat, enc_df], axis=1)

        elif self.strategy == "target":
            enc_df = X_cat.copy()
            for col in cat_cols_present:
                mapping = self._target_map.get(col, {})
                enc_df[col] = X_cat[col].map(mapping).fillna(self._global_mean)
            return pd.concat([non_cat, enc_df], axis=1)

        elif self.strategy == "binary":
            parts = [non_cat]
            for col in cat_cols_present:
                cats = self._binary_cats[col]
                n_bits = max(1, int(np.ceil(np.log2(len(cats) + 1))))
                cat_to_int = {c: i for i, c in enumerate(cats)}
                int_codes = X_cat[col].map(cat_to_int).fillna(0).astype(int)
                int_codes_arr = int_codes.to_numpy()
                for bit in range(n_bits):
                    parts.append(
                        pd.Series(
                            ((int_codes_arr >> bit) & 1),
                            index=X.index,
                            name=f"{col}_bin{bit}",
                        )
                    )
            return pd.concat(parts, axis=1)

        return X

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------

    def _fill_missing(self, X_cat: pd.DataFrame) -> pd.DataFrame:
        return X_cat.fillna("__MISSING__").astype(str)

    def _fit_target_encoding(self, X_cat: pd.DataFrame, y: pd.Series) -> None:
        """
        Cross-validated target encoding to avoid leakage.
        For each fold: fit mean on out-of-fold samples, apply to in-fold.
        Final mapping = mean over all folds (global smoothed estimate).
        """
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        for col in self._cat_cols:
            # Collect per-fold category means
            fold_means: Dict = {}
            for train_idx, val_idx in kf.split(X_cat):
                train_cats = X_cat[col].iloc[train_idx]
                train_y = y.iloc[train_idx]
                col_mean = train_y.groupby(train_cats.values).mean().to_dict()
                for cat, mean in col_mean.items():
                    if cat not in fold_means:
                        fold_means[cat] = []
                    fold_means[cat].append(mean)
            # Average across folds
            self._target_map[col] = {
                cat: np.mean(means) for cat, means in fold_means.items()
            }
