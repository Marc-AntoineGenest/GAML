"""
CategoricalEncoder
------------------
Encodes categorical (object / category dtype) columns.

Strategies
----------
onehot  : OneHotEncoder — binary dummy columns; unknown → all-zeros
ordinal : OrdinalEncoder — integer codes; unknown → per-column mid-range index
target  : Cross-validated target encoding (requires y at fit time)
binary  : Bit-decomposition encoding

Missing values are filled with '__MISSING__' before encoding.
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
        'fill' — replace NaN with '__MISSING__' (default)
    n_folds : int
        Folds used for cross-validated target encoding.
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
        self._target_map: Dict[str, Dict] = {}
        self._global_mean: float = 0.0
        self._binary_cats: Dict[str, List] = {}
        self._ohe_cols: List[str] = []
        self._ordinal_fill: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CategoricalEncoder":
        self._cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if not self._cat_cols:
            log.info("CategoricalEncoder: no categorical columns found")
            return self

        log.info("CategoricalEncoder(strategy=%s): %d columns", self.strategy, len(self._cat_cols))
        X_cat = self._fill_missing(X[self._cat_cols])

        if self.strategy == "onehot":
            from sklearn.preprocessing import OneHotEncoder
            self._encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
            self._encoder.fit(X_cat)
            self._ohe_cols = list(self._encoder.get_feature_names_out(self._cat_cols))

        elif self.strategy == "ordinal":
            # Unknown categories use NaN sentinel, filled post-transform with the
            # per-column mid-range index (neutral — avoids implying a rank order).
            self._encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=np.nan, dtype=np.float32,
            )
            self._encoder.fit(X_cat)
            for col, cats in zip(self._cat_cols, self._encoder.categories_):
                self._ordinal_fill[col] = float(len(cats) - 1) / 2.0

        elif self.strategy == "target":
            if y is None:
                raise ValueError("strategy='target' requires y to be passed to fit().")
            self._global_mean = float(y.mean())
            self._fit_target_encoding(X_cat, y)

        elif self.strategy == "binary":
            for col in self._cat_cols:
                self._binary_cats[col] = sorted(X_cat[col].unique().tolist())

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._cat_cols:
            return X
        X = X.copy()

        # Columns that were seen at fit time but are absent from X at transform time.
        # We fill them with the missing sentinel so downstream encoders never see a
        # column-count mismatch (e.g. OHE raises "X has 0 features but expected N").
        absent_cols = [c for c in self._cat_cols if c not in X.columns]
        if absent_cols:
            log.warning(
                "CategoricalEncoder.transform(): %d column(s) absent from input that "
                "were present at fit time — filling with '__MISSING__': %s",
                len(absent_cols), absent_cols,
            )
            for col in absent_cols:
                X[col] = "__MISSING__"

        cols = [c for c in self._cat_cols if c in X.columns]
        X_cat = self._fill_missing(X[cols])
        non_cat = X.drop(columns=cols)

        if self.strategy == "onehot":
            enc_df = pd.DataFrame(
                self._encoder.transform(X_cat), columns=self._ohe_cols, index=X.index
            )
            return pd.concat([non_cat, enc_df], axis=1)

        elif self.strategy == "ordinal":
            enc_df = pd.DataFrame(
                self._encoder.transform(X_cat), columns=cols, index=X.index
            )
            for col in cols:
                enc_df[col] = enc_df[col].fillna(self._ordinal_fill.get(col, 0.0))
            return pd.concat([non_cat, enc_df], axis=1)

        elif self.strategy == "target":
            enc_df = X_cat.copy()
            for col in cols:
                enc_df[col] = X_cat[col].map(self._target_map.get(col, {})).fillna(self._global_mean)
            return pd.concat([non_cat, enc_df], axis=1)

        elif self.strategy == "binary":
            parts = [non_cat]
            for col in cols:
                cats = self._binary_cats[col]
                n_bits = max(1, int(np.ceil(np.log2(len(cats) + 1))))
                cat_to_int = {c: i for i, c in enumerate(cats)}
                codes = X_cat[col].map(cat_to_int).fillna(0).astype(int).to_numpy()
                for bit in range(n_bits):
                    parts.append(pd.Series((codes >> bit) & 1, index=X.index, name=f"{col}_bin{bit}"))
            return pd.concat(parts, axis=1)

        return X

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def _fill_missing(self, X_cat: pd.DataFrame) -> pd.DataFrame:
        return X_cat.fillna("__MISSING__").astype(str)

    def _fit_target_encoding(self, X_cat: pd.DataFrame, y: pd.Series) -> None:
        """Cross-validated target encoding — no leakage."""
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        for col in self._cat_cols:
            fold_means: Dict = {}
            for train_idx, _ in kf.split(X_cat):
                for cat, mean in y.iloc[train_idx].groupby(X_cat[col].iloc[train_idx].values).mean().items():
                    fold_means.setdefault(cat, []).append(mean)
            self._target_map[col] = {cat: np.mean(v) for cat, v in fold_means.items()}
