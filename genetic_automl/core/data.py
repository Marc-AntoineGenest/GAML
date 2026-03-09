"""
DataManager — 3-way train/val/test split with zero test set contamination.

Split architecture
------------------
Full data
├── Test set (test_size %)     — LOCKED. Touched ONLY for final evaluation.
│                                Never enters the GA loop. Never seen by any
│                                preprocessing fit or model during evolution.
└── Dev set (1 - test_size %)
    ├── Train set (1 - val_size % of dev)   — chromosome preprocessing fit + model fit
    └── Val set   (val_size % of dev)       — k-fold CV splits (if using CV fitness)
                                             OR single val split (legacy mode)

The key invariant:
  GA fitness is measured on val (via CV on train+val).
  Final score is measured on test — completely independent.

Designed to be swappable: a future Polars or Spark backend
just needs to implement the same public interface.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from genetic_automl.core.problem import ProblemType
from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)


class DataManager:
    """
    Loads, validates, and performs 3-way stratified split.

    Parameters
    ----------
    target_column : str
    problem_type : ProblemType
    test_size : float
        Fraction of total data locked for final test evaluation (default 0.15).
    val_size : float
        Fraction of remaining dev data used as val during GA evolution (default 0.2).
        Only relevant when NOT using k-fold CV fitness (legacy mode).
    stratify : bool
        Stratify splits on the label column (classification only).
    random_seed : int
    """

    def __init__(
        self,
        target_column: str,
        problem_type: ProblemType,
        test_size: float = 0.15,
        val_size: float = 0.2,
        stratify: bool = True,
        random_seed: int = 42,
    ) -> None:
        self.target_column = target_column
        self.problem_type = problem_type
        self.test_size = test_size
        self.val_size = val_size
        self.stratify = stratify
        self.random_seed = random_seed

        self._train: Optional[pd.DataFrame] = None
        self._val: Optional[pd.DataFrame] = None
        self._test: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self, path: str) -> pd.DataFrame:
        """Load CSV / Parquet / Excel from *path* into a DataFrame."""
        path_lower = path.lower()
        if path_lower.endswith(".csv"):
            df = pd.read_csv(path)
        elif path_lower.endswith((".parquet", ".pq")):
            df = pd.read_parquet(path)
        elif path_lower.endswith((".xlsx", ".xls")):
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")
        log.info("Loaded %d rows × %d cols from '%s'", len(df), df.shape[1], path)
        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic sanity checks — returns the (possibly coerced) DataFrame."""
        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found. "
                f"Available: {list(df.columns)}"
            )
        missing_pct = df.isnull().mean().max() * 100
        if missing_pct > 80:
            log.warning("Some columns have >80%% missing values.")
        log.info(
            "Shape: %s | Target: '%s' | Missing (max col): %.1f%%",
            df.shape, self.target_column, missing_pct,
        )
        return df

    def three_way_split(
        self,
        df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split into (train, val, test).

        - If *test_df* is provided: df → train+val, test_df → test.
        - Otherwise: df → test first, then remainder → train+val.

        Returns
        -------
        (train_df, val_df, test_df)
            train_df + val_df = dev set for GA evolution
            test_df           = locked hold-out for final evaluation only
        """
        strat = self._stratify_col

        if test_df is not None:
            # External test set provided — split df into train/val
            train_df, val_df = self._split_two(df, self.val_size, strat(df))
            self._train, self._val, self._test = train_df, val_df, test_df
        else:
            # Step 1: carve out test
            dev_df, test_df_split = self._split_two(df, self.test_size, strat(df))
            # Step 2: split remaining dev into train+val
            train_df, val_df = self._split_two(dev_df, self.val_size, strat(dev_df))
            self._train, self._val, self._test = train_df, val_df, test_df_split

        n_total = len(df) + (len(test_df) if test_df is not None else 0)
        log.info(
            "3-way split | train=%d (%.0f%%) | val=%d (%.0f%%) | test=%d (%.0f%%) | total=%d",
            len(self._train), 100 * len(self._train) / n_total,
            len(self._val),   100 * len(self._val)   / n_total,
            len(self._test),  100 * len(self._test)  / n_total,
            n_total,
        )
        return self._train, self._val, self._test

    # Backward-compatible alias
    def split(
        self,
        df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Legacy 2-way split. Returns (train, test). Val is carved from train."""
        train, _val, test = self.three_way_split(df, test_df)
        return train, test

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def train(self) -> pd.DataFrame:
        if self._train is None:
            raise RuntimeError("Call three_way_split() first.")
        return self._train

    @property
    def val(self) -> pd.DataFrame:
        if self._val is None:
            raise RuntimeError("Call three_way_split() first.")
        return self._val

    @property
    def test(self) -> pd.DataFrame:
        if self._test is None:
            raise RuntimeError("Call three_way_split() first.")
        return self._test

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=[self.target_column])

    def labels(self, df: pd.DataFrame) -> pd.Series:
        return df[self.target_column]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_two(
        self,
        df: pd.DataFrame,
        split_size: float,
        stratify_col,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return train_test_split(
            df,
            test_size=split_size,
            random_state=self.random_seed,
            stratify=stratify_col,
        )

    @property
    def _stratify_col(self):
        """Returns a function: df → stratify array or None."""
        def _get(df):
            if self.stratify and self.problem_type == ProblemType.CLASSIFICATION:
                return df[self.target_column]
            return None
        return _get
