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
                                             or single val split

The key invariant:
  GA fitness is measured on val (via CV on train+val).
  Final score is measured on test — completely independent.

Designed to be swappable: pass backend="polars" to load() for 2-10x faster
file loading on large datasets. The result is always a pandas DataFrame so
the rest of the GAML stack is unaffected.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False

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
        Only relevant when not using k-fold CV fitness.
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

        self._train: pd.DataFrame | None = None
        self._val: pd.DataFrame | None = None
        self._test: pd.DataFrame | None = None


    def load(
        self, path: str, backend: str = "pandas"
    ) -> pd.DataFrame:
        """
        Load CSV / Parquet / Excel from *path* into a pandas DataFrame.

        Parameters
        ----------
        path : str
        backend : str
            "pandas" (default) or "polars".
            Polars is 2-10x faster for large CSV/Parquet files.
            Regardless of backend, the result is always a pandas DataFrame
            so the rest of the GAML stack is unaffected.
        """
        if backend == "polars":
            return self._load_polars(path)
        return self._load_pandas(path)

    def _load_pandas(self, path: str) -> pd.DataFrame:
        """Load with pandas (default)."""
        path_lower = path.lower()
        if path_lower.endswith(".csv"):
            df = pd.read_csv(path)
        elif path_lower.endswith((".parquet", ".pq")):
            df = pd.read_parquet(path)
        elif path_lower.endswith((".xlsx", ".xls")):
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")
        log.info("Loaded %d rows x %d cols from '%s' (pandas)", len(df), df.shape[1], path)
        return df

    def _load_polars(self, path: str) -> pd.DataFrame:
        """
        Load with Polars, then convert to pandas.

        Polars is 2-10x faster than pandas for CSV/Parquet loading because
        it uses a multi-threaded Rust parser. The conversion to pandas adds
        ~10-20% overhead but is negligible compared to the loading speedup.
        Requires: pip install polars pyarrow
        """
        if not _POLARS_AVAILABLE:
            log.warning(
                "Polars is not installed — falling back to pandas. "
                "Install with: pip install polars pyarrow"
            )
            return self._load_pandas(path)

        path_lower = path.lower()
        try:
            if path_lower.endswith(".csv"):
                lf = pl.read_csv(path)
            elif path_lower.endswith((".parquet", ".pq")):
                lf = pl.read_parquet(path)
            else:
                log.warning(
                    "Polars backend does not support '%s' — falling back to pandas.", path
                )
                return self._load_pandas(path)

            df = lf.to_pandas()
            log.info(
                "Loaded %d rows x %d cols from '%s' (polars)",
                len(df), df.shape[1], path,
            )
            return df
        except Exception as exc:
            log.warning(
                "Polars load failed (%s) — falling back to pandas.", exc
            )
            return self._load_pandas(path)

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
        test_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        test_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """2-way split alias. Returns (train, test). Val is carved from train."""
        train, _val, test = self.three_way_split(df, test_df)
        return train, test


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


    def _split_two(
        self,
        df: pd.DataFrame,
        split_size: float,
        stratify_col,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
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
