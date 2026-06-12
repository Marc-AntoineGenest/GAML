"""Shared pytest fixtures."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes

from genetic_automl.config import (
    AutoMLConfig,
    DataConfig,
    GeneticConfig,
    PipelineConfig,
    ReportConfig,
)
from genetic_automl.core.problem import ProblemType


@pytest.fixture
def clf_df():
    """Small classification dataset with synthetic noise."""
    data = load_breast_cancer(as_frame=True)
    df = data.frame.rename(columns={"target": "label"})
    rng = np.random.default_rng(42)
    for col in df.columns[:3]:
        mask = rng.random(len(df)) < 0.10
        df.loc[mask, col] = np.nan
    df["skewed"] = rng.lognormal(0, 1.5, len(df))
    df["cat"] = rng.choice(["A", "B", "C"], len(df))
    return df


@pytest.fixture
def reg_df():
    """Small regression dataset with synthetic missing values."""
    data = load_diabetes(as_frame=True)
    df = data.frame
    rng = np.random.default_rng(7)
    mask = rng.random(len(df)) < 0.08
    df.loc[mask, "bmi"] = np.nan
    return df


@pytest.fixture
def small_X_y():
    """Tiny X/y for fast unit tests (100 samples, 5 features)."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((100, 5)), columns=list("abcde"))
    y = pd.Series(rng.integers(0, 2, 100))
    return X, y


@pytest.fixture
def tiny_clf_df():
    """Minimal classification DataFrame for smoke tests (120 rows, 4 features)."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(120, 4), columns=list("abcd"))
    y = pd.Series((X["a"] + rng.randn(120) * 0.5 > 0).astype(int), name="y")
    return pd.concat([X, y], axis=1)


def make_fast_config(
    problem_type: ProblemType = ProblemType.CLASSIFICATION,
    target_column: str = "label",
    output_dir: str = "/tmp/gaml_test_reports",
    **genetic_overrides,
) -> PipelineConfig:
    """Return a minimal PipelineConfig suitable for fast unit tests."""
    genetic_kwargs = dict(
        population_size=4,
        generations=2,
        early_stopping_rounds=2,
        n_cv_folds=2,
        warm_start=True,
        warm_start_n_seeds=2,
        warm_start_halving_pool_ratio=0,
        adaptive_mutation=False,
        random_seed=42,
    )
    genetic_kwargs.update(genetic_overrides)
    return PipelineConfig(
        problem_type=problem_type,
        target_column=target_column,
        genetic=GeneticConfig(**genetic_kwargs),
        automl=AutoMLConfig(backend="sklearn"),
        data=DataConfig(test_size=0.15),
        report=ReportConfig(output_dir=output_dir),
    )
