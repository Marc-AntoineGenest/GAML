"""Shared pytest fixtures."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes


@pytest.fixture
def clf_df():
    """Small classification dataset with synthetic noise."""
    data = load_breast_cancer(as_frame=True)
    df = data.frame.rename(columns={"target": "label"})
    rng = np.random.default_rng(42)
    # Add 10% missing values to first 3 columns
    for col in df.columns[:3]:
        mask = rng.random(len(df)) < 0.10
        df.loc[mask, col] = np.nan
    # Add a skewed feature and a categorical
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
