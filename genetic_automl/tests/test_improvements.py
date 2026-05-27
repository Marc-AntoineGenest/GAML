"""
Tests for improvements introduced in this PR:

  1.  __version__ is exported from the top-level package.
  2.  AutoMLPipeline.summary() returns a populated dict after fit().
  3.  AutoMLPipeline.feature_importances_ returns a named pandas Series.
  4.  ReportConfig / load_config handles mlflow_tracking_uri: null correctly.
  5.  CategoricalEncoder.fit() no longer emits a Pandas4Warning about
      implicit string-dtype inclusion.
"""
from __future__ import annotations

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest

from genetic_automl import AutoMLPipeline, __version__
from genetic_automl.config import (
    AutoMLConfig,
    DataConfig,
    GeneticConfig,
    PipelineConfig,
    ReportConfig,
)
from genetic_automl.core.problem import ProblemType
from genetic_automl.preprocessing.categorical_encoder import CategoricalEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fast_config(problem_type=ProblemType.CLASSIFICATION, target="label"):
    return PipelineConfig(
        problem_type=problem_type,
        target_column=target,
        genetic=GeneticConfig(
            population_size=4,
            generations=2,
            early_stopping_rounds=2,
            n_cv_folds=2,
            warm_start=True,
            warm_start_n_seeds=2,
            warm_start_halving_pool_ratio=0,
            adaptive_mutation=False,
            random_seed=0,
        ),
        automl=AutoMLConfig(backend="sklearn"),
        data=DataConfig(test_size=0.15),
        report=ReportConfig(output_dir="/tmp/test_improvement_reports"),
    )


def _clf_df(n=200):
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "a": rng.standard_normal(n),
            "b": rng.standard_normal(n),
            "c": rng.standard_normal(n),
            "label": rng.integers(0, 2, n),
        }
    )


# ---------------------------------------------------------------------------
# 1. __version__
# ---------------------------------------------------------------------------

class TestVersion:
    def test_version_is_string(self):
        assert isinstance(__version__, str)

    def test_version_non_empty(self):
        assert len(__version__) > 0

    def test_version_importable_from_package(self):
        import genetic_automl
        assert hasattr(genetic_automl, "__version__")


# ---------------------------------------------------------------------------
# 2. AutoMLPipeline.summary()
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_raises_before_fit(self):
        pipeline = AutoMLPipeline(_fast_config())
        with pytest.raises(RuntimeError):
            pipeline.summary()

    def test_summary_has_required_keys(self):
        df = _clf_df()
        pipeline = AutoMLPipeline(_fast_config())
        pipeline.fit(df)
        s = pipeline.summary()
        for key in ("metric", "final_score", "generations_run", "best_chromosome_genes",
                    "best_chromosome_fitness", "preprocessing", "report_path"):
            assert key in s, f"Missing key '{key}' in summary()"

    def test_summary_final_score_matches(self):
        df = _clf_df()
        pipeline = AutoMLPipeline(_fast_config())
        pipeline.fit(df)
        assert pipeline.summary()["final_score"] == pipeline.final_score

    def test_summary_generations_run_positive(self):
        df = _clf_df()
        pipeline = AutoMLPipeline(_fast_config())
        pipeline.fit(df)
        assert pipeline.summary()["generations_run"] >= 1

    def test_summary_best_genes_non_empty(self):
        df = _clf_df()
        pipeline = AutoMLPipeline(_fast_config())
        pipeline.fit(df)
        assert len(pipeline.summary()["best_chromosome_genes"]) > 0


# ---------------------------------------------------------------------------
# 3. AutoMLPipeline.feature_importances_
# ---------------------------------------------------------------------------

class TestFeatureImportances:
    def test_none_before_fit(self):
        pipeline = AutoMLPipeline(_fast_config())
        assert pipeline.feature_importances_ is None

    def test_returns_series_after_fit(self):
        df = _clf_df()
        pipeline = AutoMLPipeline(_fast_config())
        pipeline.fit(df)
        fi = pipeline.feature_importances_
        assert fi is not None
        assert isinstance(fi, pd.Series)

    def test_importances_sum_to_one(self):
        df = _clf_df()
        pipeline = AutoMLPipeline(_fast_config())
        pipeline.fit(df)
        fi = pipeline.feature_importances_
        assert fi is not None
        assert abs(fi.sum() - 1.0) < 1e-5, f"Importances should sum to 1, got {fi.sum()}"

    def test_importances_non_negative(self):
        df = _clf_df()
        pipeline = AutoMLPipeline(_fast_config())
        pipeline.fit(df)
        fi = pipeline.feature_importances_
        assert fi is not None
        assert (fi >= 0).all(), "All importances must be non-negative"

    def test_importances_sorted_descending(self):
        df = _clf_df()
        pipeline = AutoMLPipeline(_fast_config())
        pipeline.fit(df)
        fi = pipeline.feature_importances_
        assert fi is not None
        assert list(fi.values) == sorted(fi.values, reverse=True), \
            "feature_importances_ should be sorted descending"


# ---------------------------------------------------------------------------
# 4. ReportConfig — mlflow_tracking_uri can be None
# ---------------------------------------------------------------------------

class TestReportConfig:
    def test_mlflow_none_accepted(self):
        """ReportConfig should accept None for mlflow_tracking_uri."""
        cfg = ReportConfig(mlflow_tracking_uri=None)
        assert cfg.mlflow_tracking_uri is None

    def test_mlflow_string_accepted(self):
        cfg = ReportConfig(mlflow_tracking_uri="mlflow_runs")
        assert cfg.mlflow_tracking_uri == "mlflow_runs"


# ---------------------------------------------------------------------------
# 5. CategoricalEncoder — no Pandas4Warning
# ---------------------------------------------------------------------------

class TestCategoricalEncoderPandas3:
    def test_no_pandas4_warning_on_fit(self):
        """
        Pandas 3 emits a Pandas4Warning when 'object' is passed to
        select_dtypes because string columns are included implicitly.
        After the fix, 'str' is listed explicitly and the warning is gone.
        """
        X = pd.DataFrame({"cat": ["a", "b", "c"], "num": [1.0, 2.0, 3.0]})
        enc = CategoricalEncoder("onehot")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                enc.fit(X)
            except Warning as w:
                if "Pandas4Warning" in type(w).__name__ or "select_dtypes" in str(w):
                    pytest.fail(f"Unexpected Pandas4Warning raised: {w}")
