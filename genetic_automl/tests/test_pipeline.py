"""Integration tests for the full AutoML pipeline."""
import numpy as np
import pandas as pd
import pytest

from genetic_automl import AutoMLPipeline
from genetic_automl.config import (
    AutoMLConfig,
    DataConfig,
    GeneticConfig,
    PipelineConfig,
    ReportConfig,
)
from genetic_automl.core.data import DataManager
from genetic_automl.core.problem import ProblemType


# ---------------------------------------------------------------------------
# DataManager
# ---------------------------------------------------------------------------

class TestDataManager:
    def test_three_way_split_no_overlap(self, clf_df):
        dm = DataManager("label", ProblemType.CLASSIFICATION, test_size=0.15, val_size=0.2)
        dm.validate(clf_df)
        train, val, test = dm.three_way_split(clf_df)
        train_idx = set(train.index)
        val_idx   = set(val.index)
        test_idx  = set(test.index)
        assert len(train_idx & test_idx) == 0, "train/test overlap"
        assert len(val_idx & test_idx) == 0,   "val/test overlap"
        assert len(train_idx & val_idx) == 0,  "train/val overlap"

    def test_three_way_split_full_coverage(self, clf_df):
        dm = DataManager("label", ProblemType.CLASSIFICATION)
        dm.validate(clf_df)
        train, val, test = dm.three_way_split(clf_df)
        assert len(train) + len(val) + len(test) == len(clf_df)

    def test_test_size_respected(self, clf_df):
        dm = DataManager("label", ProblemType.CLASSIFICATION, test_size=0.2)
        dm.validate(clf_df)
        train, val, test = dm.three_way_split(clf_df)
        actual_ratio = len(test) / len(clf_df)
        assert abs(actual_ratio - 0.2) < 0.05  # within 5%


# ---------------------------------------------------------------------------
# Full pipeline (fast config)
# ---------------------------------------------------------------------------

def _fast_config(problem_type, target_col):
    return PipelineConfig(
        problem_type=problem_type,
        target_column=target_col,
        genetic=GeneticConfig(
            population_size=4,
            generations=2,
            early_stopping_rounds=2,
            n_cv_folds=2,
            warm_start=True,
            warm_start_n_seeds=2,
            warm_start_halving_pool_ratio=0,  # disable halving for speed
            adaptive_mutation=False,
            random_seed=42,
        ),
        automl=AutoMLConfig(backend="sklearn"),
        data=DataConfig(test_size=0.15),
        report=ReportConfig(output_dir="/tmp/test_reports"),
    )


class TestAutoMLPipelineClassification:
    def test_fit_returns_self(self, clf_df):
        config = _fast_config(ProblemType.CLASSIFICATION, "label")
        pipeline = AutoMLPipeline(config)
        result = pipeline.fit(clf_df)
        assert result is pipeline

    def test_final_score_is_positive(self, clf_df):
        config = _fast_config(ProblemType.CLASSIFICATION, "label")
        pipeline = AutoMLPipeline(config)
        pipeline.fit(clf_df)
        assert pipeline.final_score > 0.0

    def test_predict_shape(self, clf_df):
        config = _fast_config(ProblemType.CLASSIFICATION, "label")
        pipeline = AutoMLPipeline(config)
        pipeline.fit(clf_df)
        preds = pipeline.predict(clf_df.head(20))
        assert preds.shape == (20,)

    def test_predict_proba_shape(self, clf_df):
        config = _fast_config(ProblemType.CLASSIFICATION, "label")
        pipeline = AutoMLPipeline(config)
        pipeline.fit(clf_df)
        proba = pipeline.predict_proba(clf_df.head(20))
        if proba is not None:
            assert proba.shape[0] == 20

    def test_report_generated(self, clf_df):
        import os
        config = _fast_config(ProblemType.CLASSIFICATION, "label")
        pipeline = AutoMLPipeline(config)
        pipeline.fit(clf_df)
        assert pipeline.report_path is not None
        assert os.path.exists(pipeline.report_path)

    def test_history_has_generations(self, clf_df):
        config = _fast_config(ProblemType.CLASSIFICATION, "label")
        pipeline = AutoMLPipeline(config)
        pipeline.fit(clf_df)
        assert len(pipeline.history.generations) >= 1

    def test_history_has_diversity_stats(self, clf_df):
        config = _fast_config(ProblemType.CLASSIFICATION, "label")
        pipeline = AutoMLPipeline(config)
        pipeline.fit(clf_df)
        for g in pipeline.history.generations:
            assert hasattr(g, "mean_hamming")
            assert 0.0 <= g.mean_hamming <= 1.0


class TestAutoMLPipelineRegression:
    def test_regression_fit_and_score(self, reg_df):
        config = _fast_config(ProblemType.REGRESSION, "target")
        pipeline = AutoMLPipeline(config)
        pipeline.fit(reg_df)
        # MSE is negated in the GA, stored as positive in final_score
        assert pipeline.final_score is not None

    def test_regression_predict(self, reg_df):
        config = _fast_config(ProblemType.REGRESSION, "target")
        pipeline = AutoMLPipeline(config)
        pipeline.fit(reg_df)
        preds = pipeline.predict(reg_df.head(10))
        assert preds.shape == (10,)
        assert np.isfinite(preds).all()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_explicit_test_df(self, clf_df):
        """User can pass a pre-split test set."""
        n = len(clf_df)
        train = clf_df.iloc[: int(n * 0.8)]
        test = clf_df.iloc[int(n * 0.8) :]
        config = _fast_config(ProblemType.CLASSIFICATION, "label")
        pipeline = AutoMLPipeline(config)
        pipeline.fit(train, test_df=test)
        assert pipeline.final_score > 0.0

    def test_unfitted_predict_raises(self):
        config = _fast_config(ProblemType.CLASSIFICATION, "label")
        pipeline = AutoMLPipeline(config)
        with pytest.raises(RuntimeError):
            pipeline.predict(pd.DataFrame({"a": [1, 2]}))
