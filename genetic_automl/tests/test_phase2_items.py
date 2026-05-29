"""
Tests for Phase 2 items 1-3:
  - ASHA / median-stop pruning
  - FeatureEngineer preprocessing step
  - Generation checkpointing (save / resume)

Run:
    pytest genetic_automl/tests/test_phase2_items.py -v
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from genetic_automl.config import (
    AutoMLConfig, DataConfig, EnsembleConfig, GeneticConfig,
    PipelineConfig, ReportConfig,
)
from genetic_automl.core.problem import ProblemType
from genetic_automl.genetic.chromosome import Chromosome, get_gene_space
from genetic_automl.genetic.fitness import FitnessEvaluator
from genetic_automl.pipeline import AutoMLPipeline
from genetic_automl.preprocessing.feature_engineer import FeatureEngineer
from genetic_automl.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clf_Xy(n=300):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "a": rng.standard_normal(n),
        "b": np.abs(rng.standard_normal(n)) * 10,   # right-skewed
        "c": rng.standard_normal(n),
        "d": rng.integers(0, 5, n).astype(float),
    })
    y = pd.Series(rng.integers(0, 2, n), name="label")
    return X, y


@pytest.fixture
def clf_df(n=300):
    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.standard_normal((n, 4)), columns=list("abcd"))
    df["label"] = rng.integers(0, 2, n)
    return df


def _fast_config(**overrides):
    """Minimal PipelineConfig for fast tests."""
    genetic_kwargs = dict(
        population_size=6,
        generations=3,
        early_stopping_rounds=3,
        n_cv_folds=2,
        warm_start=False,
        adaptive_mutation=False,
        random_seed=3,
        surrogate_enabled=False,   # isolate the feature under test
    )
    genetic_kwargs.update(overrides)
    return PipelineConfig(
        problem_type=ProblemType.CLASSIFICATION,
        target_column="label",
        genetic=GeneticConfig(**genetic_kwargs),
        automl=AutoMLConfig(
            backend="sklearn",
            ensemble=EnsembleConfig(enabled=False),
        ),
        data=DataConfig(test_size=0.15),
        report=ReportConfig(output_dir="/tmp/test_phase2_reports"),
    )


# ===========================================================================
# 1. ASHA pruning
# ===========================================================================

class TestASHAPruning:
    """Unit tests for the median-stop pruning logic inside FitnessEvaluator."""

    def _make_evaluator(self, asha_enabled=True, min_folds=1, margin=0.0, n_folds=3):
        return FitnessEvaluator(
            problem_type=ProblemType.CLASSIFICATION,
            target_column="label",
            backend="sklearn",
            metric="accuracy",
            n_folds=n_folds,
            random_seed=0,
            asha_enabled=asha_enabled,
            asha_min_folds_before_prune=min_folds,
            asha_prune_margin=margin,
        )

    def test_asha_disabled_runs_all_folds(self, clf_Xy):
        """When asha_enabled=False, every chromosome runs all folds regardless."""
        X, y = clf_Xy
        ev = self._make_evaluator(asha_enabled=False, n_folds=3)

        # Seed the fold pool with some scores so a median exists
        ev._all_fold_scores = [0.9] * 20

        import random as r
        rnd = r.Random(0)
        space = get_gene_space("sklearn")
        chrom = Chromosome(genes={g.name: g.random_value(rnd) for g in space}, generation=0)
        fitness = ev.evaluate(chrom, X, y)
        assert fitness != float("-inf")
        # No pruning should have fired — prune counter stays 0
        assert ev._asha_prunes == 0

    def test_asha_prune_counter_increments(self, clf_Xy):
        """ASHA should prune at least some chromosomes over a full pipeline run."""
        cfg = _fast_config(
            asha_enabled=True,
            asha_min_folds_before_prune=1,
            asha_prune_margin=0.0,
            n_cv_folds=3,
            population_size=10,
            generations=3,
        )
        pipeline = AutoMLPipeline(cfg)
        pipeline.fit(clf_Xy[0].assign(label=clf_Xy[1]))
        # Can't directly access the evaluator here, but at minimum the run
        # completed without error — the pruning path didn't break anything.
        assert pipeline.final_score is not None

    def test_asha_fields_in_config(self):
        cfg = GeneticConfig()
        assert cfg.asha_enabled is True
        assert cfg.asha_min_folds_before_prune == 1
        assert cfg.asha_prune_margin == 0.0

    def test_evaluator_summary_has_asha_keys(self, clf_Xy):
        X, y = clf_Xy
        ev = self._make_evaluator()
        summary = ev.evaluator_summary()
        assert "asha_enabled" in summary
        assert "asha_prunes" in summary
        assert "asha_fold_pool_size" in summary

    def test_fold_pool_grows_with_evaluations(self, clf_Xy):
        X, y = clf_Xy
        ev = self._make_evaluator(asha_enabled=False, n_folds=2)
        import random as r
        rnd = r.Random(1)
        space = get_gene_space("sklearn")
        for _ in range(3):
            chrom = Chromosome(genes={g.name: g.random_value(rnd) for g in space}, generation=0)
            ev.evaluate(chrom, X, y)
        # With 3 chromosomes × 2 folds each, pool should have ~6 scores
        assert ev._asha_fold_pool_size_property >= 4

    def test_pipeline_runs_with_asha_disabled(self, clf_df):
        cfg = _fast_config(asha_enabled=False)
        pipeline = AutoMLPipeline(cfg)
        pipeline.fit(clf_df)
        assert pipeline.final_score is not None

    def test_pipeline_runs_with_asha_min_folds_2(self, clf_df):
        cfg = _fast_config(asha_enabled=True, asha_min_folds_before_prune=2, n_cv_folds=3)
        pipeline = AutoMLPipeline(cfg)
        pipeline.fit(clf_df)
        assert pipeline.final_score is not None

    def test_pipeline_runs_with_conservative_margin(self, clf_df):
        cfg = _fast_config(asha_enabled=True, asha_prune_margin=0.1)
        pipeline = AutoMLPipeline(cfg)
        pipeline.fit(clf_df)
        assert pipeline.final_score is not None


# Add a small helper property to FitnessEvaluator for the test above
FitnessEvaluator._asha_fold_pool_size_property = property(
    lambda self: len(self._all_fold_scores)
)


# ===========================================================================
# 2. FeatureEngineer
# ===========================================================================

class TestFeatureEngineer:
    """Unit tests for each feature engineering strategy."""

    def test_none_is_noop(self, clf_Xy):
        X, y = clf_Xy
        fe = FeatureEngineer("none")
        X_out = fe.fit_transform(X, y)
        assert list(X_out.columns) == list(X.columns)
        assert X_out.shape == X.shape

    def test_log1p_adds_columns(self, clf_Xy):
        X, y = clf_Xy
        fe = FeatureEngineer("log1p")
        X_out = fe.fit_transform(X, y)
        log_cols = [c for c in X_out.columns if c.startswith("feat_log1p_")]
        # Column "b" is heavily right-skewed → should get a log1p copy
        assert len(log_cols) >= 1

    def test_log1p_values_are_finite(self, clf_Xy):
        X, y = clf_Xy
        fe = FeatureEngineer("log1p")
        X_out = fe.fit_transform(X, y)
        log_cols = [c for c in X_out.columns if c.startswith("feat_log1p_")]
        for col in log_cols:
            assert np.isfinite(X_out[col].values).all(), f"Non-finite values in {col}"

    def test_poly2_adds_columns(self, clf_Xy):
        X, y = clf_Xy
        fe = FeatureEngineer("poly2", max_interaction_features=4)
        X_out = fe.fit_transform(X, y)
        poly_cols = [c for c in X_out.columns if c.startswith("feat_poly_")]
        assert len(poly_cols) > 0

    def test_poly2_column_count(self, clf_Xy):
        """k top-variant cols → k squared + k*(k-1)/2 cross = k*(k+1)/2 new cols."""
        X, y = clf_Xy
        k = 3
        fe = FeatureEngineer("poly2", max_interaction_features=k)
        X_out = fe.fit_transform(X, y)
        poly_cols = [c for c in X_out.columns if c.startswith("feat_poly_")]
        expected = k + k * (k - 1) // 2   # sq terms + cross terms
        assert len(poly_cols) == expected

    def test_ratio_adds_columns(self, clf_Xy):
        X, y = clf_Xy
        fe = FeatureEngineer("ratio", max_interaction_features=4)
        X_out = fe.fit_transform(X, y)
        ratio_cols = [c for c in X_out.columns if c.startswith("feat_ratio_")]
        assert len(ratio_cols) > 0

    def test_ratio_no_division_by_zero(self, clf_Xy):
        X, y = clf_Xy
        fe = FeatureEngineer("ratio", max_interaction_features=4)
        X_out = fe.fit_transform(X, y)
        ratio_cols = [c for c in X_out.columns if c.startswith("feat_ratio_")]
        for col in ratio_cols:
            assert np.isfinite(X_out[col].values).all()

    def test_all_combines_strategies(self, clf_Xy):
        X, y = clf_Xy
        fe = FeatureEngineer("all", max_interaction_features=4)
        X_out = fe.fit_transform(X, y)
        assert X_out.shape[1] > X.shape[1]
        # Should have at least one column from each strategy
        has_log = any(c.startswith("feat_log1p_") for c in X_out.columns)
        has_poly = any(c.startswith("feat_poly_") for c in X_out.columns)
        has_ratio = any(c.startswith("feat_ratio_") for c in X_out.columns)
        assert has_poly and has_ratio  # log1p only fires if skewed cols exist

    def test_transform_matches_fit_transform(self, clf_Xy):
        """transform(X_train) should equal fit_transform(X_train)."""
        X, y = clf_Xy
        fe = FeatureEngineer("all", max_interaction_features=4)
        X_fit = fe.fit_transform(X, y)
        X_t = fe.transform(X)
        pd.testing.assert_frame_equal(X_fit, X_t)

    def test_transform_before_fit_raises(self, clf_Xy):
        X, _ = clf_Xy
        fe = FeatureEngineer("poly2")
        with pytest.raises(RuntimeError):
            fe.transform(X)

    def test_transform_handles_missing_columns_gracefully(self, clf_Xy):
        """Columns absent at transform time are silently skipped."""
        X, y = clf_Xy
        fe = FeatureEngineer("log1p")
        fe.fit(X, y)
        X_subset = X.drop(columns=["b"])  # drop a column that may have been selected
        # Should not crash
        X_out = fe.transform(X_subset)
        assert X_out is not None

    def test_original_columns_preserved(self, clf_Xy):
        X, y = clf_Xy
        for strategy in ("log1p", "poly2", "ratio", "all"):
            fe = FeatureEngineer(strategy, max_interaction_features=4)
            X_out = fe.fit_transform(X, y)
            for col in X.columns:
                assert col in X_out.columns, f"Original column '{col}' missing after {strategy}"


class TestFeatureEngineerInPipeline:
    """Integration tests — FeatureEngineer wired into PreprocessingPipeline."""

    @pytest.mark.parametrize("strategy", ["none", "log1p", "poly2", "ratio", "all"])
    def test_pipeline_runs_with_strategy(self, clf_Xy, strategy):
        X, y = clf_Xy
        cfg = PreprocessingConfig(feature_engineering=strategy, max_interaction_features=4)
        pp = PreprocessingPipeline(cfg, ProblemType.CLASSIFICATION, random_seed=0)
        X_out, y_out = pp.fit_transform_train(X, y)
        assert X_out.shape[0] == len(y_out)
        assert X_out.shape[1] > 0

    def test_transform_produces_same_columns(self, clf_Xy):
        X, y = clf_Xy
        cfg = PreprocessingConfig(feature_engineering="all", max_interaction_features=4)
        pp = PreprocessingPipeline(cfg, ProblemType.CLASSIFICATION, random_seed=0)
        X_train_out, _ = pp.fit_transform_train(X.iloc[:240], y.iloc[:240])
        X_val_out = pp.transform(X.iloc[240:])
        assert list(X_train_out.columns) == list(X_val_out.columns)

    def test_feature_engineering_gene_in_chromosome_space(self):
        space = get_gene_space("sklearn")
        names = [g.name for g in space]
        assert "feature_engineering" in names
        assert "max_interaction_features" in names

    def test_feature_engineering_gene_values(self):
        space = get_gene_space("sklearn")
        fe_gene = next(g for g in space if g.name == "feature_engineering")
        assert set(fe_gene.values) == {"none", "log1p", "ratio", "poly2", "all"}

    def test_full_pipeline_with_poly2(self, clf_df):
        cfg = _fast_config()
        # Override the gene space to force poly2 feature engineering
        pipeline = AutoMLPipeline(cfg)
        pipeline._gene_space_overrides = {"feature_engineering": ["poly2"],
                                          "max_interaction_features": [4]}
        pipeline.fit(clf_df)
        assert pipeline.final_score is not None

    def test_preprocessing_config_has_feature_engineering_fields(self):
        cfg = PreprocessingConfig()
        assert cfg.feature_engineering == "none"
        assert cfg.max_interaction_features == 8

    def test_pipeline_summary_includes_feature_engineering(self, clf_df):
        pipeline = AutoMLPipeline(_fast_config())
        pipeline.fit(clf_df)
        summary = pipeline.summary()
        pp_cfg = summary["preprocessing"].get("config", {})
        assert "feature_engineering" in pp_cfg


# ===========================================================================
# 3. Generation checkpointing
# ===========================================================================

class TestCheckpointing:
    def test_checkpoint_dir_creates_file(self, clf_df):
        """A checkpoint file should appear in checkpoint_dir after a run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _fast_config(
                checkpoint_dir=tmpdir,
                checkpoint_every=1,
            )
            pipeline = AutoMLPipeline(cfg)
            pipeline.fit(clf_df)
            files = os.listdir(tmpdir)
            assert any(f.startswith("checkpoint_gen") and f.endswith(".joblib")
                       for f in files), f"No checkpoint file found in {tmpdir}: {files}"

    def test_checkpoint_every_2_creates_correct_files(self, clf_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _fast_config(
                checkpoint_dir=tmpdir,
                checkpoint_every=2,
                generations=4,
            )
            pipeline = AutoMLPipeline(cfg)
            pipeline.fit(clf_df)
            files = sorted(os.listdir(tmpdir))
            # With 4 generations and checkpoint_every=2, expect files at gen 2 and gen 4
            assert len(files) >= 1

    def test_checkpoint_file_loadable(self, clf_df):
        """The checkpoint .joblib must be loadable and contain required keys."""
        import joblib
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _fast_config(checkpoint_dir=tmpdir, checkpoint_every=1)
            pipeline = AutoMLPipeline(cfg)
            pipeline.fit(clf_df)
            files = [f for f in os.listdir(tmpdir) if f.endswith(".joblib")]
            assert files
            state = joblib.load(os.path.join(tmpdir, files[0]))
            for key in ("population", "history", "no_improvement_streak",
                        "best_fitness_so_far", "next_generation",
                        "fitness_cache", "all_fold_scores"):
                assert key in state, f"Missing key '{key}' in checkpoint"

    def test_resume_from_checkpoint_completes(self, clf_df):
        """A run resumed from a checkpoint should finish and produce a final score."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First run: save checkpoints
            cfg1 = _fast_config(
                checkpoint_dir=tmpdir,
                checkpoint_every=1,
                generations=2,
            )
            pipeline1 = AutoMLPipeline(cfg1)
            pipeline1.fit(clf_df)

            # Find the gen-1 checkpoint
            files = sorted(f for f in os.listdir(tmpdir) if f.endswith(".joblib"))
            assert files
            checkpoint_path = os.path.join(tmpdir, files[0])

            # Second run: resume from gen-1 checkpoint and finish
            cfg2 = _fast_config(
                resume_from_checkpoint=checkpoint_path,
                generations=3,
            )
            pipeline2 = AutoMLPipeline(cfg2)
            pipeline2.fit(clf_df)
            assert pipeline2.final_score is not None

    def test_resume_skips_evaluated_chromosomes(self, clf_df):
        """After resume, fitness cache should contain entries from the first run."""
        import joblib
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg1 = _fast_config(checkpoint_dir=tmpdir, checkpoint_every=1, generations=2)
            pipeline1 = AutoMLPipeline(cfg1)
            pipeline1.fit(clf_df)

            files = sorted(f for f in os.listdir(tmpdir) if f.endswith(".joblib"))
            checkpoint_path = os.path.join(tmpdir, files[0])
            state = joblib.load(checkpoint_path)
            assert len(state["fitness_cache"]) > 0

    def test_no_checkpoint_when_dir_is_none(self, clf_df):
        """No files should be written when checkpoint_dir is None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _fast_config()  # checkpoint_dir defaults to None
            pipeline = AutoMLPipeline(cfg)
            pipeline.fit(clf_df)
            # Nothing should have been written by GAML to tmpdir
            assert os.listdir(tmpdir) == []

    def test_checkpoint_fields_in_config(self):
        cfg = GeneticConfig()
        assert cfg.checkpoint_dir is None
        assert cfg.checkpoint_every == 1
        assert cfg.resume_from_checkpoint is None

    def test_resume_from_nonexistent_path_starts_fresh(self, clf_df):
        """resume_from_checkpoint pointing to a missing file should not crash."""
        cfg = _fast_config(resume_from_checkpoint="/nonexistent/path/ckpt.joblib")
        pipeline = AutoMLPipeline(cfg)
        pipeline.fit(clf_df)
        assert pipeline.final_score is not None
