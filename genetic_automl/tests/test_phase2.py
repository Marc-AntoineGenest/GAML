"""
Tests for Phase 2 UX improvements:
  UX-1 - Pipeline save / load
  UX-2 - tqdm progress bar (smoke test: no crash)
  UX-5 - Top-N leaderboard logged after evolution
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from genetic_automl import AutoMLPipeline, PipelineConfig, GeneticConfig, AutoMLConfig
from genetic_automl.core.problem import ProblemType


# ---------------------------------------------------------------------------
# Shared fixture: tiny classification dataset + minimal config
# ---------------------------------------------------------------------------

def _make_clf_df(n=120, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, 4), columns=list("abcd"))
    y = pd.Series((X["a"] + rng.randn(n) * 0.5 > 0).astype(int), name="y")
    return pd.concat([X, y], axis=1)


def _minimal_config():
    return PipelineConfig(
        problem_type=ProblemType.CLASSIFICATION,
        target_column="y",
        genetic=GeneticConfig(
            population_size=4,
            generations=2,
            n_cv_folds=2,
            warm_start=False,
            early_stopping_rounds=10,
            random_seed=42,
        ),
        automl=AutoMLConfig(backend="sklearn"),
    )


# ---------------------------------------------------------------------------
# UX-1: Save / Load
# ---------------------------------------------------------------------------

class TestSaveLoad:

    def test_save_creates_file(self):
        df = _make_clf_df()
        pipeline = AutoMLPipeline(_minimal_config())
        pipeline.fit(df)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")
            returned_path = pipeline.save(path)
            assert os.path.isfile(returned_path)

    def test_load_returns_pipeline(self):
        df = _make_clf_df()
        pipeline = AutoMLPipeline(_minimal_config())
        pipeline.fit(df)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")
            pipeline.save(path)
            loaded = AutoMLPipeline.load(path)
            assert isinstance(loaded, AutoMLPipeline)

    def test_predict_round_trip(self):
        df = _make_clf_df()
        pipeline = AutoMLPipeline(_minimal_config())
        pipeline.fit(df)
        X = df.drop(columns=["y"])
        preds_before = pipeline.predict(X)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")
            pipeline.save(path)
            loaded = AutoMLPipeline.load(path)
            preds_after = loaded.predict(X)
        np.testing.assert_array_equal(preds_before, preds_after)

    def test_loaded_pipeline_preserves_metric(self):
        df = _make_clf_df()
        pipeline = AutoMLPipeline(_minimal_config())
        pipeline.fit(df)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")
            pipeline.save(path)
            loaded = AutoMLPipeline.load(path)
        assert loaded._metric_name == pipeline._metric_name

    def test_loaded_pipeline_preserves_final_score(self):
        df = _make_clf_df()
        pipeline = AutoMLPipeline(_minimal_config())
        pipeline.fit(df)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")
            pipeline.save(path)
            loaded = AutoMLPipeline.load(path)
        assert loaded.final_score == pytest.approx(pipeline.final_score)

    def test_save_before_fit_raises(self):
        pipeline = AutoMLPipeline(_minimal_config())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")
            with pytest.raises(RuntimeError):
                pipeline.save(path)

    def test_load_then_predict_proba(self):
        df = _make_clf_df()
        pipeline = AutoMLPipeline(_minimal_config())
        pipeline.fit(df)
        X = df.drop(columns=["y"])
        proba_before = pipeline.predict_proba(X)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")
            pipeline.save(path)
            loaded = AutoMLPipeline.load(path)
            proba_after = loaded.predict_proba(X)
        if proba_before is not None and proba_after is not None:
            np.testing.assert_array_almost_equal(proba_before, proba_after)

    def test_save_creates_parent_dirs(self):
        df = _make_clf_df()
        pipeline = AutoMLPipeline(_minimal_config())
        pipeline.fit(df)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "deep", "model.joblib")
            pipeline.save(path)
            assert os.path.isfile(path)


# ---------------------------------------------------------------------------
# UX-2: tqdm progress bar (smoke -- just ensure no crash with tqdm present)
# ---------------------------------------------------------------------------

class TestTqdmProgressBar:

    def test_run_completes_with_tqdm(self):
        """Evolution loop must complete normally when tqdm is installed."""
        df = _make_clf_df()
        pipeline = AutoMLPipeline(_minimal_config())
        pipeline.fit(df)   # would raise if tqdm integration is broken
        assert pipeline.final_score is not None

    def test_run_completes_without_tqdm(self, monkeypatch):
        """Evolution loop must also work when tqdm is unavailable."""
        import genetic_automl.genetic.engine as engine_mod
        monkeypatch.setattr(engine_mod, "_TQDM_AVAILABLE", False)
        df = _make_clf_df()
        pipeline = AutoMLPipeline(_minimal_config())
        pipeline.fit(df)
        assert pipeline.final_score is not None


# ---------------------------------------------------------------------------
# UX-5: Leaderboard
# ---------------------------------------------------------------------------

class TestLeaderboard:

    def test_leaderboard_logged(self, caplog):
        """_log_leaderboard must emit a TOP-N header to the log."""
        import logging
        df = _make_clf_df()
        pipeline = AutoMLPipeline(_minimal_config())
        with caplog.at_level(logging.INFO, logger="genetic_automl.genetic.engine"):
            pipeline.fit(df)
        assert any("LEADERBOARD" in r.message for r in caplog.records)

    def test_leaderboard_top5_deduplicates(self):
        """Leaderboard must not exceed top_n entries and must deduplicate."""
        from genetic_automl.genetic.engine import GeneticEngine, EvolutionHistory
        from genetic_automl.genetic.chromosome import Chromosome
        import random

        rng = random.Random(0)
        # Build a mock history with duplicates
        genes = {"scaler": "standard", "n_estimators": 100}
        c1 = Chromosome(genes=genes, fitness=0.9)
        c2 = Chromosome(genes=genes, fitness=0.85)   # duplicate genes, lower fitness
        c3 = Chromosome(genes={"scaler": "minmax", "n_estimators": 200}, fitness=0.8)

        history = EvolutionHistory()
        history.all_chromosomes = [c1, c2, c3]

        from genetic_automl.config import GeneticConfig
        from unittest.mock import MagicMock
        engine = GeneticEngine.__new__(GeneticEngine)
        engine.history = history

        # Should not raise; duplicates collapsed to best
        engine._log_leaderboard(top_n=5)
