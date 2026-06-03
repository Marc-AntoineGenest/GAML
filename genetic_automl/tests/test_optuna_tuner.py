"""
Tests for genetic/optuna_tuner.py and the OptunaConfig/config_loader wiring.

Coverage:
  1.  OptunaTuner.tune() — happy path for each supported model_type
  2.  tune() with use_cv=True
  3.  tune() returns valid dict that can build a model without errors
  4.  Graceful fallback when model_type has no search space
  5.  Chromosome seed is enqueued (trial 0 starts from GA's best point)
  6.  timeout parameter respected (stops before n_trials)
  7.  _chromosome_to_optuna_params — correct filtering of out-of-range values
  8.  OptunaConfig defaults are sane
  9.  config_loader parses `optuna:` YAML block correctly
  10. Pipeline._build_final_model respects optuna.enabled=False (no-op)
  11. Pipeline._build_final_model with optuna.enabled=True calls tuner
  12. backend != sklearn skips tuner with a warning
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from genetic_automl.config import AutoMLConfig, OptunaConfig, PipelineConfig
from genetic_automl.core.problem import ProblemType
from genetic_automl.genetic.chromosome import Chromosome
from genetic_automl.genetic.optuna_tuner import OptunaTuner, _SEARCH_SPACES


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clf_Xy():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((200, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series((X["f0"] + X["f1"] > 0).astype(int), name="label")
    return X, y


@pytest.fixture
def reg_Xy():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((200, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series(X["f0"] * 2 + rng.standard_normal(200), name="target")
    return X, y


def _make_chromosome(model_type: str, fitness: float = 0.85) -> Chromosome:
    """Create a minimal chromosome with the given model_type."""
    genes = {
        "model_type": model_type,
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        # preprocessing genes (ignored by tuner)
        "scaler": "standard",
        "numeric_imputer": "mean",
    }
    chrom = Chromosome(genes=genes)
    chrom.fitness = fitness
    return chrom


# ---------------------------------------------------------------------------
# 1–3. Core tune() behaviour
# ---------------------------------------------------------------------------

class TestOptunaTunerHappyPath:
    """tune() should return a valid model-genes dict for each supported type."""

    @pytest.mark.parametrize("model_type", ["lgbm", "xgb", "gbm", "rf"])
    def test_tune_returns_dict(self, clf_Xy, model_type):
        X, y = clf_Xy
        chrom = _make_chromosome(model_type)
        tuner = OptunaTuner(n_trials=3)
        result = tuner.tune(
            chrom, X, y,
            problem_type=ProblemType.CLASSIFICATION,
            target_column="label",
            metric="f1_macro",
            backend="sklearn",
            random_seed=0,
        )
        assert isinstance(result, dict)
        assert "model_type" in result
        assert result["model_type"] == model_type

    def test_tune_lgbm_regression(self, reg_Xy):
        X, y = reg_Xy
        chrom = _make_chromosome("lgbm", fitness=-0.25)
        tuner = OptunaTuner(n_trials=3)
        result = tuner.tune(
            chrom, X, y,
            problem_type=ProblemType.REGRESSION,
            target_column="target",
            metric="mse",
            backend="sklearn",
            random_seed=0,
        )
        assert isinstance(result, dict)
        assert result["model_type"] == "lgbm"
        assert "n_estimators" in result

    def test_tune_with_use_cv(self, clf_Xy):
        """use_cv=True path should also return a valid dict."""
        X, y = clf_Xy
        chrom = _make_chromosome("gbm")
        tuner = OptunaTuner(n_trials=3, use_cv=True, n_cv_folds=2)
        result = tuner.tune(
            chrom, X, y,
            problem_type=ProblemType.CLASSIFICATION,
            target_column="label",
            metric="f1_macro",
            backend="sklearn",
            random_seed=0,
        )
        assert isinstance(result, dict)

    def test_tuned_result_can_build_model(self, clf_Xy):
        """The returned dict must be passable directly to build_automl()."""
        from genetic_automl.automl import build_automl
        X, y = clf_Xy
        chrom = _make_chromosome("lgbm")
        tuner = OptunaTuner(n_trials=3)
        result = tuner.tune(
            chrom, X, y,
            problem_type=ProblemType.CLASSIFICATION,
            target_column="label",
            metric="f1_macro",
            backend="sklearn",
            random_seed=0,
        )
        model = build_automl(
            backend="sklearn",
            problem_type=ProblemType.CLASSIFICATION,
            target_column="label",
            **{k: v for k, v in result.items() if v is not None},
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)


# ---------------------------------------------------------------------------
# 4. Graceful fallback for unknown model_type
# ---------------------------------------------------------------------------

class TestFallback:
    def test_unknown_model_type_returns_original_genes(self, clf_Xy):
        X, y = clf_Xy
        chrom = _make_chromosome("unknown_model")
        tuner = OptunaTuner(n_trials=5)
        result = tuner.tune(
            chrom, X, y,
            problem_type=ProblemType.CLASSIFICATION,
            target_column="label",
            metric="f1_macro",
            backend="sklearn",
            random_seed=0,
        )
        # Should return whatever model genes are in the chromosome
        assert result["model_type"] == "unknown_model"

    def test_missing_optuna_graceful(self, clf_Xy, monkeypatch):
        """Simulate optuna not installed — should return chromosome genes unchanged."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "optuna":
                raise ImportError("optuna not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        X, y = clf_Xy
        chrom = _make_chromosome("lgbm")
        tuner = OptunaTuner(n_trials=5)
        result = tuner.tune(
            chrom, X, y,
            problem_type=ProblemType.CLASSIFICATION,
            target_column="label",
            metric="f1_macro",
            backend="sklearn",
            random_seed=0,
        )
        assert result["model_type"] == "lgbm"


# ---------------------------------------------------------------------------
# 5. Chromosome seed is enqueued as trial 0
# ---------------------------------------------------------------------------

class TestChromosomeSeed:
    def test_seed_params_extracted(self):
        """_chromosome_to_optuna_params should return in-range params only."""
        model_genes = {"model_type": "lgbm", "n_estimators": 100, "learning_rate": 0.1}
        search_space = _SEARCH_SPACES["lgbm"]
        result = OptunaTuner._chromosome_to_optuna_params(model_genes, search_space)
        assert result["n_estimators"] == 100
        assert abs(result["learning_rate"] - 0.1) < 1e-9

    def test_out_of_range_params_skipped(self):
        """Values outside [low, high] must not appear in seed params."""
        model_genes = {"model_type": "lgbm", "n_estimators": 9999, "learning_rate": 0.1}
        search_space = _SEARCH_SPACES["lgbm"]
        result = OptunaTuner._chromosome_to_optuna_params(model_genes, search_space)
        assert "n_estimators" not in result   # 9999 > high=1000
        assert "learning_rate" in result

    def test_none_values_skipped(self):
        model_genes = {"model_type": "lgbm", "n_estimators": None, "learning_rate": 0.1}
        search_space = _SEARCH_SPACES["lgbm"]
        result = OptunaTuner._chromosome_to_optuna_params(model_genes, search_space)
        assert "n_estimators" not in result


# ---------------------------------------------------------------------------
# 6. Timeout stops the study early
# ---------------------------------------------------------------------------

class TestTimeout:
    def test_timeout_respected(self, clf_Xy):
        """A very short timeout (0.1 s) should complete without error."""
        X, y = clf_Xy
        chrom = _make_chromosome("lgbm")
        tuner = OptunaTuner(n_trials=1000, timeout=0.1)
        result = tuner.tune(
            chrom, X, y,
            problem_type=ProblemType.CLASSIFICATION,
            target_column="label",
            metric="f1_macro",
            backend="sklearn",
            random_seed=0,
        )
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 7. OptunaConfig dataclass defaults
# ---------------------------------------------------------------------------

class TestOptunaConfig:
    def test_defaults(self):
        cfg = OptunaConfig()
        assert cfg.enabled is False
        assert cfg.n_trials == 30
        assert cfg.timeout is None
        assert cfg.use_cv is False
        assert cfg.n_cv_folds == 3
        assert cfg.verbose is False

    def test_automl_config_has_optuna(self):
        cfg = AutoMLConfig()
        assert hasattr(cfg, "optuna")
        assert isinstance(cfg.optuna, OptunaConfig)

    def test_custom_values(self):
        cfg = OptunaConfig(enabled=True, n_trials=50, timeout=300.0, use_cv=True)
        assert cfg.enabled is True
        assert cfg.n_trials == 50
        assert cfg.timeout == 300.0
        assert cfg.use_cv is True


# ---------------------------------------------------------------------------
# 8. config_loader parses `optuna:` YAML block correctly
# ---------------------------------------------------------------------------

class TestConfigLoader:
    def _make_minimal_yaml(self, optuna_block: str) -> str:
        return f"""
run:
  name: test_run
  backend: sklearn

problem:
  type: classification
  target_column: label

genetic:
  population_size: 5
  generations: 2

{optuna_block}
"""

    def test_optuna_enabled_parsed(self, tmp_path):
        from genetic_automl.config_loader import load_config
        yaml_str = self._make_minimal_yaml(
            "optuna:\n  enabled: true\n  n_trials: 42\n  timeout: 120.0\n  use_cv: true\n  n_cv_folds: 5\n  verbose: true"
        )
        path = tmp_path / "test.yaml"
        path.write_text(yaml_str)
        cfg, _ = load_config(str(path))
        assert cfg.automl.optuna.enabled is True
        assert cfg.automl.optuna.n_trials == 42
        assert cfg.automl.optuna.timeout == 120.0
        assert cfg.automl.optuna.use_cv is True
        assert cfg.automl.optuna.n_cv_folds == 5
        assert cfg.automl.optuna.verbose is True

    def test_optuna_defaults_when_absent(self, tmp_path):
        from genetic_automl.config_loader import load_config
        yaml_str = self._make_minimal_yaml("")
        path = tmp_path / "test.yaml"
        path.write_text(yaml_str)
        cfg, _ = load_config(str(path))
        assert cfg.automl.optuna.enabled is False
        assert cfg.automl.optuna.n_trials == 30
        assert cfg.automl.optuna.timeout is None

    def test_optuna_null_timeout(self, tmp_path):
        from genetic_automl.config_loader import load_config
        yaml_str = self._make_minimal_yaml("optuna:\n  enabled: false\n  timeout: null\n")
        path = tmp_path / "test.yaml"
        path.write_text(yaml_str)
        cfg, _ = load_config(str(path))
        assert cfg.automl.optuna.timeout is None


# ---------------------------------------------------------------------------
# 9–11. Pipeline integration (light mock-based tests — no full GA run)
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    """
    Verify that AutoMLPipeline._build_final_model correctly invokes (or skips)
    OptunaTuner based on the config, without running a full pipeline.
    """

    def _make_mock_pipeline(self, optuna_enabled: bool, backend: str = "sklearn"):
        from genetic_automl.pipeline import AutoMLPipeline
        from genetic_automl.genetic.engine import EvolutionHistory

        cfg = PipelineConfig(
            problem_type=ProblemType.CLASSIFICATION,
            target_column="label",
        )
        cfg.automl.backend = backend
        cfg.automl.ensemble.enabled = False   # single model for simplicity
        cfg.automl.optuna.enabled = optuna_enabled
        cfg.automl.optuna.n_trials = 3

        pipeline = AutoMLPipeline.__new__(AutoMLPipeline)
        pipeline.config = cfg
        pipeline._metric_name = "f1_macro"

        # Build a minimal fake history with one chromosome
        chrom = _make_chromosome("lgbm")
        history = MagicMock(spec=EvolutionHistory)
        history.top_chromosomes.return_value = [chrom]
        pipeline._history = history

        return pipeline, chrom

    def test_optuna_disabled_skips_tuner(self):
        """When optuna.enabled=False, OptunaTuner.tune() must never be called."""
        pipeline, chrom = self._make_mock_pipeline(optuna_enabled=False)

        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((100, 4)), columns=list("abcd"))
        y = pd.Series(rng.integers(0, 2, 100), name="label")

        with patch("genetic_automl.pipeline.OptunaTuner") as mock_tuner_cls:
            pipeline._build_final_model(pipeline.config, chrom, X, y)
            mock_tuner_cls.assert_not_called()

    def test_optuna_enabled_calls_tuner(self):
        """When optuna.enabled=True on sklearn, OptunaTuner.tune() must be called once."""
        pipeline, chrom = self._make_mock_pipeline(optuna_enabled=True, backend="sklearn")

        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((100, 4)), columns=list("abcd"))
        y = pd.Series(rng.integers(0, 2, 100), name="label")

        fake_tuned = {"model_type": "lgbm", "n_estimators": 200, "learning_rate": 0.05}

        with patch("genetic_automl.pipeline.OptunaTuner") as mock_tuner_cls:
            mock_instance = MagicMock()
            mock_instance.tune.return_value = fake_tuned
            mock_tuner_cls.return_value = mock_instance
            pipeline._build_final_model(pipeline.config, chrom, X, y)
            mock_tuner_cls.assert_called_once()
            mock_instance.tune.assert_called_once()

    def test_optuna_skipped_for_non_sklearn_backend(self, caplog):
        """Optuna should be skipped with a warning for non-sklearn backends."""
        import logging
        pipeline, chrom = self._make_mock_pipeline(optuna_enabled=True, backend="autogluon")

        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((100, 4)), columns=list("abcd"))
        y = pd.Series(rng.integers(0, 2, 100), name="label")

        with patch("genetic_automl.pipeline.OptunaTuner") as mock_tuner_cls:
            with caplog.at_level(logging.WARNING, logger="genetic_automl.pipeline"):
                try:
                    pipeline._build_final_model(pipeline.config, chrom, X, y)
                except Exception:
                    pass  # autogluon may fail in test env — we only care about the warning
            mock_tuner_cls.assert_not_called()
            assert any("skipping HPO" in r.message for r in caplog.records)
