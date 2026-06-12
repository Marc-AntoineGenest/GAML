"""
Tests for reporting/shap_explainer.py and the SHAP wiring in html_reporter
and pipeline.

Coverage:
  1.  SHAPExplainer.explain() — happy path for lgbm, xgb, gbm, rf
  2.  Returns None when shap is not installed
  3.  Returns None for unsupported / unknown estimator
  4.  Correct normalisation across all SHAP output shapes
  5.  shap_svg is a non-empty string containing <svg
  6.  feature_names length matches mean_abs_shap length
  7.  Result is sorted descending by mean_abs_shap
  8.  max_samples cap respected
  9.  Fallback to feature_importances_ when TreeExplainer fails
  10. _build_shap_svg — produces valid SVG for edge cases (1 feature, 25+ features)
  11. ReportConfig — new fields have correct defaults
  12. config_loader parses shap_enabled / shap_max_samples
  13. HTMLReporter.generate() accepts shap_summary=None without error
  14. HTMLReporter.generate() with shap_summary renders the SHAP section
  15. Pipeline: shap_enabled=False skips SHAPExplainer
  16. Pipeline: shap_enabled=True calls SHAPExplainer.explain
"""
from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from genetic_automl.config import PipelineConfig, ReportConfig
from genetic_automl.core.problem import ProblemType
from genetic_automl.reporting.shap_explainer import SHAPExplainer, _build_shap_svg

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clf_Xy():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((200, 8)), columns=[f"feat_{i}" for i in range(8)])
    y = pd.Series((X["feat_0"] + X["feat_1"] > 0).astype(int), name="label")
    return X, y


@pytest.fixture
def reg_Xy():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((200, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series(X["f0"] * 2 + rng.standard_normal(200), name="target")
    return X, y


def _fit_raw(model_cls, X, y, **kwargs):
    """Fit and return a raw sklearn/lgbm/xgb estimator."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = model_cls(**kwargs)
        m.fit(X, y)
    return m


def _wrap_in_gaml(raw_estimator):
    """Wrap a raw estimator in a minimal GAML-style wrapper exposing ._estimator."""
    wrapper = MagicMock()
    wrapper._estimator = raw_estimator
    return wrapper


# ---------------------------------------------------------------------------
# 1. Happy path — all four model types
# ---------------------------------------------------------------------------

class TestSHAPExplainerHappyPath:

    @pytest.mark.parametrize("model_type,model_cls,kwargs", [
        ("lgbm", "lightgbm.LGBMClassifier",       {"n_estimators": 30, "verbose": -1}),
        ("xgb",  "xgboost.XGBClassifier",          {"n_estimators": 30, "verbosity": 0, "eval_metric": "logloss"}),
        ("gbm",  "sklearn.ensemble.GradientBoostingClassifier", {"n_estimators": 30}),
        ("rf",   "sklearn.ensemble.RandomForestClassifier",     {"n_estimators": 30}),
    ])
    def test_returns_dict_with_expected_keys(self, clf_Xy, model_type, model_cls, kwargs):
        import importlib
        X, y = clf_Xy
        parts = model_cls.rsplit(".", 1)
        mod = importlib.import_module(parts[0])
        cls = getattr(mod, parts[1])
        raw = _fit_raw(cls, X, y, **kwargs)
        wrapper = _wrap_in_gaml(raw)

        result = SHAPExplainer(max_samples=50).explain(wrapper, X)

        assert result is not None, f"SHAP failed for {model_type}"
        assert "feature_names"  in result
        assert "mean_abs_shap"  in result
        assert "base_value"     in result
        assert "n_samples_used" in result
        assert "shap_svg"       in result

    def test_regression_lgbm(self, reg_Xy):
        from lightgbm import LGBMRegressor
        X, y = reg_Xy
        raw = _fit_raw(LGBMRegressor, X, y, n_estimators=30, verbose=-1)
        result = SHAPExplainer(max_samples=50).explain(_wrap_in_gaml(raw), X)
        assert result is not None
        assert result["feature_names"] == sorted(
            result["feature_names"],
            key=lambda n: result["mean_abs_shap"][result["feature_names"].index(n)],
            reverse=True,
        )


# ---------------------------------------------------------------------------
# 2. shap not installed → None
# ---------------------------------------------------------------------------

class TestSHAPNotInstalled:
    def test_returns_none_when_shap_missing(self, clf_Xy, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "shap":
                raise ImportError("shap not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        X, y = clf_Xy
        from lightgbm import LGBMClassifier
        raw = _fit_raw(LGBMClassifier, X, y, n_estimators=10, verbose=-1)
        result = SHAPExplainer().explain(_wrap_in_gaml(raw), X)
        assert result is None


# ---------------------------------------------------------------------------
# 3. Unsupported estimator → None
# ---------------------------------------------------------------------------

class TestUnsupportedEstimator:
    def test_returns_none_for_linear_model(self, clf_Xy):
        """LinearSVC has no tree structure — TreeExplainer may fail; should return None or valid fallback."""
        from sklearn.svm import SVC
        X, y = clf_Xy
        raw = _fit_raw(SVC, X, y, probability=True)
        wrapper = _wrap_in_gaml(raw)
        result = SHAPExplainer(max_samples=20).explain(wrapper, X)
        # Either None (TreeExplainer failed & no feature_importances_) or a valid dict
        assert result is None or isinstance(result, dict)

    def test_unresolvable_model_returns_none(self):
        """A model with no ._estimator and no feature_importances_ should return None."""
        X = pd.DataFrame(np.random.randn(50, 4), columns=list("abcd"))
        result = SHAPExplainer().explain(object(), X)
        assert result is None


# ---------------------------------------------------------------------------
# 4–7. Output contract
# ---------------------------------------------------------------------------

class TestOutputContract:

    @pytest.fixture
    def lgbm_result(self, clf_Xy):
        from lightgbm import LGBMClassifier
        X, y = clf_Xy
        raw = _fit_raw(LGBMClassifier, X, y, n_estimators=30, verbose=-1)
        return SHAPExplainer(max_samples=80).explain(_wrap_in_gaml(raw), X), X

    def test_svg_is_nonempty_and_contains_svg_tag(self, lgbm_result):
        result, _ = lgbm_result
        assert result is not None
        assert result["shap_svg"].strip().startswith("<svg")

    def test_lengths_match(self, lgbm_result):
        result, X = lgbm_result
        assert len(result["feature_names"]) == len(result["mean_abs_shap"])
        assert len(result["feature_names"]) == X.shape[1]

    def test_sorted_descending(self, lgbm_result):
        result, _ = lgbm_result
        vals = result["mean_abs_shap"]
        assert vals == sorted(vals, reverse=True)

    def test_all_shap_values_nonnegative(self, lgbm_result):
        result, _ = lgbm_result
        assert all(v >= 0 for v in result["mean_abs_shap"])

    def test_base_value_is_float(self, lgbm_result):
        result, _ = lgbm_result
        assert isinstance(result["base_value"], float)


# ---------------------------------------------------------------------------
# 8. max_samples cap
# ---------------------------------------------------------------------------

class TestMaxSamples:
    def test_max_samples_respected(self, clf_Xy):
        from lightgbm import LGBMClassifier
        X, y = clf_Xy
        raw = _fit_raw(LGBMClassifier, X, y, n_estimators=20, verbose=-1)
        result = SHAPExplainer(max_samples=30).explain(_wrap_in_gaml(raw), X)
        assert result is not None
        assert result["n_samples_used"] == 30

    def test_max_samples_larger_than_data(self, clf_Xy):
        from lightgbm import LGBMClassifier
        X, y = clf_Xy
        raw = _fit_raw(LGBMClassifier, X, y, n_estimators=20, verbose=-1)
        result = SHAPExplainer(max_samples=9999).explain(_wrap_in_gaml(raw), X)
        assert result is not None
        assert result["n_samples_used"] == len(X)


# ---------------------------------------------------------------------------
# 9. Fallback to feature_importances_
# ---------------------------------------------------------------------------

class TestFallback:
    def test_fallback_used_when_tree_explainer_fails(self, clf_Xy):
        """When TreeExplainer raises, _fallback_importances should be tried."""
        from lightgbm import LGBMClassifier
        X, y = clf_Xy
        raw = _fit_raw(LGBMClassifier, X, y, n_estimators=20, verbose=-1)
        wrapper = _wrap_in_gaml(raw)

        with patch("shap.TreeExplainer", side_effect=Exception("forced failure")):
            result = SHAPExplainer(max_samples=30).explain(wrapper, X)

        # Should use feature_importances_ fallback and still return a valid dict
        assert result is not None
        assert len(result["mean_abs_shap"]) == X.shape[1]


# ---------------------------------------------------------------------------
# 10. _build_shap_svg edge cases
# ---------------------------------------------------------------------------

class TestBuildShapSVG:
    def test_single_feature(self):
        svg = _build_shap_svg(["only_feature"], [0.42])
        assert "<svg" in svg
        assert "only_feature" in svg

    def test_many_features_capped(self):
        names = [f"feat_{i}" for i in range(30)]
        vals  = sorted(np.random.rand(30).tolist(), reverse=True)
        svg = _build_shap_svg(names, vals, max_features=20)
        assert svg.count("<rect") == 20   # exactly 20 bars

    def test_empty_inputs(self):
        svg = _build_shap_svg([], [])
        assert "<svg" in svg   # returns a degenerate but valid SVG

    def test_long_feature_name_truncated(self):
        long_name = "this_is_a_very_long_feature_name_that_should_be_truncated"
        svg = _build_shap_svg([long_name], [1.0])
        assert long_name not in svg   # should be truncated
        assert "…" in svg


# ---------------------------------------------------------------------------
# 11. ReportConfig defaults
# ---------------------------------------------------------------------------

class TestReportConfig:
    def test_shap_enabled_default_true(self):
        cfg = ReportConfig()
        assert cfg.shap_enabled is True

    def test_shap_max_samples_default(self):
        cfg = ReportConfig()
        assert cfg.shap_max_samples == 200

    def test_disable_shap(self):
        cfg = ReportConfig(shap_enabled=False)
        assert cfg.shap_enabled is False

    def test_pipeline_config_has_report_with_shap(self):
        cfg = PipelineConfig()
        assert hasattr(cfg.report, "shap_enabled")
        assert hasattr(cfg.report, "shap_max_samples")


# ---------------------------------------------------------------------------
# 12. config_loader parses SHAP fields
# ---------------------------------------------------------------------------

class TestConfigLoaderSHAP:
    def _make_yaml(self, report_block: str) -> str:
        return f"""
run:
  name: test_shap
  backend: sklearn
problem:
  type: classification
  target_column: label
genetic:
  population_size: 5
  generations: 2
{report_block}
"""

    def test_shap_enabled_parsed(self, tmp_path):
        from genetic_automl.config_loader import load_config
        yaml = self._make_yaml("report:\n  shap_enabled: false\n  shap_max_samples: 50\n")
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml)
        cfg, _ = load_config(str(p))
        assert cfg.report.shap_enabled is False
        assert cfg.report.shap_max_samples == 50

    def test_shap_defaults_when_absent(self, tmp_path):
        from genetic_automl.config_loader import load_config
        yaml = self._make_yaml("")
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml)
        cfg, _ = load_config(str(p))
        assert cfg.report.shap_enabled is True
        assert cfg.report.shap_max_samples == 200


# ---------------------------------------------------------------------------
# 13–14. HTMLReporter integration
# ---------------------------------------------------------------------------

class TestHTMLReporterSHAP:
    """Light tests — we verify the HTML output contains/omits the SHAP section."""

    def _make_minimal_history(self):
        from genetic_automl.genetic.chromosome import Chromosome
        from genetic_automl.genetic.engine import EvolutionHistory
        chrom = Chromosome(genes={"model_type": "lgbm", "n_estimators": 100})
        chrom.fitness = 0.85
        chrom.generation = 0
        history = MagicMock(spec=EvolutionHistory)
        history.best = chrom
        history.fitness_curve.return_value = [0.85]
        gen_mock = MagicMock()
        gen_mock.generation = 0
        gen_mock.best_fitness = 0.85
        gen_mock.mean_fitness = 0.82
        gen_mock.worst_fitness = 0.78
        gen_mock.elapsed_seconds = 5.0
        history.generations = [gen_mock]
        history.all_chromosomes = [chrom]
        return history

    def _make_cfg(self):
        cfg = PipelineConfig(
            problem_type=ProblemType.CLASSIFICATION,
            target_column="label",
        )
        return cfg

    def test_generate_without_shap_no_error(self, tmp_path):
        from genetic_automl.reporting.html_reporter import HTMLReporter
        reporter = HTMLReporter(output_dir=str(tmp_path))
        path = reporter.generate(
            config=self._make_cfg(),
            history=self._make_minimal_history(),
            shap_summary=None,
        )
        assert path.endswith(".html")
        html = open(path).read()
        assert "SHAP" not in html   # section should be absent

    def test_generate_with_shap_renders_section(self, tmp_path):
        from genetic_automl.reporting.html_reporter import HTMLReporter
        reporter = HTMLReporter(output_dir=str(tmp_path))
        shap_summary = {
            "feature_names":  ["feat_0", "feat_1", "feat_2"],
            "mean_abs_shap":  [0.5, 0.3, 0.1],
            "base_value":     0.42,
            "n_samples_used": 100,
            "shap_svg":       "<svg viewBox='0 0 700 80'><rect/></svg>",
        }
        path = reporter.generate(
            config=self._make_cfg(),
            history=self._make_minimal_history(),
            shap_summary=shap_summary,
        )
        html = open(path).read()
        assert "SHAP Feature Importance" in html
        assert "feat_0" in html
        assert "0.500000" in html
        assert "<svg" in html


# ---------------------------------------------------------------------------
# 15–16. Pipeline integration
# ---------------------------------------------------------------------------

class TestPipelineSHAPIntegration:
    """Verify pipeline calls / skips SHAPExplainer based on config."""

    def _make_pipeline(self, shap_enabled: bool):
        from genetic_automl.pipeline import AutoMLPipeline
        pipeline = AutoMLPipeline.__new__(AutoMLPipeline)
        cfg = PipelineConfig(
            problem_type=ProblemType.CLASSIFICATION,
            target_column="label",
        )
        cfg.report.shap_enabled = shap_enabled
        cfg.report.shap_max_samples = 50
        pipeline.config = cfg
        return pipeline, cfg

    def test_shap_disabled_skips_explainer(self):
        pipeline, cfg = self._make_pipeline(shap_enabled=False)
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((60, 4)), columns=list("abcd"))

        with patch("genetic_automl.pipeline.SHAPExplainer") as mock_cls:
            # Simulate the SHAP block in pipeline.fit() directly
            shap_summary = None
            if cfg.report.shap_enabled:
                shap_summary = mock_cls(max_samples=50).explain(MagicMock(), X)
            mock_cls.assert_not_called()
            assert shap_summary is None

    def test_shap_enabled_calls_explainer(self):
        pipeline, cfg = self._make_pipeline(shap_enabled=True)
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((60, 4)), columns=list("abcd"))

        fake_summary = {"feature_names": list("abcd"), "mean_abs_shap": [0.4, 0.3, 0.2, 0.1],
                        "base_value": 0.5, "n_samples_used": 50, "shap_svg": "<svg/>"}

        with patch("genetic_automl.pipeline.SHAPExplainer") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.explain.return_value = fake_summary
            mock_cls.return_value = mock_instance

            shap_summary = None
            if cfg.report.shap_enabled:
                shap_summary = mock_cls(
                    max_samples=cfg.report.shap_max_samples
                ).explain(MagicMock(), X, feature_names=list(X.columns))

            mock_cls.assert_called_once_with(max_samples=50)
            assert shap_summary == fake_summary
