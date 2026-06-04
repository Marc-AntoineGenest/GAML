"""
Tests for genetic_automl/cli.py

Coverage:
  1.  gaml version — prints version string, exits 0
  2.  gaml fit — missing data file exits 1
  3.  gaml fit — missing --target exits 1
  4.  gaml fit — unknown target column exits 1
  5.  gaml fit — bad --problem value caught by argparse
  6.  gaml fit — minimal happy path (mocked pipeline)
  7.  gaml fit -- CLI flags override YAML config values
  8.  gaml fit -- --no-shap sets shap_enabled=False
  9.  gaml fit -- --save calls pipeline.save()
  10. gaml fit -- --config loads yaml and merges with CLI
  11. gaml fit -- --generations / --population / --seed applied
  12. gaml predict -- missing model file exits 1
  13. gaml predict -- missing data file exits 1
  14. gaml predict -- happy path writes predictions CSV
  15. gaml predict -- target column dropped from input
  16. gaml predict -- --output flag respected
  17. main() returns int (not None)
  18. CLI installed as console_script entry point
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from genetic_automl.cli import main, _build_parser


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv(tmp_path) -> Path:
    """A small classification CSV with a 'label' target column."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.standard_normal((80, 4)),
        columns=["a", "b", "c", "d"],
    )
    df["label"] = rng.integers(0, 2, 80)
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def minimal_yaml(tmp_path) -> Path:
    """A minimal gaml_config.yaml for testing."""
    content = """\
run:
  name: cli_test
  backend: sklearn
problem:
  type: classification
  target_column: label
genetic:
  population_size: 4
  generations: 2
"""
    p = tmp_path / "gaml_config.yaml"
    p.write_text(content)
    return p


def _make_mock_pipeline(score: float = 0.85, metric: str = "f1_macro"):
    """Return a MagicMock that mimics a fitted AutoMLPipeline."""
    mock = MagicMock()
    mock.final_score = score
    mock._metric_name = metric
    mock.report_path = "/tmp/report.html"
    # Use a real object for config so attribute access behaves predictably
    mock.config = MagicMock()
    mock.config.target_column = "label"
    mock.predict.return_value = np.array([0, 1, 0, 1])
    mock.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7],
                                                 [0.9, 0.1], [0.4, 0.6]])
    mock.save.return_value = "/tmp/model.joblib"
    return mock


# ---------------------------------------------------------------------------
# 1. version
# ---------------------------------------------------------------------------

class TestVersion:
    def test_version_exit_0(self, capsys):
        rc = main(["version"])
        assert rc == 0

    def test_version_prints_gaml(self, capsys):
        main(["version"])
        captured = capsys.readouterr()
        assert "gaml" in captured.out


# ---------------------------------------------------------------------------
# 2-5. fit — error paths
# ---------------------------------------------------------------------------

class TestFitErrors:
    def test_missing_data_file_exits_1(self, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            main(["fit", str(tmp_path / "nonexistent.csv"), "--target", "label"])
        assert exc_info.value.code == 1

    def test_missing_target_exits_1(self, sample_csv):
        with pytest.raises(SystemExit) as exc_info:
            main(["fit", str(sample_csv)])
        assert exc_info.value.code == 1

    def test_unknown_target_column_exits_1(self, sample_csv):
        with pytest.raises(SystemExit) as exc_info:
            main(["fit", str(sample_csv), "--target", "nonexistent_col"])
        assert exc_info.value.code == 1

    def test_bad_problem_type_caught_by_argparse(self, sample_csv):
        """argparse should reject unknown --problem values."""
        with pytest.raises(SystemExit) as exc_info:
            main(["fit", str(sample_csv), "--target", "label", "--problem", "clustering"])
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# 6. fit — happy path (mocked pipeline)
# ---------------------------------------------------------------------------

class TestFitHappyPath:
    def test_fit_returns_0_and_prints_score(self, sample_csv, tmp_path, capsys):
        mock_pipeline = _make_mock_pipeline()

        with patch("genetic_automl.cli.AutoMLPipeline") as MockCls:
            MockCls.return_value = mock_pipeline
            rc = main([
                "fit", str(sample_csv),
                "--target", "label",
                "--output-dir", str(tmp_path / "reports"),
            ])

        assert rc == 0
        captured = capsys.readouterr()
        assert "f1_macro=0.850000" in captured.out

    def test_fit_calls_pipeline_fit(self, sample_csv, tmp_path):
        mock_pipeline = _make_mock_pipeline()

        with patch("genetic_automl.cli.AutoMLPipeline") as MockCls:
            MockCls.return_value = mock_pipeline
            main(["fit", str(sample_csv), "--target", "label",
                  "--output-dir", str(tmp_path)])

        mock_pipeline.fit.assert_called_once()
        call_args = mock_pipeline.fit.call_args[0][0]
        assert isinstance(call_args, pd.DataFrame)


# ---------------------------------------------------------------------------
# 7-11. fit — CLI flag overrides
# ---------------------------------------------------------------------------

class TestFitFlagOverrides:
    def _run_fit(self, sample_csv, tmp_path, extra_args=None):
        """Run fit with mocked pipeline, return the config passed to AutoMLPipeline."""
        mock_pipeline = _make_mock_pipeline()
        captured_config = {}

        def fake_init(config, gene_space_overrides=None):
            captured_config["config"] = config
            return mock_pipeline

        with patch("genetic_automl.cli.AutoMLPipeline", side_effect=fake_init):
            args = ["fit", str(sample_csv), "--target", "label",
                    "--output-dir", str(tmp_path)] + (extra_args or [])
            main(args)

        return captured_config["config"]

    def test_problem_regression_applied(self, sample_csv, tmp_path):
        from genetic_automl.core.problem import ProblemType
        cfg = self._run_fit(sample_csv, tmp_path, ["--problem", "regression"])
        assert cfg.problem_type == ProblemType.REGRESSION

    def test_backend_applied(self, sample_csv, tmp_path):
        cfg = self._run_fit(sample_csv, tmp_path, ["--backend", "sklearn"])
        assert cfg.automl.backend == "sklearn"

    def test_generations_applied(self, sample_csv, tmp_path):
        cfg = self._run_fit(sample_csv, tmp_path, ["--generations", "7"])
        assert cfg.genetic.generations == 7

    def test_population_applied(self, sample_csv, tmp_path):
        cfg = self._run_fit(sample_csv, tmp_path, ["--population", "12"])
        assert cfg.genetic.population_size == 12

    def test_seed_applied(self, sample_csv, tmp_path):
        cfg = self._run_fit(sample_csv, tmp_path, ["--seed", "99"])
        assert cfg.genetic.random_seed == 99

    def test_no_shap_disables_shap(self, sample_csv, tmp_path):
        cfg = self._run_fit(sample_csv, tmp_path, ["--no-shap"])
        assert cfg.report.shap_enabled is False

    def test_run_name_applied(self, sample_csv, tmp_path):
        cfg = self._run_fit(sample_csv, tmp_path, ["--run-name", "my_test_run"])
        assert cfg.run_name == "my_test_run"

    def test_output_dir_applied(self, sample_csv, tmp_path):
        out = str(tmp_path / "custom_reports")
        cfg = self._run_fit(sample_csv, tmp_path, ["--output-dir", out])
        assert cfg.report.output_dir == out

    def test_save_calls_pipeline_save(self, sample_csv, tmp_path):
        mock_pipeline = _make_mock_pipeline()
        save_path = str(tmp_path / "model.joblib")

        with patch("genetic_automl.cli.AutoMLPipeline") as MockCls:
            MockCls.return_value = mock_pipeline
            main(["fit", str(sample_csv), "--target", "label",
                  "--output-dir", str(tmp_path), "--save", save_path])

        mock_pipeline.save.assert_called_once_with(save_path)


# ---------------------------------------------------------------------------
# 10. fit -- YAML config loaded and merged
# ---------------------------------------------------------------------------

class TestFitYAMLMerge:
    def test_yaml_config_loaded(self, sample_csv, minimal_yaml, tmp_path):
        """--config should load the YAML; CLI --generations should win."""
        mock_pipeline = _make_mock_pipeline()
        captured = {}

        def fake_init(config, gene_space_overrides=None):
            captured["config"] = config
            return mock_pipeline

        with patch("genetic_automl.cli.AutoMLPipeline", side_effect=fake_init):
            main([
                "fit", str(sample_csv),
                "--target", "label",
                "--config", str(minimal_yaml),
                "--generations", "5",     # CLI override wins over YAML's 2
                "--output-dir", str(tmp_path),
            ])

        cfg = captured["config"]
        assert cfg.run_name == "cli_test"      # from YAML
        assert cfg.genetic.generations == 5    # CLI wins over YAML's 2


# ---------------------------------------------------------------------------
# 12-16. predict
# ---------------------------------------------------------------------------

class TestPredict:
    def _save_fake_model(self, tmp_path) -> Path:
        """Write a fake joblib that AutoMLPipeline.load can read."""
        import joblib
        mock_pipeline = _make_mock_pipeline()
        mock_pipeline.config.target_column = "label"
        p = tmp_path / "model.joblib"
        # patch load() to return our mock
        return p

    def test_missing_model_exits_1(self, sample_csv, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            main(["predict", str(tmp_path / "missing.joblib"), str(sample_csv)])
        assert exc_info.value.code == 1

    def test_missing_data_exits_1(self, tmp_path):
        fake_model = tmp_path / "model.joblib"
        fake_model.write_bytes(b"fake")
        with pytest.raises(SystemExit) as exc_info:
            main(["predict", str(fake_model), str(tmp_path / "missing.csv")])
        assert exc_info.value.code == 1

    def test_happy_path_writes_csv(self, sample_csv, tmp_path):
        mock_pipeline = _make_mock_pipeline()
        mock_pipeline.config.target_column = "label"
        mock_pipeline.predict.return_value = np.zeros(80, dtype=int)
        mock_pipeline.predict_proba.return_value = None

        fake_model = tmp_path / "model.joblib"
        fake_model.write_bytes(b"fake")
        out_path = tmp_path / "preds.csv"

        with patch("genetic_automl.cli.AutoMLPipeline") as MockCls:
            MockCls.load.return_value = mock_pipeline
            rc = main(["predict", str(fake_model), str(sample_csv),
                       "--output", str(out_path)])

        assert rc == 0
        assert out_path.exists()
        pred_df = pd.read_csv(out_path)
        assert "prediction" in pred_df.columns
        assert len(pred_df) == 80

    def test_target_column_dropped_before_predict(self, sample_csv, tmp_path):
        """The target column must not be fed to model.predict()."""
        mock_pipeline = _make_mock_pipeline()
        mock_pipeline.config.target_column = "label"
        mock_pipeline.predict.return_value = np.zeros(80, dtype=int)
        mock_pipeline.predict_proba.return_value = None

        fake_model = tmp_path / "model.joblib"
        fake_model.write_bytes(b"fake")

        with patch("genetic_automl.cli.AutoMLPipeline") as MockCls:
            MockCls.load.return_value = mock_pipeline
            main(["predict", str(fake_model), str(sample_csv),
                  "--output", str(tmp_path / "out.csv")])

        predict_call_df = mock_pipeline.predict.call_args[0][0]
        assert "label" not in predict_call_df.columns

    def test_output_flag_respected(self, sample_csv, tmp_path, capsys):
        mock_pipeline = _make_mock_pipeline()
        mock_pipeline.config.target_column = "label"
        mock_pipeline.predict.return_value = np.zeros(80, dtype=int)
        mock_pipeline.predict_proba.return_value = None

        fake_model = tmp_path / "model.joblib"
        fake_model.write_bytes(b"fake")
        custom_out = tmp_path / "custom_output.csv"

        with patch("genetic_automl.cli.AutoMLPipeline") as MockCls:
            MockCls.load.return_value = mock_pipeline
            main(["predict", str(fake_model), str(sample_csv),
                  "--output", str(custom_out)])

        captured = capsys.readouterr()
        assert str(custom_out) in captured.out

    def test_proba_columns_added(self, sample_csv, tmp_path):
        """When predict_proba returns values, proba_class_N columns should appear."""
        mock_pipeline = _make_mock_pipeline()
        mock_pipeline.config.target_column = "label"
        mock_pipeline.predict.return_value = np.zeros(80, dtype=int)
        mock_pipeline.predict_proba.return_value = np.column_stack([
            np.full(80, 0.7), np.full(80, 0.3)
        ])

        fake_model = tmp_path / "model.joblib"
        fake_model.write_bytes(b"fake")
        out_path = tmp_path / "out_proba.csv"

        with patch("genetic_automl.cli.AutoMLPipeline") as MockCls:
            MockCls.load.return_value = mock_pipeline
            main(["predict", str(fake_model), str(sample_csv),
                  "--output", str(out_path)])

        df = pd.read_csv(out_path)
        assert "proba_class_0" in df.columns
        assert "proba_class_1" in df.columns


# ---------------------------------------------------------------------------
# 17. main() return type
# ---------------------------------------------------------------------------

class TestMainReturnType:
    def test_main_returns_int(self, capsys):
        rc = main(["version"])
        assert isinstance(rc, int)


# ---------------------------------------------------------------------------
# 18. Entry point registered
# ---------------------------------------------------------------------------

class TestEntryPoint:
    def test_gaml_entry_point_registered(self):
        """The 'gaml' console_script should be registered after pip install -e ."""
        import subprocess
        result = subprocess.run(
            ["gaml", "version"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "gaml" in result.stdout

    def test_python_m_invocation(self):
        """python -m genetic_automl.cli version should also work."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "genetic_automl.cli", "version"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "gaml" in result.stdout
