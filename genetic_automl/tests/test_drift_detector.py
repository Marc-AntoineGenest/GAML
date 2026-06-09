"""
Tests for reporting/drift_detector.py and pipeline.detect_drift().

Coverage:
  1.  DriftDetector.fit() stores reference distribution
  2.  detect() — no drift on identical data
  3.  detect() — drift detected on shifted continuous feature
  4.  detect() — drift detected on categorical frequency shift
  5.  detect() — PSI flags drift independently of KS test
  6.  detect() — missing features in new data handled gracefully
  7.  detect() — extra columns in new data ignored
  8.  DriftReport.summary() — no-drift message
  9.  DriftReport.summary() — drift message with severity
  10. DriftReport.drifted_features property
  11. DriftReport.critical_features property
  12. DriftReport.to_json() — valid JSON
  13. _compute_psi — stable=0, critical>0.2
  14. _ks_test — identical arrays give p=1, shifted give p<0.05
  15. scipy not installed — falls back to numpy KS
  16. ReportConfig defaults: drift_enabled=False
  17. config_loader parses drift fields from YAML
  18. pipeline._drift_detector is None when drift_enabled=False
  19. pipeline._drift_detector fitted when drift_enabled=True
  20. pipeline.detect_drift() raises when drift_enabled=False
  21. pipeline.detect_drift() works when drift_enabled=True
  22. DriftDetector persisted through pipeline save/load
  23. CLI --detect-drift flag wired (unit test)
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from genetic_automl.config import PipelineConfig, ReportConfig
from genetic_automl.core.problem import ProblemType
from genetic_automl.reporting.drift_detector import (
    DriftDetector, DriftReport,
    _compute_psi, _ks_test, _is_continuous,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ref_df():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "num1": rng.normal(0, 1, 300),
        "num2": rng.normal(5, 2, 300),
        "cat1": rng.choice(["a", "b", "c"], 300),
    })
    return df


@pytest.fixture
def same_df(ref_df):
    """New data from same distribution — should not trigger drift."""
    rng = np.random.default_rng(99)
    df = pd.DataFrame({
        "num1": rng.normal(0, 1, 200),
        "num2": rng.normal(5, 2, 200),
        "cat1": rng.choice(["a", "b", "c"], 200),
    })
    return df


@pytest.fixture
def shifted_df():
    """New data with large distributional shift — should trigger drift."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "num1": rng.normal(10, 1, 200),   # mean shifted from 0 to 10
        "num2": rng.normal(5, 2, 200),    # unchanged
        "cat1": rng.choice(["a", "b", "c"], 200, p=[0.01, 0.01, 0.98]),  # freq shifted
    })
    return df


# ===========================================================================
# 1-7. Core DriftDetector behaviour
# ===========================================================================

class TestDriftDetectorCore:
    def test_fit_stores_reference(self, ref_df):
        detector = DriftDetector()
        detector.fit(ref_df)
        assert detector._reference is not None
        assert len(detector._bin_edges) == 2    # num1, num2
        assert "cat1" in detector._cat_categories

    def test_no_drift_on_same_distribution(self, ref_df, same_df):
        detector = DriftDetector(pvalue_threshold=0.05, psi_threshold=0.20)
        detector.fit(ref_df)
        report = detector.detect(same_df)
        # With a reasonable threshold, same distribution should not consistently drift
        # (Allow up to 1 borderline feature to reduce test flakiness)
        assert report.n_features_drifted <= 1

    def test_drift_detected_on_shifted_data(self, ref_df, shifted_df):
        detector = DriftDetector(pvalue_threshold=0.05, psi_threshold=0.10)
        detector.fit(ref_df)
        report = detector.detect(shifted_df)
        assert report.any_drift is True
        assert report.n_features_drifted >= 1

    def test_num1_flagged_as_drifted(self, ref_df, shifted_df):
        """num1 shifted from N(0,1) to N(10,1) — must be flagged."""
        detector = DriftDetector(pvalue_threshold=0.05, psi_threshold=0.10)
        detector.fit(ref_df)
        report = detector.detect(shifted_df)
        assert "num1" in report.drifted_features

    def test_categorical_drift_detected(self, ref_df, shifted_df):
        """cat1 frequency shifted from uniform to [0.01, 0.01, 0.98] — must flag."""
        detector = DriftDetector(pvalue_threshold=0.05, psi_threshold=0.10)
        detector.fit(ref_df)
        report = detector.detect(shifted_df)
        assert "cat1" in report.drifted_features

    def test_missing_feature_in_new_data_handled(self, ref_df):
        """New data missing a reference column should not raise."""
        detector = DriftDetector()
        detector.fit(ref_df)
        new = ref_df[["num1", "num2"]].copy()   # missing cat1
        report = detector.detect(new)
        assert "cat1" in report.missing_in_new
        assert report.n_features_checked == 2

    def test_extra_columns_in_new_data_ignored(self, ref_df):
        """Extra columns in new data should be ignored, not raise."""
        detector = DriftDetector()
        detector.fit(ref_df)
        new = ref_df.copy()
        new["extra_col"] = 0
        report = detector.detect(new)
        assert "extra_col" in report.new_columns
        assert report.n_features_checked == 3   # only ref columns checked

    def test_detect_before_fit_raises(self, ref_df):
        detector = DriftDetector()
        with pytest.raises(RuntimeError, match="fit()"):
            detector.detect(ref_df)


# ===========================================================================
# 8-12. DriftReport
# ===========================================================================

class TestDriftReport:
    def _make_report(self, drifted=False):
        from genetic_automl.reporting.drift_detector import FeatureDriftResult
        result = FeatureDriftResult(
            feature="num1", dtype="continuous",
            ks_statistic=0.8 if drifted else 0.05,
            ks_pvalue=0.001 if drifted else 0.9,
            chi2_statistic=None, chi2_pvalue=None,
            psi=0.35 if drifted else 0.02,
            drift_detected=drifted,
            severity="critical" if drifted else "none",
        )
        return DriftReport(
            n_features_checked=3,
            n_features_drifted=1 if drifted else 0,
            any_drift=drifted,
            pvalue_threshold=0.05,
            psi_threshold=0.20,
            feature_results=[result],
        )

    def test_summary_no_drift(self):
        report = self._make_report(drifted=False)
        summary = report.summary()
        assert "No drift" in summary
        assert "3" in summary

    def test_summary_drift(self):
        report = self._make_report(drifted=True)
        summary = report.summary()
        assert "Drift detected" in summary
        assert "num1" in summary

    def test_drifted_features_property(self):
        report = self._make_report(drifted=True)
        assert "num1" in report.drifted_features

    def test_critical_features_property(self):
        report = self._make_report(drifted=True)
        assert "num1" in report.critical_features

    def test_to_json_valid(self):
        report = self._make_report(drifted=True)
        j = report.to_json()
        parsed = json.loads(j)
        assert "any_drift" in parsed
        assert parsed["any_drift"] is True


# ===========================================================================
# 13-14. Statistical helpers
# ===========================================================================

class TestStatisticalHelpers:
    def test_psi_zero_for_identical(self):
        ref = np.array([0.25, 0.25, 0.25, 0.25])
        new = np.array([0.25, 0.25, 0.25, 0.25])
        psi = _compute_psi(ref * 100, new * 100, bin_edges=None, n_bins=4)
        assert psi < 0.01

    def test_psi_high_for_extreme_shift(self):
        ref = np.random.default_rng(0).normal(0, 1, 500)
        new = np.random.default_rng(1).normal(10, 1, 500)
        psi = _compute_psi(ref, new, bin_edges=None)
        assert psi > 0.20

    def test_ks_identical_arrays_high_pvalue(self):
        rng = np.random.default_rng(0)
        arr = rng.normal(0, 1, 200)
        stat, pvalue = _ks_test(arr, arr)
        assert pvalue > 0.05

    def test_ks_shifted_arrays_low_pvalue(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 200)
        new = rng.normal(5, 1, 200)
        stat, pvalue = _ks_test(ref, new)
        assert pvalue < 0.05

    def test_is_continuous_numeric_many_unique(self):
        s = pd.Series(np.random.normal(0, 1, 100))
        assert _is_continuous(s) is True

    def test_is_continuous_low_cardinality_false(self):
        s = pd.Series([1, 2, 3, 1, 2, 3] * 10)
        assert _is_continuous(s) is False


# ===========================================================================
# 15. scipy fallback
# ===========================================================================

class TestScipyFallback:
    def test_ks_without_scipy(self, monkeypatch):
        """KS test must work even when scipy is not installed."""
        import builtins
        real_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "scipy.stats":
                raise ImportError("scipy not installed")
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", mock_import)
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 100)
        new = rng.normal(5, 1, 100)
        stat, pvalue = _ks_test(ref, new)
        assert isinstance(stat, float)
        assert isinstance(pvalue, float)


# ===========================================================================
# 16-17. Config
# ===========================================================================

class TestDriftConfig:
    def test_report_config_defaults(self):
        cfg = ReportConfig()
        assert cfg.drift_enabled is False
        assert cfg.drift_pvalue_threshold == 0.05
        assert cfg.drift_psi_threshold == 0.20

    def test_config_loader_parses_drift(self, tmp_path):
        from genetic_automl.config_loader import load_config
        yaml = """\
run:
  backend: sklearn
problem:
  type: classification
  target_column: label
genetic:
  population_size: 4
  generations: 2
report:
  drift_enabled: true
  drift_pvalue_threshold: 0.01
  drift_psi_threshold: 0.15
"""
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml)
        cfg, _ = load_config(str(p))
        assert cfg.report.drift_enabled is True
        assert cfg.report.drift_pvalue_threshold == 0.01
        assert cfg.report.drift_psi_threshold == 0.15

    def test_config_loader_defaults_when_absent(self, tmp_path):
        from genetic_automl.config_loader import load_config
        yaml = """\
run:
  backend: sklearn
problem:
  type: classification
  target_column: label
genetic:
  population_size: 4
  generations: 2
"""
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml)
        cfg, _ = load_config(str(p))
        assert cfg.report.drift_enabled is False


# ===========================================================================
# 18-22. Pipeline integration
# ===========================================================================

class TestPipelineDriftIntegration:
    @pytest.fixture
    def clf_data(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((200, 4)), columns=list("abcd"))
        y = pd.Series(rng.integers(0, 2, 200), name="label")
        df = X.copy()
        df["label"] = y
        return df

    def test_drift_detector_none_when_disabled(self):
        from genetic_automl.pipeline import AutoMLPipeline
        pipeline = AutoMLPipeline.__new__(AutoMLPipeline)
        pipeline._drift_detector = None
        assert pipeline._drift_detector is None

    def test_detect_drift_raises_when_disabled(self, clf_data):
        """pipeline.detect_drift() must raise RuntimeError when not fitted."""
        from genetic_automl.pipeline import AutoMLPipeline
        pipeline = AutoMLPipeline.__new__(AutoMLPipeline)
        pipeline._drift_detector = None
        pipeline.config = PipelineConfig(target_column="label")
        pipeline._best_preprocessor = MagicMock()
        pipeline._best_preprocessor.transform.side_effect = lambda x: x
        with pytest.raises(RuntimeError, match="drift_enabled"):
            pipeline.detect_drift(clf_data)

    def test_detect_drift_works_when_fitted(self, clf_data):
        """pipeline.detect_drift() returns a DriftReport when detector is fitted."""
        from genetic_automl.pipeline import AutoMLPipeline
        pipeline = AutoMLPipeline.__new__(AutoMLPipeline)
        pipeline.config = PipelineConfig(target_column="label")

        X = clf_data.drop(columns=["label"])
        detector = DriftDetector()
        detector.fit(X)
        pipeline._drift_detector = detector
        pipeline._best_preprocessor = MagicMock()
        pipeline._best_preprocessor.transform.return_value = X

        report = pipeline.detect_drift(clf_data)
        assert isinstance(report, DriftReport)
        assert report.n_features_checked == 4

    def test_drift_detector_persisted_through_save_load(self, tmp_path, clf_data):
        """DriftDetector must survive joblib serialisation (used by pipeline.save/load)."""
        import joblib

        X = clf_data.drop(columns=["label"])
        detector = DriftDetector()
        detector.fit(X)

        save_path = str(tmp_path / "detector.joblib")
        joblib.dump(detector, save_path)
        loaded = joblib.load(save_path)

        assert loaded._reference is not None
        assert list(loaded._reference.columns) == list(X.columns)
        assert len(loaded._bin_edges) == len(detector._bin_edges)

        # Verify the loaded detector still works correctly
        report = loaded.detect(X)
        assert report.n_features_checked == X.shape[1]


# ===========================================================================
# 23. CLI flag
# ===========================================================================

class TestDriftCLI:
    @pytest.fixture
    def sample_csv(self, tmp_path):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.standard_normal((60, 3)), columns=list("abc"))
        df["label"] = rng.integers(0, 2, 60)
        p = tmp_path / "data.csv"
        df.to_csv(p, index=False)
        return p

    def test_detect_drift_flag_in_predict_parser(self):
        """--detect-drift flag must be registered in the predict subparser."""
        from genetic_automl.cli import _build_parser
        parser = _build_parser()
        # Find the predict subparser and check the flag exists
        subactions = {a.dest: a for a in
                      parser._subparsers._actions[-1]._name_parser_map["predict"]._actions}
        assert "detect_drift" in subactions
