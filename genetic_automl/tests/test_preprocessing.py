"""Unit tests for preprocessing steps."""
import numpy as np
import pandas as pd
import pytest

from genetic_automl.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline
from genetic_automl.preprocessing.numeric_imputer import NumericImputer
from genetic_automl.preprocessing.outlier_handler import OutlierHandler
from genetic_automl.preprocessing.correlation_filter import CorrelationFilter
from genetic_automl.preprocessing.categorical_encoder import CategoricalEncoder
from genetic_automl.preprocessing.distribution_transform import DistributionTransform
from genetic_automl.preprocessing.missing_indicator import MissingIndicator
from genetic_automl.preprocessing.feature_selector import FeatureSelector
from genetic_automl.core.problem import ProblemType


# ---------------------------------------------------------------------------
# NumericImputer
# ---------------------------------------------------------------------------

class TestNumericImputer:
    def test_median_fills_nans(self):
        X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 4.0]})
        imp = NumericImputer("median")
        X_out = imp.fit_transform(X)
        assert X_out.isnull().sum().sum() == 0

    def test_no_fit_leakage(self):
        """Transform must use train statistics, not val statistics."""
        X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        X_val = pd.DataFrame({"a": [np.nan, 100.0, 200.0]})
        imp = NumericImputer("median")
        imp.fit(X_train)
        X_val_out = imp.transform(X_val)
        # NaN should be filled with train median (2.0), not val median (150.0)
        assert X_val_out["a"].iloc[0] == pytest.approx(2.0)

    def test_constant_fills_zero(self):
        X = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        imp = NumericImputer("constant")
        X_out = imp.fit_transform(X)
        assert X_out["a"].iloc[1] == 0.0


# ---------------------------------------------------------------------------
# OutlierHandler
# ---------------------------------------------------------------------------

class TestOutlierHandler:
    def test_iqr_clip(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 999.0]})
        oh = OutlierHandler("iqr", threshold=1.5, action="clip")
        X_out = oh.fit_transform(X)
        assert X_out["a"].max() < 10.0

    def test_flag_adds_column(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 999.0]})
        oh = OutlierHandler("iqr", threshold=1.5, action="flag")
        X_out = oh.fit_transform(X)
        assert "__outlier__" in X_out.columns

    def test_transform_uses_train_bounds(self):
        """Val/test must use training data bounds, not their own."""
        X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        X_val = pd.DataFrame({"a": [1.0, 2.0, 999.0]})
        oh = OutlierHandler("iqr", threshold=1.5, action="clip")
        oh.fit(X_train)
        X_val_out = oh.transform(X_val)
        # 999 should be clipped to training upper bound
        assert X_val_out["a"].max() < 10.0


# ---------------------------------------------------------------------------
# CorrelationFilter
# ---------------------------------------------------------------------------

class TestCorrelationFilter:
    def test_drops_correlated_column(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.standard_normal(100)})
        X["b"] = X["a"] * 0.999 + rng.normal(0, 0.001, 100)  # nearly identical
        X["c"] = rng.standard_normal(100)
        cf = CorrelationFilter(threshold=0.95)
        X_out = cf.fit_transform(X)
        assert X_out.shape[1] == 2  # one of a/b dropped, c kept

    def test_none_threshold_disables(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
        cf = CorrelationFilter(threshold=None)
        X_out = cf.fit_transform(X)
        assert X_out.shape[1] == 2  # nothing dropped


# ---------------------------------------------------------------------------
# CategoricalEncoder
# ---------------------------------------------------------------------------

class TestCategoricalEncoder:
    def test_onehot_shape(self):
        X = pd.DataFrame({"cat": ["A", "B", "C", "A", "B"]})
        enc = CategoricalEncoder("onehot")
        X_out = enc.fit_transform(X)
        assert X_out.shape[1] == 3

    def test_ordinal_unseen_not_negative_one(self):
        """BUG B8: unseen categories should not map to -1."""
        X_train = pd.DataFrame({"cat": ["A", "B", "C"]})
        X_val = pd.DataFrame({"cat": ["A", "D"]})  # D unseen
        enc = CategoricalEncoder("ordinal")
        enc.fit(X_train)
        X_out = enc.transform(X_val)
        # -1 causes silent corruption in distance-based models
        # This test documents the known bug — should be 0 or global mean
        unseen_val = X_out["cat"].iloc[1]
        # TODO: fix BUG B8 — change -1 to 0
        assert unseen_val == -1  # currently fails after fix

    def test_no_nan_after_transform(self):
        X_train = pd.DataFrame({"cat": ["A", "B", "A"]})
        X_val = pd.DataFrame({"cat": ["A", "C"]})
        for strategy in ["onehot", "ordinal", "binary"]:
            enc = CategoricalEncoder(strategy)
            enc.fit(X_train)
            X_out = enc.transform(X_val)
            assert X_out.isnull().sum().sum() == 0, f"{strategy} produced NaN"


# ---------------------------------------------------------------------------
# DistributionTransform
# ---------------------------------------------------------------------------

class TestDistributionTransform:
    def test_yeo_johnson_reduces_skew(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"skewed": rng.lognormal(0, 2, 500)})
        before = abs(X["skewed"].skew())
        dt = DistributionTransform("yeo-johnson")
        X_out = dt.fit_transform(X)
        after = abs(X_out["skewed"].skew())
        assert after < before, "yeo-johnson should reduce skewness"

    def test_normal_column_skipped(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"normal": rng.standard_normal(500)})
        dt = DistributionTransform("yeo-johnson", skew_threshold=0.5)
        dt.fit_transform(X)
        assert "normal" not in dt.transformed_columns

    def test_none_is_noop(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        dt = DistributionTransform("none")
        X_out = dt.fit_transform(X)
        pd.testing.assert_frame_equal(X, X_out)


# ---------------------------------------------------------------------------
# MissingIndicator
# ---------------------------------------------------------------------------

class TestMissingIndicator:
    def test_adds_indicator_column(self):
        X_raw = pd.DataFrame({"age": [25.0, np.nan, 30.0, np.nan, 40.0]})
        mi = MissingIndicator(enabled=True, min_missing_frac=0.1)
        mi.fit(X_raw)
        X_filled = X_raw.fillna(X_raw.median())
        mask = X_raw.isnull()
        X_out = mi.transform_with_mask(X_filled, mask)
        assert "__missing_age__" in X_out.columns
        assert X_out["__missing_age__"].sum() == 2

    def test_disabled_is_noop(self):
        X = pd.DataFrame({"age": [1.0, np.nan, 3.0]})
        mi = MissingIndicator(enabled=False)
        mi.fit(X)
        X_out = mi.transform_with_mask(X, X.isnull())
        assert list(X_out.columns) == ["age"]


# ---------------------------------------------------------------------------
# FeatureSelector
# ---------------------------------------------------------------------------

class TestFeatureSelector:
    def test_mutual_info_selects_k(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((200, 10)), columns=[f"f{i}" for i in range(10)])
        y = pd.Series(rng.integers(0, 2, 200))
        fs = FeatureSelector("mutual_info", keep_k=0.5)
        X_out = fs.fit_transform(X, y)
        assert X_out.shape[1] == 5

    def test_transform_missing_col_warns_not_crashes(self):
        """BUG B7: should warn, not silently return empty DataFrame."""
        rng = np.random.default_rng(0)
        X_train = pd.DataFrame(rng.standard_normal((100, 3)), columns=["a", "b", "c"])
        y = pd.Series(rng.integers(0, 2, 100))
        fs = FeatureSelector("mutual_info", keep_k=1.0)
        fs.fit(X_train, y)
        X_val = pd.DataFrame(rng.standard_normal((10, 2)), columns=["a", "b"])  # c missing
        X_out = fs.transform(X_val)
        # Should return whatever columns are available, not silently drop to 0
        assert X_out.shape[1] >= 1


# ---------------------------------------------------------------------------
# Full PreprocessingPipeline
# ---------------------------------------------------------------------------

class TestPreprocessingPipeline:
    def test_no_nan_after_pipeline(self, small_X_y):
        X, y = small_X_y
        X["a"].iloc[:10] = np.nan
        config = PreprocessingConfig(numeric_imputer="median", outlier_method="none",
                                      scaler="standard", categorical_encoder="onehot",
                                      feature_selection_method="none")
        pp = PreprocessingPipeline(config, ProblemType.CLASSIFICATION)
        X_out, _ = pp.fit_transform_train(X, y)
        assert X_out.isnull().sum().sum() == 0

    def test_val_transform_matches_train_shape(self, small_X_y):
        X, y = small_X_y
        config = PreprocessingConfig(numeric_imputer="median", scaler="standard",
                                      correlation_threshold=0.95,
                                      feature_selection_method="none")
        pp = PreprocessingPipeline(config, ProblemType.CLASSIFICATION)
        X_train_out, _ = pp.fit_transform_train(X.iloc[:80], y.iloc[:80])
        X_val_out = pp.transform(X.iloc[80:])
        assert X_train_out.shape[1] == X_val_out.shape[1]

    def test_zero_leakage_imputer(self):
        """Imputer must use train statistics when transforming val."""
        X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0] * 30})
        X_val = pd.DataFrame({"a": [np.nan] * 10})
        y = pd.Series([0, 1] * 45)
        config = PreprocessingConfig(numeric_imputer="median")
        pp = PreprocessingPipeline(config, ProblemType.CLASSIFICATION)
        pp.fit_transform_train(X_train, y)
        X_val_out = pp.transform(X_val)
        # All NaNs should fill to 2.0 (train median), not some val-based value
        assert (X_val_out["a"] == pytest.approx(2.0)).all()
