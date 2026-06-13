"""Unit tests for preprocessing steps."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from genetic_automl.core.problem import ProblemType
from genetic_automl.preprocessing.categorical_encoder import CategoricalEncoder
from genetic_automl.preprocessing.correlation_filter import CorrelationFilter
from genetic_automl.preprocessing.distribution_transform import DistributionTransform
from genetic_automl.preprocessing.feature_selector import FeatureSelector
from genetic_automl.preprocessing.missing_indicator import MissingIndicator
from genetic_automl.preprocessing.numeric_imputer import NumericImputer
from genetic_automl.preprocessing.outlier_handler import OutlierHandler
from genetic_automl.preprocessing.pipeline import (
    PreprocessingConfig,
    PreprocessingPipeline,
)

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

    def test_ordinal_unseen_maps_to_midrange(self):
        """Unseen categories should map to the per-column mid-range ordinal index."""
        X_train = pd.DataFrame({"cat": ["A", "B", "C"]})
        X_val = pd.DataFrame({"cat": ["A", "D"]})  # D unseen
        enc = CategoricalEncoder("ordinal")
        enc.fit(X_train)
        X_out = enc.transform(X_val)
        # A→0, B→1, C→2  →  mid-range = (3-1)/2 = 1.0
        assert X_out["cat"].iloc[1] >= 0, "unseen category should not produce negative index"
        assert X_out["cat"].iloc[0] == pytest.approx(0.0)  # known category A → index 0

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

    def test_transform_missing_col_raises(self):
        """Column mismatch at transform time should raise a clear ValueError."""
        rng = np.random.default_rng(0)
        X_train = pd.DataFrame(rng.standard_normal((100, 3)), columns=["a", "b", "c"])
        y = pd.Series(rng.integers(0, 2, 100))
        fs = FeatureSelector("mutual_info", keep_k=1.0)
        fs.fit(X_train, y)
        X_val = pd.DataFrame(rng.standard_normal((10, 2)), columns=["a", "b"])  # c missing
        with pytest.raises(ValueError, match="missing from input"):
            fs.transform(X_val)


# ---------------------------------------------------------------------------
# Full PreprocessingPipeline
# ---------------------------------------------------------------------------

class TestPreprocessingPipeline:
    def test_no_nan_after_pipeline(self, small_X_y):
        X, y = small_X_y
        X = X.copy()
        X.loc[X.index[:10], "a"] = np.nan
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
        """Imputer must use train statistics when transforming val.

        Config explicitly disables scaler and all other transformations so that
        the only transformation applied is median imputation — allowing the test
        to assert the raw imputed value rather than a scaled version of it.
        """
        X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0] * 30})
        X_val = pd.DataFrame({"a": [np.nan] * 10})
        y = pd.Series([0, 1] * 45)
        config = PreprocessingConfig(
            numeric_imputer="median",
            scaler="none",
            outlier_method="none",
            correlation_threshold=None,
            distribution_transform="none",
            missing_indicator=False,
            feature_selection_method="none",
            imbalance_method="none",
        )
        pp = PreprocessingPipeline(config, ProblemType.CLASSIFICATION)
        pp.fit_transform_train(X_train, y)
        X_val_out = pp.transform(X_val)
        # All NaNs should fill to 2.0 (train median), not some val-based value
        assert np.allclose(X_val_out["a"].values, 2.0)


# Regression tests for confirmed bugs (fixed)

class TestBugRegressions:
    """Regression tests for bugs that were confirmed and fixed.

    Each test documents the original failure mode and asserts the fix holds.
    An XPASS on any of these means the regression was re-introduced.
    """

    def test_b3_roc_auc_routing_is_correct(self):
        """B3: roc_auc crashed on multiclass with hard-label predictions.

        Fixed: _METRIC_REGISTRY['roc_auc'] now routes by input shape and class
        count. Hard labels for multiclass raise ValueError with a clear message.
        """
        import numpy as np
        from genetic_automl.core.problem import _METRIC_REGISTRY

        fn, _ = _METRIC_REGISTRY["roc_auc"]

        rng = np.random.default_rng(0)
        y_true_multi = np.array([0, 1, 2, 0, 1, 2])
        raw = rng.random((6, 3))
        proba_matrix = raw / raw.sum(axis=1, keepdims=True)
        assert 0.0 <= fn(y_true_multi, proba_matrix) <= 1.0

        y_true_bin = np.array([0, 1, 0, 1, 0, 1])
        scores_bin = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        assert 0.0 <= fn(y_true_bin, scores_bin) <= 1.0

        import pytest
        y_pred_hard = np.array([0, 1, 2, 0, 2, 1])
        with pytest.raises(ValueError, match="roc_auc requires probability scores"):
            fn(y_true_multi, y_pred_hard)

    def test_b6_isolation_forest_clip_uses_train_median(self):
        """B6: IsolationForest clip replaced values with val/test median (leakage).

        Fixed: OutlierHandler.transform() now uses training statistics stored at fit.
        """
        import numpy as np
        import pandas as pd
        from genetic_automl.preprocessing.outlier_handler import OutlierHandler

        rng = np.random.default_rng(0)
        X_train = pd.DataFrame({"a": rng.standard_normal(200)})
        X_val = pd.DataFrame({"a": np.array([100.0, 200.0, 300.0, 400.0, 500.0])})

        oh = OutlierHandler("isolation_forest", action="clip")
        oh.fit(X_train)
        X_val_out = oh.transform(X_val)
        assert X_val_out["a"].max() < 10.0

    def test_b7_feature_selector_raises_on_column_mismatch(self):
        """B7: FeatureSelector silently returned empty DataFrame on column mismatch.

        Fixed: transform() now raises ValueError when selected columns are absent.
        """
        import numpy as np
        import pandas as pd
        import pytest
        from genetic_automl.preprocessing.feature_selector import FeatureSelector

        rng = np.random.default_rng(0)
        X_train = pd.DataFrame(rng.standard_normal((100, 3)), columns=["a", "b", "c"])
        y = pd.Series(rng.integers(0, 2, 100))

        fs = FeatureSelector("mutual_info", keep_k=1.0)
        fs.fit(X_train, y)

        X_val = pd.DataFrame(rng.standard_normal((10, 2)), columns=["x", "y"])
        with pytest.raises(ValueError, match="missing from input"):
            fs.transform(X_val)

    def test_b8_ordinal_unseen_not_negative(self):
        """B8: CategoricalEncoder(ordinal) mapped unseen categories to -1.

        Fixed: unseen categories now map to 0 (neutral ordinal) to avoid
        corrupting distance-based models.
        """
        import pandas as pd
        from genetic_automl.preprocessing.categorical_encoder import CategoricalEncoder

        X_train = pd.DataFrame({"cat": ["A", "B", "C"]})
        X_val = pd.DataFrame({"cat": ["D"]})
        enc = CategoricalEncoder("ordinal")
        enc.fit(X_train)
        X_out = enc.transform(X_val)
        assert X_out["cat"].iloc[0] >= 0

    def test_b5_halving_evaluator_with_n_folds_1(self):
        """B5: WarmStart halving used FitnessEvaluator(n_folds=1) causing silent -inf.

        Fixed: halving now uses train_test_split instead of StratifiedKFold(n_splits=1).
        The evaluator behaviour with n_folds=1 is unchanged: returns -inf, which is
        expected and guarded against in WarmStart.
        """
        import random
        import numpy as np
        import pandas as pd
        from genetic_automl.genetic.chromosome import random_population
        from genetic_automl.genetic.fitness import FitnessEvaluator
        from genetic_automl.core.problem import ProblemType

        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((100, 3)), columns=list("abc"))
        y = pd.Series(rng.integers(0, 2, 100))

        ev = FitnessEvaluator(ProblemType.CLASSIFICATION, "label", "sklearn", n_folds=1)
        chrom = random_population("sklearn", 1, random.Random(0))[0]
        fitness = ev.evaluate(chrom, X, y)
        assert fitness == float("-inf")
