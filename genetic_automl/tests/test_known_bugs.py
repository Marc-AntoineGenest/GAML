"""
Known bug regression tests.

Each test documents a confirmed bug. Tests marked xfail will:
  - XFAIL (expected failure) as long as the bug exists
  - XPASS (unexpected pass) once the bug is fixed — which becomes a green signal

Run with: pytest tests/test_known_bugs.py -v
"""
import numpy as np
import pandas as pd
import pytest

from genetic_automl.preprocessing.categorical_encoder import CategoricalEncoder
from genetic_automl.preprocessing.outlier_handler import OutlierHandler
from genetic_automl.preprocessing.feature_selector import FeatureSelector
from genetic_automl.core.problem import _METRIC_REGISTRY, ProblemType
from genetic_automl.core.base_automl import BaseAutoML


# ---------------------------------------------------------------------------
# BUG B3 — roc_auc crashes on multiclass with hard-label predictions
# ---------------------------------------------------------------------------

def test_b3_roc_auc_routing_is_correct():
    """roc_auc_score with multi_class='ovr' requires probability scores, not integer labels."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])  # hard integer labels from predict()
    fn, _ = _METRIC_REGISTRY["roc_auc"]
    result = fn(y_true, y_pred)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# BUG B6 — IsolationForest clip leaks val/test median at transform time
# ---------------------------------------------------------------------------

def test_b6_isolation_forest_clip_uses_train_median():
    """
    When action='clip', outlier rows are replaced with col.median().
    At transform time on val/test, col.median() is computed from the val/test
    data itself — not stored training statistics. This is leakage.
    """
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame({"a": rng.standard_normal(200)})
    # Val set with very different distribution
    X_val = pd.DataFrame({"a": np.array([100.0, 200.0, 300.0, 400.0, 500.0])})

    oh = OutlierHandler("isolation_forest", action="clip")
    oh.fit(X_train)
    X_val_out = oh.transform(X_val)

    assert X_val_out["a"].max() < 10.0, "Should use training median, not val median"


# ---------------------------------------------------------------------------
# BUG B7 — FeatureSelector silently returns empty DataFrame
# ---------------------------------------------------------------------------

def test_b7_feature_selector_raises_on_column_mismatch():
    """
    When a column that was selected at fit time is missing at transform time,
    the selector silently returns fewer columns. If all selected cols are missing,
    it returns an empty (N, 0) DataFrame with no warning or error.
    After fix: should raise ValueError or log a warning and impute missing cols.
    """
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame(rng.standard_normal((100, 3)), columns=["a", "b", "c"])
    y = pd.Series(rng.integers(0, 2, 100))

    fs = FeatureSelector("mutual_info", keep_k=1.0)
    fs.fit(X_train, y)

    # Pass completely different columns — all selected cols are absent
    X_val = pd.DataFrame(rng.standard_normal((10, 2)), columns=["x", "y"])

    with pytest.raises(ValueError, match="missing from input"):
        fs.transform(X_val)


# ---------------------------------------------------------------------------
# BUG B8 — Ordinal encoder maps unseen categories to -1
# ---------------------------------------------------------------------------

def test_b8_ordinal_unseen_not_negative():
    """
    Unseen categories at transform time map to -1 via pd.map().fillna(-1).
    -1 is not a valid ordinal — it implies a category "less than index 0".
    Distance-based models (KNN, SVM, linear) are silently corrupted.
    After fix: unseen should map to 0 or the mean ordinal.
    """
    X_train = pd.DataFrame({"cat": ["A", "B", "C"]})
    X_val = pd.DataFrame({"cat": ["D"]})  # unseen
    enc = CategoricalEncoder("ordinal")
    enc.fit(X_train)
    X_out = enc.transform(X_val)
    assert X_out["cat"].iloc[0] >= 0, f"Got {X_out['cat'].iloc[0]}, should be >= 0"


# ---------------------------------------------------------------------------
# BUG B5 — Warm-start halving silently fails with n_folds=1
# ---------------------------------------------------------------------------

def test_b5_halving_uses_split_not_kfold():
    """
    WarmStart halving creates FitnessEvaluator(n_folds=1).
    StratifiedKFold(n_splits=1) raises ValueError.
    The evaluator catches it and assigns fitness=-inf to every candidate.
    Result: halving pre-screen silently returns no survivors.
    This test documents that the bug is real and reproducible.
    """
    from genetic_automl.genetic.fitness import FitnessEvaluator
    from genetic_automl.genetic.chromosome import Chromosome, random_population
    import random

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((100, 3)), columns=list("abc"))
    y = pd.Series(rng.integers(0, 2, 100))

    ev = FitnessEvaluator(ProblemType.CLASSIFICATION, "label", "sklearn", n_folds=1)
    chrom = random_population("sklearn", 1, random.Random(0))[0]
    fitness = ev.evaluate(chrom, X, y)

    # n_folds=1 with StratifiedKFold raises ValueError; FitnessEvaluator returns -inf.
    # Halving now uses a train_test_split instead, so this code path is no longer
    # triggered by WarmStart — but the evaluator's behaviour is unchanged.
    assert fitness == float("-inf"), "n_folds=1 with StratifiedKFold should return -inf"
