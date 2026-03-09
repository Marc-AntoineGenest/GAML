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

@pytest.mark.xfail(reason="BUG B3: roc_auc needs predict_proba, not predict labels")
def test_b3_roc_auc_multiclass_does_not_crash():
    """roc_auc_score with multi_class='ovr' requires probability scores, not integer labels."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])  # hard integer labels from predict()
    fn, _ = _METRIC_REGISTRY["roc_auc"]
    # This should NOT raise — but currently does:
    # ValueError: axis 1 is out of bounds for array of dimension 1
    result = fn(y_true, y_pred)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# BUG B6 — IsolationForest clip leaks val/test median at transform time
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="BUG B6: IsolationForest clip uses val median, not train median")
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

    # Currently: clip fills with X_val["a"].median() = 300.0
    # Correct: should fill with a stored training statistic (e.g. train median ~0)
    # After fix, the clipped value should be close to 0, not 300
    assert X_val_out["a"].max() < 10.0, "Should use training median, not val median"


# ---------------------------------------------------------------------------
# BUG B7 — FeatureSelector silently returns empty DataFrame
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="BUG B7: FeatureSelector silently returns 0-col DataFrame on mismatch")
def test_b7_feature_selector_warns_on_column_mismatch():
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

    with pytest.raises((ValueError, KeyError)):
        # Should raise — currently returns empty (10, 0) DataFrame silently
        X_out = fs.transform(X_val)
        assert X_out.shape[1] > 0, "Should not silently return empty DataFrame"


# ---------------------------------------------------------------------------
# BUG B8 — Ordinal encoder maps unseen categories to -1
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="BUG B8: ordinal encodes unseen as -1, should be 0 or neutral")
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
    # After fix: unseen category should NOT produce a negative index
    assert X_out["cat"].iloc[0] >= 0, f"Got {X_out['cat'].iloc[0]}, should be >= 0"


# ---------------------------------------------------------------------------
# BUG B5 — Warm-start halving silently fails with n_folds=1
# ---------------------------------------------------------------------------

def test_b5_halving_n_folds_1_crashes():
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

    # Confirms the bug: every chromosome evaluated with n_folds=1 gets -inf
    assert fitness == float("-inf"), (
        "n_folds=1 should crash StratifiedKFold and return -inf. "
        "If this fails, the bug is fixed — remove this test."
    )
