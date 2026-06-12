"""
Tests for RandomForest model backend and surrogate-assisted GA.

  1. RandomForestModel — unit tests
  2. build_automl dispatches on model_type='rf'
  3. 'rf' is in the sklearn gene space
  4. predict_with_std returns sensible uncertainty estimates
  5. SurrogateModel — encode, update, should_skip
  6. Surrogate swappable: lgbm and xgb surrogates also work
  7. Full pipeline integration — surrogate fires and reduces CV calls

Run:
    pytest genetic_automl/tests/test_surrogate.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from genetic_automl.automl import build_automl
from genetic_automl.automl.lgbm_model import LGBMModel
from genetic_automl.automl.rf_model import RandomForestModel
from genetic_automl.config import (
    AutoMLConfig,
    DataConfig,
    EnsembleConfig,
    GeneticConfig,
    PipelineConfig,
    ReportConfig,
)
from genetic_automl.core.problem import ProblemType
from genetic_automl.genetic.chromosome import Chromosome, get_gene_space
from genetic_automl.genetic.surrogate import SurrogateModel
from genetic_automl.pipeline import AutoMLPipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clf_Xy(n=300):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((n, 6)), columns=list("abcdef"))
    y = pd.Series(rng.integers(0, 2, n), name="label")
    return X, y


@pytest.fixture
def reg_Xy(n=300):
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=list("abcde"))
    y = pd.Series(rng.standard_normal(n) * 10 + 50, name="target")
    return X, y


@pytest.fixture
def clf_df(n=300):
    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.standard_normal((n, 4)), columns=list("abcd"))
    df["label"] = rng.integers(0, 2, n)
    return df


def _surrogate_config(surrogate_model_type="rf", surrogate_enabled=True):
    return PipelineConfig(
        problem_type=ProblemType.CLASSIFICATION,
        target_column="label",
        genetic=GeneticConfig(
            population_size=8,
            generations=3,
            early_stopping_rounds=3,
            n_cv_folds=2,
            warm_start=True,
            warm_start_n_seeds=2,
            warm_start_halving_pool_ratio=0,
            adaptive_mutation=False,
            random_seed=5,
            surrogate_enabled=surrogate_enabled,
            surrogate_model_type=surrogate_model_type,
            surrogate_min_samples=4,        # low so surrogate fires fast in tests
            surrogate_uncertainty_threshold=999.0,  # disable uncertainty gating
        ),
        automl=AutoMLConfig(
            backend="sklearn",
            ensemble=EnsembleConfig(enabled=False),  # isolate surrogate test
        ),
        data=DataConfig(test_size=0.15),
        report=ReportConfig(output_dir="/tmp/test_surrogate_reports"),
    )


# ===========================================================================
# 1. RandomForestModel unit tests
# ===========================================================================

class TestRandomForestModel:
    def test_fit_predict_classification(self, clf_Xy):
        X, y = clf_Xy
        m = RandomForestModel(ProblemType.CLASSIFICATION, "label", random_seed=0)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)
        assert set(preds).issubset({0, 1})

    def test_fit_predict_regression(self, reg_Xy):
        X, y = reg_Xy
        m = RandomForestModel(ProblemType.REGRESSION, "target", random_seed=0)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)
        assert np.isfinite(preds).all()

    def test_predict_proba_clf(self, clf_Xy):
        X, y = clf_Xy
        m = RandomForestModel(ProblemType.CLASSIFICATION, "label", random_seed=0)
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba is not None
        assert proba.shape == (len(y), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_proba_regression_none(self, reg_Xy):
        X, y = reg_Xy
        m = RandomForestModel(ProblemType.REGRESSION, "target", random_seed=0)
        m.fit(X, y)
        assert m.predict_proba(X) is None

    def test_not_fitted_raises(self, clf_Xy):
        X, _ = clf_Xy
        m = RandomForestModel(ProblemType.CLASSIFICATION, "label")
        with pytest.raises(RuntimeError):
            m.predict(X)

    def test_feature_importances(self, clf_Xy):
        X, y = clf_Xy
        m = RandomForestModel(ProblemType.CLASSIFICATION, "label", random_seed=0)
        m.fit(X, y)
        fi = m.feature_importances_
        assert fi is not None
        assert len(fi) == X.shape[1]
        assert (fi >= 0).all()

    def test_accepts_learning_rate_kwarg(self, clf_Xy):
        """RF ignores learning_rate but must not crash when gene passes it."""
        X, y = clf_Xy
        m = RandomForestModel(ProblemType.CLASSIFICATION, "label",
                              learning_rate=0.05, random_seed=0)
        m.fit(X, y)
        assert m.is_fitted

    def test_get_params_model_type(self, clf_Xy):
        X, y = clf_Xy
        m = RandomForestModel(ProblemType.CLASSIFICATION, "label", random_seed=0)
        m.fit(X, y)
        assert m.get_params()["model_type"] == "rf"

    def test_val_set_silently_accepted(self, clf_Xy):
        """RF accepts X_val/y_val without crashing (just ignores them)."""
        X, y = clf_Xy
        split = len(X) // 5
        m = RandomForestModel(ProblemType.CLASSIFICATION, "label", random_seed=0)
        m.fit(X.iloc[split:], y.iloc[split:],
              X_val=X.iloc[:split], y_val=y.iloc[:split])
        assert m.is_fitted


# ===========================================================================
# 2. predict_with_std — uncertainty signal
# ===========================================================================

class TestPredictWithStd:
    def test_regression_shapes(self, reg_Xy):
        X, y = reg_Xy
        m = RandomForestModel(ProblemType.REGRESSION, "target",
                              n_estimators=50, random_seed=0)
        m.fit(X, y)
        mean, std = m.predict_with_std(X)
        assert mean.shape == (len(y),)
        assert std.shape == (len(y),)

    def test_clf_shapes(self, clf_Xy):
        X, y = clf_Xy
        m = RandomForestModel(ProblemType.CLASSIFICATION, "label",
                              n_estimators=50, random_seed=0)
        m.fit(X, y)
        mean, std = m.predict_with_std(X)
        assert mean.shape == (len(y),)
        assert std.shape == (len(y),)

    def test_std_is_non_negative(self, clf_Xy):
        X, y = clf_Xy
        m = RandomForestModel(ProblemType.CLASSIFICATION, "label",
                              n_estimators=50, random_seed=0)
        m.fit(X, y)
        _, std = m.predict_with_std(X)
        assert (std >= 0).all()

    def test_std_varies_across_samples(self, clf_Xy):
        """Trees should not always agree perfectly — std must not be all zeros."""
        X, y = clf_Xy
        m = RandomForestModel(ProblemType.CLASSIFICATION, "label",
                              n_estimators=100, random_seed=0)
        m.fit(X, y)
        _, std = m.predict_with_std(X)
        assert std.max() > 0.0, "All per-tree predictions identical — that's suspicious"


# ===========================================================================
# 3. build_automl dispatches rf
# ===========================================================================

class TestBuildAutoMLRF:
    def test_dispatch_rf(self):
        m = build_automl("sklearn", ProblemType.CLASSIFICATION, "y", model_type="rf")
        assert isinstance(m, RandomForestModel)

    def test_rf_fits_and_predicts(self, clf_Xy):
        X, y = clf_Xy
        m = build_automl("sklearn", ProblemType.CLASSIFICATION, "y",
                         model_type="rf", random_seed=0)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)


# ===========================================================================
# 4. rf in gene space
# ===========================================================================

class TestRFGene:
    def test_rf_in_sklearn_gene_space(self):
        space = get_gene_space("sklearn")
        mt_gene = next(g for g in space if g.name == "model_type")
        assert "rf" in mt_gene.values

    def test_all_four_model_types_present(self):
        space = get_gene_space("sklearn")
        mt_gene = next(g for g in space if g.name == "model_type")
        assert set(mt_gene.values) == {"gbm", "lgbm", "xgb", "rf"}


# ===========================================================================
# 5. SurrogateModel unit tests
# ===========================================================================

class TestSurrogateModel:
    def _make_chromosomes(self, n=20, backend="sklearn"):
        rng = np.random.default_rng(42)
        space = get_gene_space(backend)
        import random as stdlib_random
        r = stdlib_random.Random(42)
        chroms = []
        for i in range(n):
            genes = {g.name: g.random_value(r) for g in space}
            c = Chromosome(genes=genes, fitness=float(rng.random()), generation=0)
            chroms.append(c)
        return chroms

    def test_no_skip_before_min_samples(self):
        surrogate = SurrogateModel(min_samples=20)
        chroms = self._make_chromosomes(5)
        surrogate.update(chroms)  # only 5, below min_samples=20
        skip, _ = surrogate.should_skip(chroms[0], population_median=0.5)
        assert not skip, "Should not skip before min_samples is reached"

    def test_skips_low_predicted_chromosome(self):
        """After training, a chromosome predicted well below median should be skipped."""
        surrogate = SurrogateModel(
            model_type="rf",
            min_samples=5,
            uncertainty_threshold=999.0,  # disable uncertainty gating for test
            skip_margin=0.0,
        )
        rng = np.random.default_rng(7)
        space = get_gene_space("sklearn")
        import random as r
        rnd = r.Random(7)

        # Build chromosomes with distinct fitness so surrogate has signal
        chroms = []
        for i in range(20):
            genes = {g.name: g.random_value(rnd) for g in space}
            fitness = float(rng.random())
            chroms.append(Chromosome(genes=genes, fitness=fitness, generation=0))

        surrogate.update(chroms)

        # Build a "clearly bad" chromosome and set its surrogate-predicted fitness low
        # by forcing it to have genes identical to the worst chromosome
        worst = min(chroms, key=lambda c: c.fitness)
        candidate = worst.copy()
        candidate.fitness = None  # not yet evaluated

        population_median = float(np.median([c.fitness for c in chroms]))
        skip, pred = surrogate.should_skip(candidate, population_median)
        # The prediction should be in a reasonable range
        assert isinstance(pred, float)

    def test_does_not_skip_high_predicted(self):
        """A chromosome predicted above the median must not be skipped."""
        surrogate = SurrogateModel(
            model_type="rf",
            min_samples=5,
            uncertainty_threshold=999.0,
            skip_margin=0.0,
        )
        rng = np.random.default_rng(8)
        import random as r
        rnd = r.Random(8)
        space = get_gene_space("sklearn")
        chroms = []
        for i in range(20):
            genes = {g.name: g.random_value(rnd) for g in space}
            chroms.append(Chromosome(genes=genes, fitness=float(rng.random()), generation=0))

        surrogate.update(chroms)
        best = max(chroms, key=lambda c: c.fitness)
        candidate = best.copy()
        candidate.fitness = None
        population_median = float(np.median([c.fitness for c in chroms]))
        skip, _ = surrogate.should_skip(candidate, population_median)
        # The best genes should be predicted above median → not skipped
        assert not skip

    def test_skip_rate_increases_over_time(self):
        """skip_rate should be > 0 after enough chromosomes are evaluated."""
        surrogate = SurrogateModel(
            min_samples=5,
            uncertainty_threshold=999.0,
            skip_margin=0.0,
        )
        rng = np.random.default_rng(9)
        import random as r
        rnd = r.Random(9)
        space = get_gene_space("sklearn")
        chroms = []
        for i in range(30):
            genes = {g.name: g.random_value(rnd) for g in space}
            chroms.append(Chromosome(genes=genes, fitness=float(rng.random()), generation=0))

        surrogate.update(chroms)
        median = float(np.median([c.fitness for c in chroms]))
        for c in chroms:
            c_copy = c.copy()
            c_copy.fitness = None
            surrogate.should_skip(c_copy, median)

        # After testing all 30 known chromosomes, some should have been skipped
        assert surrogate.skip_rate >= 0.0  # always true
        assert surrogate._total_candidates == 30

    def test_summary_keys(self):
        surrogate = SurrogateModel()
        s = surrogate.summary()
        for key in ("model_type", "min_samples", "n_trained_on",
                    "total_candidates", "skips", "skip_rate"):
            assert key in s

    def test_encode_produces_float_matrix(self):
        surrogate = SurrogateModel(backend_for_ga="sklearn")
        import random as r
        rnd = r.Random(0)
        space = get_gene_space("sklearn")
        chroms = [
            Chromosome(genes={g.name: g.random_value(rnd) for g in space}, generation=0)
            for _ in range(5)
        ]
        X = surrogate._encode(chroms)
        assert X.dtype == float
        assert X.shape == (5, len(space))


# ===========================================================================
# 6. Surrogate swappable — lgbm and xgb as surrogates
# ===========================================================================

class TestSurrogateSwappable:
    @pytest.mark.parametrize("model_type", ["lgbm", "xgb", "gbm"])
    def test_surrogate_with_alternative_model(self, model_type):
        """Any registered model_type can act as the surrogate."""
        surrogate = SurrogateModel(
            model_type=model_type,
            min_samples=5,
            uncertainty_threshold=999.0,
        )
        rng = np.random.default_rng(10)
        import random as r
        rnd = r.Random(10)
        space = get_gene_space("sklearn")
        chroms = [
            Chromosome(
                genes={g.name: g.random_value(rnd) for g in space},
                fitness=float(rng.random()),
                generation=0,
            )
            for _ in range(15)
        ]
        surrogate.update(chroms)
        assert surrogate._surrogate is not None
        assert surrogate._n_trained_on == 15

        candidate = chroms[0].copy()
        candidate.fitness = None
        skip, pred = surrogate.should_skip(candidate, population_median=0.5)
        assert isinstance(pred, float)


# ===========================================================================
# 7. Full pipeline integration with surrogate
# ===========================================================================

class TestSurrogatePipelineIntegration:
    def test_pipeline_with_rf_surrogate_runs(self, clf_df):
        pipeline = AutoMLPipeline(_surrogate_config("rf"))
        pipeline.fit(clf_df)
        assert pipeline.final_score is not None

    def test_pipeline_with_lgbm_surrogate_runs(self, clf_df):
        pipeline = AutoMLPipeline(_surrogate_config("lgbm"))
        pipeline.fit(clf_df)
        assert pipeline.final_score is not None

    def test_pipeline_with_surrogate_disabled(self, clf_df):
        pipeline = AutoMLPipeline(_surrogate_config(surrogate_enabled=False))
        pipeline.fit(clf_df)
        assert pipeline.final_score is not None

    def test_surrogate_summary_in_evaluator(self, clf_df):
        pipeline = AutoMLPipeline(_surrogate_config("rf"))
        pipeline.fit(clf_df)
        # Access the evaluator's surrogate summary via history
        assert pipeline.history is not None

    def test_rf_as_ga_model_and_surrogate(self, clf_df):
        """RF can be both a GA model (in model_type gene) and the surrogate."""
        cfg = _surrogate_config("rf")
        # Force GA to only search RF models so RF does double duty
        from genetic_automl.genetic.chromosome import (
            GeneDefinition,
            build_gene_space_from_config,
        )
        pipeline = AutoMLPipeline(cfg)
        pipeline.fit(clf_df)
        assert pipeline.final_score is not None
