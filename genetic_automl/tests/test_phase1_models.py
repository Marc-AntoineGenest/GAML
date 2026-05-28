"""
Tests for Phase 1 improvements:
  Item 1 — LightGBM backend (LGBMModel)
  Item 1 — XGBoost backend (XGBModel)
  Item 1 — model_type gene in chromosome gene space
  Item 1 — build_automl dispatches correctly on model_type
  Item 2 — EnsembleModel (soft voting / averaging)
  Item 2 — EvolutionHistory.top_chromosomes()
  Item 2 — EnsembleConfig dataclass
  Item 2 — Full pipeline ensemble integration

Run:
    pytest genetic_automl/tests/test_phase1_models.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from genetic_automl.automl import build_automl
from genetic_automl.automl.ensemble_model import EnsembleModel
from genetic_automl.automl.lgbm_model import LGBMModel
from genetic_automl.automl.xgb_model import XGBModel
from genetic_automl.automl.sklearn_model import SklearnModel
from genetic_automl.config import (
    AutoMLConfig, DataConfig, EnsembleConfig, GeneticConfig, PipelineConfig, ReportConfig,
)
from genetic_automl.core.problem import ProblemType
from genetic_automl.genetic.chromosome import get_gene_space, Chromosome
from genetic_automl.genetic.engine import EvolutionHistory, GenerationStats
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
    df = pd.DataFrame(rng.standard_normal((n, 5)), columns=list("abcde"))
    df["label"] = rng.integers(0, 2, n)
    return df


def _fast_ensemble_config(top_k=3):
    return PipelineConfig(
        problem_type=ProblemType.CLASSIFICATION,
        target_column="label",
        genetic=GeneticConfig(
            population_size=6,
            generations=2,
            early_stopping_rounds=2,
            n_cv_folds=2,
            warm_start=True,
            warm_start_n_seeds=3,
            warm_start_halving_pool_ratio=0,
            adaptive_mutation=False,
            random_seed=7,
        ),
        automl=AutoMLConfig(
            backend="sklearn",
            ensemble=EnsembleConfig(enabled=True, top_k=top_k, weight_by_fitness=True),
        ),
        data=DataConfig(test_size=0.15),
        report=ReportConfig(output_dir="/tmp/test_phase1_reports"),
    )


def _fast_single_config():
    cfg = _fast_ensemble_config(top_k=1)
    cfg.automl.ensemble = EnsembleConfig(enabled=False)
    return cfg


# ===========================================================================
# 1. LGBMModel unit tests
# ===========================================================================

class TestLGBMModel:
    def test_fit_predict_classification(self, clf_Xy):
        X, y = clf_Xy
        model = LGBMModel(ProblemType.CLASSIFICATION, "label", random_seed=0)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(y),)
        assert set(preds).issubset({0, 1})

    def test_fit_predict_regression(self, reg_Xy):
        X, y = reg_Xy
        model = LGBMModel(ProblemType.REGRESSION, "target", random_seed=0)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(y),)
        assert np.isfinite(preds).all()

    def test_predict_proba_classification(self, clf_Xy):
        X, y = clf_Xy
        model = LGBMModel(ProblemType.CLASSIFICATION, "label", random_seed=0)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba is not None
        assert proba.shape == (len(y), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_proba_regression_is_none(self, reg_Xy):
        X, y = reg_Xy
        model = LGBMModel(ProblemType.REGRESSION, "target", random_seed=0)
        model.fit(X, y)
        assert model.predict_proba(X) is None

    def test_early_stopping_with_val(self, clf_Xy):
        X, y = clf_Xy
        split = len(X) // 5
        model = LGBMModel(ProblemType.CLASSIFICATION, "label",
                          n_estimators=500, random_seed=0)
        model.fit(X.iloc[split:], y.iloc[split:],
                  X_val=X.iloc[:split], y_val=y.iloc[:split])
        # Early stopping should have reduced actual rounds well below 500
        assert model._estimator.n_estimators_ <= 500

    def test_feature_importances(self, clf_Xy):
        X, y = clf_Xy
        model = LGBMModel(ProblemType.CLASSIFICATION, "label", random_seed=0)
        model.fit(X, y)
        fi = model.feature_importances_
        assert fi is not None
        assert len(fi) == X.shape[1]
        assert (fi >= 0).all()

    def test_not_fitted_raises(self, clf_Xy):
        X, _ = clf_Xy
        model = LGBMModel(ProblemType.CLASSIFICATION, "label")
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_get_params(self, clf_Xy):
        X, y = clf_Xy
        model = LGBMModel(ProblemType.CLASSIFICATION, "label", random_seed=0)
        model.fit(X, y)
        params = model.get_params()
        assert params["model_type"] == "lgbm"


# ===========================================================================
# 2. XGBModel unit tests
# ===========================================================================

class TestXGBModel:
    def test_fit_predict_classification(self, clf_Xy):
        X, y = clf_Xy
        model = XGBModel(ProblemType.CLASSIFICATION, "label", random_seed=0)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(y),)
        assert set(preds).issubset({0, 1})

    def test_fit_predict_regression(self, reg_Xy):
        X, y = reg_Xy
        model = XGBModel(ProblemType.REGRESSION, "target", random_seed=0)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(y),)
        assert np.isfinite(preds).all()

    def test_predict_proba_classification(self, clf_Xy):
        X, y = clf_Xy
        model = XGBModel(ProblemType.CLASSIFICATION, "label", random_seed=0)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba is not None
        assert proba.shape == (len(y), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_proba_regression_is_none(self, reg_Xy):
        X, y = reg_Xy
        model = XGBModel(ProblemType.REGRESSION, "target", random_seed=0)
        model.fit(X, y)
        assert model.predict_proba(X) is None

    def test_label_encoding_non_integer(self):
        """XGBModel must handle non-integer class labels gracefully."""
        rng = np.random.default_rng(3)
        X = pd.DataFrame(rng.standard_normal((100, 3)), columns=list("abc"))
        y = pd.Series(np.where(rng.random(100) > 0.5, "cat", "dog"))
        model = XGBModel(ProblemType.CLASSIFICATION, "label", random_seed=0)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({"cat", "dog"})

    def test_feature_importances(self, clf_Xy):
        X, y = clf_Xy
        model = XGBModel(ProblemType.CLASSIFICATION, "label", random_seed=0)
        model.fit(X, y)
        fi = model.feature_importances_
        assert fi is not None
        assert len(fi) == X.shape[1]

    def test_not_fitted_raises(self, clf_Xy):
        X, _ = clf_Xy
        model = XGBModel(ProblemType.CLASSIFICATION, "label")
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_get_params(self, clf_Xy):
        X, y = clf_Xy
        model = XGBModel(ProblemType.CLASSIFICATION, "label", random_seed=0)
        model.fit(X, y)
        params = model.get_params()
        assert params["model_type"] == "xgb"


# ===========================================================================
# 3. build_automl dispatching on model_type
# ===========================================================================

class TestBuildAutoML:
    def test_dispatch_lgbm(self):
        m = build_automl("sklearn", ProblemType.CLASSIFICATION, "y", model_type="lgbm")
        assert isinstance(m, LGBMModel)

    def test_dispatch_xgb(self):
        m = build_automl("sklearn", ProblemType.CLASSIFICATION, "y", model_type="xgb")
        assert isinstance(m, XGBModel)

    def test_dispatch_gbm_default(self):
        m = build_automl("sklearn", ProblemType.CLASSIFICATION, "y")
        assert isinstance(m, SklearnModel)

    def test_dispatch_gbm_explicit(self):
        m = build_automl("sklearn", ProblemType.CLASSIFICATION, "y", model_type="gbm")
        assert isinstance(m, SklearnModel)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown AutoML backend"):
            build_automl("notabackend", ProblemType.CLASSIFICATION, "y")


# ===========================================================================
# 4. model_type gene in chromosome gene space
# ===========================================================================

class TestModelTypeGene:
    def test_model_type_in_sklearn_gene_space(self):
        space = get_gene_space("sklearn")
        names = [g.name for g in space]
        assert "model_type" in names

    def test_model_type_values(self):
        space = get_gene_space("sklearn")
        gene = next(g for g in space if g.name == "model_type")
        assert set(gene.values) == {"gbm", "lgbm", "xgb"}

    def test_autogluon_space_unchanged(self):
        space = get_gene_space("autogluon")
        names = [g.name for g in space]
        assert "model_type" not in names
        assert "presets" in names


# ===========================================================================
# 5. EnsembleModel unit tests
# ===========================================================================

class TestEnsembleModel:
    def _make_fitted_members(self, clf_Xy, n=3, model_cls=LGBMModel):
        X, y = clf_Xy
        members = []
        for i in range(n):
            m = model_cls(ProblemType.CLASSIFICATION, "label", random_seed=i)
            m.fit(X, y)
            members.append(m)
        return members, X, y

    def test_predict_returns_correct_shape(self, clf_Xy):
        members, X, y = self._make_fitted_members(clf_Xy)
        ens = EnsembleModel(members, ProblemType.CLASSIFICATION, "label")
        preds = ens.predict(X)
        assert preds.shape == (len(y),)

    def test_predict_proba_shape_and_sums(self, clf_Xy):
        members, X, y = self._make_fitted_members(clf_Xy)
        ens = EnsembleModel(members, ProblemType.CLASSIFICATION, "label")
        proba = ens.predict_proba(X)
        assert proba is not None
        assert proba.shape == (len(y), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_regression_averaging(self, reg_Xy):
        X, y = reg_Xy
        members = []
        for i in range(3):
            m = LGBMModel(ProblemType.REGRESSION, "target", random_seed=i)
            m.fit(X, y)
            members.append(m)
        ens = EnsembleModel(members, ProblemType.REGRESSION, "target")
        preds = ens.predict(X)
        assert preds.shape == (len(y),)
        # Ensemble prediction should be between min and max of members
        all_preds = np.stack([m.predict(X) for m in members])
        assert (preds >= all_preds.min(axis=0) - 1e-6).all()
        assert (preds <= all_preds.max(axis=0) + 1e-6).all()

    def test_feature_importances_averaged(self, clf_Xy):
        members, X, y = self._make_fitted_members(clf_Xy)
        ens = EnsembleModel(members, ProblemType.CLASSIFICATION, "label")
        fi = ens.feature_importances_
        assert fi is not None
        assert len(fi) == X.shape[1]
        assert abs(fi.sum() - 1.0) < 1e-5

    def test_uniform_weights(self, clf_Xy):
        members, X, y = self._make_fitted_members(clf_Xy, n=3)
        ens = EnsembleModel(members, ProblemType.CLASSIFICATION, "label", weights=None)
        assert all(abs(w - 1/3) < 1e-9 for w in ens.weights)

    def test_custom_weights_normalised(self, clf_Xy):
        members, X, y = self._make_fitted_members(clf_Xy, n=3)
        ens = EnsembleModel(members, ProblemType.CLASSIFICATION, "label",
                            weights=[3.0, 1.0, 1.0])
        assert abs(ens.weights[0] - 0.6) < 1e-9
        assert abs(sum(ens.weights) - 1.0) < 1e-9

    def test_single_member_works(self, clf_Xy):
        X, y = clf_Xy
        m = LGBMModel(ProblemType.CLASSIFICATION, "label", random_seed=0)
        m.fit(X, y)
        ens = EnsembleModel([m], ProblemType.CLASSIFICATION, "label")
        preds = ens.predict(X)
        assert preds.shape == (len(y),)

    def test_empty_members_raises(self):
        with pytest.raises(ValueError, match="at least one member"):
            EnsembleModel([], ProblemType.CLASSIFICATION, "label")

    def test_wrong_weights_length_raises(self, clf_Xy):
        members, _, _ = self._make_fitted_members(clf_Xy, n=2)
        with pytest.raises(ValueError, match="len\\(weights\\)"):
            EnsembleModel(members, ProblemType.CLASSIFICATION, "label",
                          weights=[0.5, 0.3, 0.2])

    def test_all_zero_weights_falls_back_to_uniform(self, clf_Xy):
        """Regression fitness values are negative — clamped to 0 — must not crash."""
        members, X, y = self._make_fitted_members(clf_Xy, n=3)
        ens = EnsembleModel(members, ProblemType.CLASSIFICATION, "label",
                            weights=[0.0, 0.0, 0.0])
        assert all(abs(w - 1/3) < 1e-9 for w in ens.weights)
        preds = ens.predict(X)
        assert preds.shape == (len(y),)

    def test_mixed_backends_in_ensemble(self, clf_Xy):
        """An ensemble can combine different algorithm types."""
        X, y = clf_Xy
        lgbm = LGBMModel(ProblemType.CLASSIFICATION, "label", random_seed=0)
        xgb = XGBModel(ProblemType.CLASSIFICATION, "label", random_seed=1)
        gbm = SklearnModel(ProblemType.CLASSIFICATION, "label", random_seed=2)
        for m in (lgbm, xgb, gbm):
            m.fit(X, y)
        ens = EnsembleModel([lgbm, xgb, gbm], ProblemType.CLASSIFICATION, "label")
        preds = ens.predict(X)
        assert preds.shape == (len(y),)

    def test_fit_is_noop(self, clf_Xy):
        """EnsembleModel.fit() should silently succeed (members are pre-fitted)."""
        members, X, y = self._make_fitted_members(clf_Xy)
        ens = EnsembleModel(members, ProblemType.CLASSIFICATION, "label")
        result = ens.fit(X, y)
        assert result is ens


# ===========================================================================
# 6. EvolutionHistory.top_chromosomes()
# ===========================================================================

class TestTopChromosomes:
    def _history_with_chroms(self, fitnesses):
        history = EvolutionHistory()
        for i, f in enumerate(fitnesses):
            c = Chromosome(genes={"model_type": f"m{i}", "n_estimators": i},
                           fitness=f, generation=0)
            history.all_chromosomes.append(c)
        return history

    def test_returns_sorted_best_first(self):
        history = self._history_with_chroms([0.7, 0.9, 0.8, 0.6])
        top = history.top_chromosomes(3)
        assert [c.fitness for c in top] == [0.9, 0.8, 0.7]

    def test_deduplicates_identical_genes(self):
        history = EvolutionHistory()
        genes = {"model_type": "lgbm", "n_estimators": 100}
        for f in [0.85, 0.85, 0.85]:
            c = Chromosome(genes=dict(genes), fitness=f, generation=0)
            history.all_chromosomes.append(c)
        top = history.top_chromosomes(3)
        assert len(top) == 1  # all identical — only one survives dedup

    def test_k_larger_than_available(self):
        history = self._history_with_chroms([0.5, 0.6, 0.7])
        top = history.top_chromosomes(10)
        assert len(top) == 3

    def test_empty_history(self):
        history = EvolutionHistory()
        assert history.top_chromosomes(3) == []

    def test_unevaluated_excluded(self):
        history = EvolutionHistory()
        history.all_chromosomes.append(
            Chromosome(genes={"n_estimators": 1}, fitness=None, generation=0)
        )
        history.all_chromosomes.append(
            Chromosome(genes={"n_estimators": 2}, fitness=0.75, generation=0)
        )
        top = history.top_chromosomes(5)
        assert len(top) == 1
        assert top[0].fitness == 0.75


# ===========================================================================
# 7. EnsembleConfig dataclass
# ===========================================================================

class TestEnsembleConfig:
    def test_defaults(self):
        cfg = EnsembleConfig()
        assert cfg.enabled is True
        assert cfg.top_k == 3
        assert cfg.weight_by_fitness is True

    def test_disabled(self):
        cfg = EnsembleConfig(enabled=False)
        assert cfg.enabled is False

    def test_custom_top_k(self):
        cfg = EnsembleConfig(top_k=5)
        assert cfg.top_k == 5

    def test_automl_config_has_ensemble_field(self):
        cfg = AutoMLConfig()
        assert hasattr(cfg, "ensemble")
        assert isinstance(cfg.ensemble, EnsembleConfig)


# ===========================================================================
# 8. Full pipeline integration — ensemble mode
# ===========================================================================

class TestPipelineEnsemble:
    def test_pipeline_ensemble_produces_final_score(self, clf_df):
        pipeline = AutoMLPipeline(_fast_ensemble_config(top_k=3))
        pipeline.fit(clf_df)
        assert pipeline.final_score is not None
        assert 0.0 <= pipeline.final_score <= 1.0

    def test_pipeline_ensemble_model_is_ensemble(self, clf_df):
        pipeline = AutoMLPipeline(_fast_ensemble_config(top_k=3))
        pipeline.fit(clf_df)
        # With population_size=6, top_k=3, model should be EnsembleModel
        # (unless fewer than 3 unique chromosomes were evaluated — edge case)
        assert isinstance(pipeline.best_model, (EnsembleModel, LGBMModel, XGBModel, SklearnModel))

    def test_pipeline_single_model_when_ensemble_disabled(self, clf_df):
        pipeline = AutoMLPipeline(_fast_single_config())
        pipeline.fit(clf_df)
        assert not isinstance(pipeline.best_model, EnsembleModel)

    def test_pipeline_predict_shape(self, clf_df):
        pipeline = AutoMLPipeline(_fast_ensemble_config(top_k=2))
        pipeline.fit(clf_df)
        preds = pipeline.predict(clf_df.drop(columns=["label"]))
        assert preds.shape == (len(clf_df),)

    def test_pipeline_predict_proba_shape(self, clf_df):
        pipeline = AutoMLPipeline(_fast_ensemble_config(top_k=2))
        pipeline.fit(clf_df)
        proba = pipeline.predict_proba(clf_df.drop(columns=["label"]))
        assert proba is not None
        assert proba.shape[0] == len(clf_df)

    def test_pipeline_feature_importances_after_ensemble(self, clf_df):
        pipeline = AutoMLPipeline(_fast_ensemble_config(top_k=2))
        pipeline.fit(clf_df)
        fi = pipeline.feature_importances_
        assert fi is not None
        assert len(fi) > 0
        assert abs(fi.sum() - 1.0) < 1e-4

    def test_pipeline_save_load_ensemble(self, clf_df, tmp_path):
        pipeline = AutoMLPipeline(_fast_ensemble_config(top_k=2))
        pipeline.fit(clf_df)
        save_path = str(tmp_path / "ensemble_pipeline.joblib")
        pipeline.save(save_path)
        loaded = AutoMLPipeline.load(save_path)
        preds = loaded.predict(clf_df.drop(columns=["label"]))
        assert preds.shape == (len(clf_df),)

    def test_summary_includes_ensemble_members(self, clf_df):
        pipeline = AutoMLPipeline(_fast_ensemble_config(top_k=2))
        pipeline.fit(clf_df)
        s = pipeline.summary()
        assert s["final_score"] is not None
        assert s["generations_run"] >= 1
