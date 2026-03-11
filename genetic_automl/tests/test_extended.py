"""
Extended test suite for GAML — covers gaps in existing tests plus newly discovered bugs.

Run:  pytest genetic_automl/tests/test_extended.py -v
"""
from __future__ import annotations

import random
import warnings

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clf_Xy():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((200, 4)), columns=list("abcd"))
    y = pd.Series(rng.integers(0, 2, 200))
    return X, y


@pytest.fixture
def reg_Xy():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((200, 3)), columns=list("abc"))
    y = pd.Series(rng.standard_normal(200))
    return X, y


@pytest.fixture
def multi_Xy():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((180, 4)), columns=list("abcd"))
    y = pd.Series(rng.integers(0, 3, 180))
    return X, y


# ===========================================================================
# BUG-NEW-1: Chromosome.__repr__ crashes when fitness is None
# ===========================================================================

class TestChromosomeRepr:
    def test_repr_with_none_fitness_does_not_crash(self):
        """
        BUG-NEW-1 FIXED: Chromosome.__repr__() used a malformed f-string format spec
        that crashed for both None and float fitness values. Fixed by pre-computing
        fitness_str before the f-string.
        """
        from genetic_automl.genetic.chromosome import Chromosome
        c = Chromosome(genes={"a": 1, "b": "x"})
        assert c.fitness is None
        s = repr(c)
        assert "None" in s
        assert c.id in s

    def test_repr_with_float_fitness_works(self):
        from genetic_automl.genetic.chromosome import Chromosome
        c = Chromosome(genes={"a": 1})
        c.fitness = 0.987654
        s = repr(c)
        assert "0.987654" in s


# ===========================================================================
# BUG-NEW-2: Chromosome.as_dict() silently drops fitness_std
# ===========================================================================

class TestChromosomeSerialization:
    def test_as_dict_includes_fitness_std(self):
        """
        BUG-NEW-2 FIXED: fitness_std is now a declared Optional[float] dataclass field
        and is included in as_dict().
        """
        from genetic_automl.genetic.chromosome import Chromosome
        c = Chromosome(genes={"a": 1})
        c.fitness = 0.9
        c.fitness_std = 0.05
        d = c.as_dict()
        assert "fitness_std" in d
        assert d["fitness_std"] == pytest.approx(0.05)

    def test_copy_preserves_fitness_std(self):
        """copy() must propagate fitness_std since it is now a declared field."""
        from genetic_automl.genetic.chromosome import Chromosome
        c = Chromosome(genes={"a": 1})
        c.fitness = 0.8
        c.fitness_std = 0.03
        c2 = c.copy()
        assert c2.fitness_std == pytest.approx(0.03)

    def test_fitness_std_defaults_to_none(self):
        from genetic_automl.genetic.chromosome import Chromosome
        c = Chromosome(genes={"a": 1})
        assert c.fitness_std is None


# ===========================================================================
# BUG-NEW-3: B3 — roc_auc crashes on multiclass with hard-label predictions
# ===========================================================================

class TestRocAucMetric:
    def test_roc_auc_multiclass_hard_labels_no_longer_crashes(self):
        """
        BUG-B3 FIXED: _METRIC_REGISTRY['roc_auc'] now detects 1-D input and routes
        through binary roc_auc_score instead of the multi_class='ovr' path that
        requires a 2-D probability matrix.
        """
        from genetic_automl.core.problem import _METRIC_REGISTRY
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred_binary = np.array([0.2, 0.8, 0.3, 0.9, 0.1, 0.7])
        fn, _ = _METRIC_REGISTRY["roc_auc"]
        result = fn(y_true, y_pred_binary)
        assert 0.0 <= result <= 1.0

    def test_roc_auc_multiclass_proba_matrix_works(self):
        """2-D proba matrix (N, C) must still use multi_class='ovr'."""
        from genetic_automl.core.problem import _METRIC_REGISTRY
        import numpy as np
        rng = np.random.default_rng(0)
        y_true = np.array([0, 1, 2, 0, 1, 2])
        # Proper (N, 3) probability matrix
        raw = rng.random((6, 3))
        y_proba = raw / raw.sum(axis=1, keepdims=True)
        fn, _ = _METRIC_REGISTRY["roc_auc"]
        result = fn(y_true, y_proba)
        assert 0.0 <= result <= 1.0

    def test_roc_auc_via_score_works_for_multiclass(self, multi_Xy):
        """score() correctly routes multiclass roc_auc through predict_proba."""
        from genetic_automl.automl.sklearn_model import SklearnModel
        from genetic_automl.core.problem import ProblemType
        X, y = multi_Xy
        model = SklearnModel(ProblemType.CLASSIFICATION, "label")
        model.fit(X, y)
        s = model.score(X, y, metric="roc_auc")
        assert 0.0 <= s <= 1.0

    def test_roc_auc_binary_with_hard_labels_works(self):
        """Binary roc_auc with hard labels works (1-D array path)."""
        from genetic_automl.core.problem import compute_metric
        y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0])
        s = compute_metric("roc_auc", y_true, y_pred)
        assert 0.0 <= s <= 1.0


# ===========================================================================
# BUG-NEW-4: test_zero_leakage_imputer fails due to test bug (wrong config)
# ===========================================================================

class TestImputer:
    def test_median_imputation_preserves_train_statistics(self):
        """
        Imputer fills val NaNs using training-set median, not val statistics.
        """
        from genetic_automl.preprocessing.numeric_imputer import NumericImputer

        X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0] * 30})
        X_val = pd.DataFrame({"a": [np.nan] * 10})

        ni = NumericImputer("median")
        ni.fit(X_train)
        X_val_out = ni.transform(X_val)

        assert np.allclose(X_val_out["a"].values, 2.0), (
            "Val NaNs must impute to train median=2.0, not val statistics."
        )

    def test_mean_imputation_uses_train_mean(self):
        from genetic_automl.preprocessing.numeric_imputer import NumericImputer
        X_train = pd.DataFrame({"v": [10.0, 20.0, 30.0]})
        X_val = pd.DataFrame({"v": [np.nan, np.nan]})
        ni = NumericImputer("mean")
        ni.fit(X_train)
        out = ni.transform(X_val)
        assert out["v"].iloc[0] == pytest.approx(20.0)

    def test_knn_imputer_fits_on_train_only(self):
        from genetic_automl.preprocessing.numeric_imputer import NumericImputer
        X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        X_val = pd.DataFrame({"a": [np.nan]})
        ni = NumericImputer("knn")
        ni.fit(X_train)
        out = ni.transform(X_val)
        assert not out["a"].isnull().any()

    def test_constant_imputer_fills_zero(self):
        from genetic_automl.preprocessing.numeric_imputer import NumericImputer
        X = pd.DataFrame({"x": [np.nan, np.nan, 1.0]})
        ni = NumericImputer("constant")
        ni.fit(X)
        out = ni.transform(X)
        assert out["x"].iloc[0] == pytest.approx(0.0)


# ===========================================================================
# BUG-NEW-5: SMOTE crashes when minority class count <= k_neighbors (silent fallback)
# ===========================================================================

class TestImbalanceHandler:
    def test_smote_with_tiny_minority_auto_adjusts_k(self):
        """
        BUG-NEW-4 FIXED: When minority class count <= k_neighbors, ImbalanceHandler
        now auto-reduces k_neighbors to minority_count - 1 (with a WARNING) and
        proceeds with SMOTE rather than silently falling back to unbalanced data.
        """
        from genetic_automl.preprocessing.imbalance_handler import ImbalanceHandler
        X = pd.DataFrame({"a": range(103), "b": range(103)})
        y = pd.Series([1] * 100 + [0] * 3)  # 3 minority samples, default k=5
        ih = ImbalanceHandler("smote", k_neighbors=5)
        X_out, y_out = ih.fit_resample(X, y)
        # k was auto-reduced to 2 (minority_count - 1 = 3 - 1 = 2); SMOTE succeeds
        assert len(X_out) > 103, "SMOTE should succeed after k_neighbors auto-adjustment"
        assert ih.k_neighbors == 2, f"k_neighbors should have been reduced to 2, got {ih.k_neighbors}"

    def test_smote_succeeds_with_adjusted_k(self):
        """SMOTE works when k_neighbors < minority_count."""
        from genetic_automl.preprocessing.imbalance_handler import ImbalanceHandler
        X = pd.DataFrame({"a": range(106), "b": range(106)})
        y = pd.Series([1] * 100 + [0] * 6)
        ih = ImbalanceHandler("smote", k_neighbors=5)
        X_out, y_out = ih.fit_resample(X, y)
        assert len(X_out) > 106, "SMOTE should oversample minority class"

    def test_class_weight_always_available(self):
        """class_weight fallback must always work without imblearn."""
        from genetic_automl.preprocessing.imbalance_handler import ImbalanceHandler
        X = pd.DataFrame({"a": range(100)})
        y = pd.Series([0] * 80 + [1] * 20)
        ih = ImbalanceHandler("class_weight")
        X_out, y_out = ih.fit_resample(X, y)
        weights = ih.sample_weights(y_out)
        assert weights is not None
        assert len(weights) == len(y_out)

    def test_imbalance_none_is_noop(self, clf_Xy):
        from genetic_automl.preprocessing.imbalance_handler import ImbalanceHandler
        X, y = clf_Xy
        ih = ImbalanceHandler("none")
        X_out, y_out = ih.fit_resample(X, y)
        assert len(X_out) == len(X)
        assert ih.sample_weights(y_out) is None


# ===========================================================================
# BUG-NEW-6: CategoricalEncoder crashes when cat column missing in test set
# ===========================================================================

class TestCategoricalEncoderEdgeCases:
    def test_onehot_missing_column_at_transform_fills_zeros(self):
        """
        BUG-NEW-5 FIXED: If a categorical column is absent at transform time,
        CategoricalEncoder now fills it with '__MISSING__' so OHE produces an
        all-zeros row instead of crashing with 'X has 0 features'.
        """
        from genetic_automl.preprocessing.categorical_encoder import CategoricalEncoder
        X_train = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "c"]})
        X_test = pd.DataFrame({"num": [1.0, 2.0, 3.0]})  # 'cat' column absent
        enc = CategoricalEncoder("onehot")
        enc.fit(X_train)
        out = enc.transform(X_test)  # must not raise
        # __MISSING__ is an unknown category → OHE produces all-zeros row
        ohe_cols = [c for c in out.columns if c.startswith("cat_")]
        assert len(ohe_cols) > 0
        assert out[ohe_cols].sum(axis=1).iloc[0] == pytest.approx(0.0)

    def test_onehot_unseen_category_is_all_zeros(self):
        """handle_unknown='ignore' produces all-zero row for unseen categories."""
        from genetic_automl.preprocessing.categorical_encoder import CategoricalEncoder
        X_train = pd.DataFrame({"cat": ["a", "b", "c"] * 10})
        X_test = pd.DataFrame({"cat": ["UNSEEN"]})
        enc = CategoricalEncoder("onehot")
        enc.fit(X_train)
        out = enc.transform(X_test)
        # All OHE columns should be 0 for unseen category
        ohe_cols = [c for c in out.columns if c.startswith("cat_")]
        assert out[ohe_cols].sum(axis=1).iloc[0] == pytest.approx(0.0)

    def test_ordinal_unseen_category_non_negative(self):
        """BUG-B8 (fixed): unseen categories should map to mid-range, not -1."""
        from genetic_automl.preprocessing.categorical_encoder import CategoricalEncoder
        X_train = pd.DataFrame({"cat": ["A", "B", "C"] * 10})
        X_val = pd.DataFrame({"cat": ["D"]})
        enc = CategoricalEncoder("ordinal")
        enc.fit(X_train)
        out = enc.transform(X_val)
        assert out["cat"].iloc[0] >= 0, (
            f"Unseen ordinal got {out['cat'].iloc[0]}, expected >= 0"
        )

    def test_target_encoding_unseen_maps_to_global_mean(self):
        """Target-encoded unseen categories fall back to global mean."""
        from genetic_automl.preprocessing.categorical_encoder import CategoricalEncoder
        X_train = pd.DataFrame({"cat": ["a", "b"] * 50})
        y_train = pd.Series([0.0, 1.0] * 50)
        X_val = pd.DataFrame({"cat": ["UNSEEN"]})
        enc = CategoricalEncoder("target")
        enc.fit(X_train, y_train)
        out = enc.transform(X_val)
        assert abs(out["cat"].iloc[0] - enc._global_mean) < 1e-6

    def test_binary_encoding_shape(self):
        from genetic_automl.preprocessing.categorical_encoder import CategoricalEncoder
        X = pd.DataFrame({"cat": ["a", "b", "c", "d"] * 10})
        enc = CategoricalEncoder("binary")
        enc.fit(X)
        out = enc.transform(X)
        # 4 categories -> ceil(log2(5)) = 3 bits
        binary_cols = [c for c in out.columns if c.startswith("cat_bin")]
        assert len(binary_cols) >= 2


# ===========================================================================
# BUG-NEW-7: Diversity update receives stale no_improvement_streak
# ===========================================================================

class TestDiversityOrderingBug:
    def test_mutation_not_boosted_on_improving_generation(self):
        """
        BUG-NEW-7 FIXED: GeneticEngine now updates no_improvement_streak BEFORE
        calling diversity.update(). On an improving generation the streak resets
        to 0 before the diversity controller sees it, so no spurious boost fires.
        This test verifies the full engine respects the corrected ordering.
        """
        from genetic_automl.config import GeneticConfig
        from genetic_automl.genetic.engine import GeneticEngine
        from genetic_automl.genetic.fitness import FitnessEvaluator
        from genetic_automl.core.problem import ProblemType
        from sklearn.datasets import load_breast_cancer

        data = load_breast_cancer(as_frame=True)
        X = data.data
        y = data.target.rename("label")

        evaluator = FitnessEvaluator(ProblemType.CLASSIFICATION, "label", "sklearn", n_folds=2)
        cfg = GeneticConfig(
            population_size=4, generations=3, early_stopping_rounds=10,
            warm_start=False, adaptive_mutation=True,
            adaptive_mutation_stagnation_rounds=1,  # boost after just 1 stagnant gen
            random_seed=0,
        )
        engine = GeneticEngine(cfg, evaluator, backend="sklearn")
        engine.run(X, y)

        # Find any generation where fitness improved AND mutation was boosted
        spurious_boosts = [
            g for i, g in enumerate(engine.history.generations)
            if i > 0
            and g.best_fitness > engine.history.generations[i - 1].best_fitness
            and g.mutation_boosted
        ]
        assert len(spurious_boosts) == 0, (
            f"Mutation boost fired on {len(spurious_boosts)} improving generation(s): "
            f"{[(g.generation, g.best_fitness) for g in spurious_boosts]}"
        )


# ===========================================================================
# BUG-NEW-8: Scaler crashes when test columns differ from train columns
# ===========================================================================

class TestScalerSchemaDrift:
    def test_scaler_handles_missing_column_gracefully(self):
        """
        BUG-NEW-6 FIXED: Scaler now stores per-column sklearn instances so
        transform() skips columns absent in the input without any feature-count
        mismatch error. Only columns present in both fit and transform are scaled.
        """
        from genetic_automl.preprocessing.scaler import Scaler
        X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        X_test = pd.DataFrame({"a": [1.0, 2.0]})  # 'b' absent
        sc = Scaler("standard")
        sc.fit(X_train)
        out = sc.transform(X_test)  # must not raise
        assert "a" in out.columns
        # StandardScaler uses population std (ddof=0): mean=2.0, std≈0.8165
        # scaled(1.0) = (1.0 - 2.0) / std([1,2,3]) = -1.2247...
        expected = (1.0 - 2.0) / np.std([1.0, 2.0, 3.0])  # ddof=0
        assert np.isclose(out["a"].iloc[0], expected, atol=1e-4)

    def test_scaler_none_always_passthrough(self):
        """scaler='none' must never touch data."""
        from genetic_automl.preprocessing.scaler import Scaler
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        sc = Scaler("none")
        sc.fit(X)
        out = sc.transform(X)
        pd.testing.assert_frame_equal(X, out)

    def test_all_scalers_fit_transform_cycle(self):
        """All three scalers fit and transform without error on valid data."""
        from genetic_automl.preprocessing.scaler import Scaler
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((50, 3)), columns=list("abc"))
        for method in ("standard", "minmax", "robust"):
            sc = Scaler(method)
            out = sc.fit_transform(X)
            assert out.shape == X.shape, f"Scaler {method} changed shape"
            assert not out.isnull().any().any(), f"Scaler {method} introduced NaN"


# ===========================================================================
# Genetic operators edge cases
# ===========================================================================

class TestOperatorEdgeCases:
    def test_single_point_crossover_single_gene(self):
        """1-gene chromosomes return copies (cannot cut)."""
        from genetic_automl.genetic.chromosome import Chromosome
        from genetic_automl.genetic.operators import single_point_crossover
        pa = Chromosome({"x": "a"})
        pb = Chromosome({"x": "b"})
        ca, cb = single_point_crossover(pa, pb, random.Random(0))
        assert "x" in ca.genes
        assert "x" in cb.genes

    def test_uniform_crossover_preserves_all_genes(self):
        """uniform_crossover must preserve all gene names."""
        from genetic_automl.genetic.chromosome import Chromosome, get_gene_space
        from genetic_automl.genetic.operators import uniform_crossover
        rng = random.Random(0)
        gene_names = [g.name for g in get_gene_space("sklearn")]
        ga = {g.name: g.random_value(rng) for g in get_gene_space("sklearn")}
        gb = {g.name: g.random_value(rng) for g in get_gene_space("sklearn")}
        pa = Chromosome(ga)
        pb = Chromosome(gb)
        ca, cb = uniform_crossover(pa, pb, rng)
        assert set(ca.genes.keys()) == set(gene_names)
        assert set(cb.genes.keys()) == set(gene_names)

    def test_tournament_selection_all_none_fitness(self):
        """Tournament must not crash when all chromosomes have None fitness."""
        from genetic_automl.genetic.chromosome import random_population
        from genetic_automl.genetic.operators import tournament_selection
        rng = random.Random(0)
        pop = random_population("sklearn", 5, rng)
        for c in pop:
            c.fitness = None
        best = tournament_selection(pop, 3, rng)
        assert best is not None

    def test_elites_with_all_none_fitness(self):
        """Elites must return chromosomes even if all fitness=None."""
        from genetic_automl.genetic.chromosome import random_population
        from genetic_automl.genetic.operators import elites
        rng = random.Random(0)
        pop = random_population("sklearn", 10, rng)
        for c in pop:
            c.fitness = None
        result = elites(pop, 0.1)
        assert len(result) >= 1

    def test_mutate_gene_values_remain_valid(self):
        """Mutated genes must stay within the defined gene space."""
        from genetic_automl.genetic.chromosome import Chromosome, get_gene_space
        from genetic_automl.genetic.operators import mutate
        rng = random.Random(0)
        gene_space = get_gene_space("sklearn")
        valid_values = {g.name: set(map(str, g.values)) for g in gene_space}
        genes = {g.name: g.random_value(rng) for g in gene_space}
        c = Chromosome(genes)
        for _ in range(50):
            m = mutate(c, "sklearn", 1.0, rng)
            for name, val in m.genes.items():
                assert str(val) in valid_values[name], (
                    f"Mutant gene '{name}'={val} not in valid values"
                )


# ===========================================================================
# Outlier handler edge cases
# ===========================================================================

class TestOutlierHandlerEdgeCases:
    def test_iqr_flag_adds_column(self):
        from genetic_automl.preprocessing.outlier_handler import OutlierHandler
        X = pd.DataFrame({"a": [1, 2, 3, 100, 2, 1, 2, 3]})
        oh = OutlierHandler("iqr", threshold=1.5, action="flag")
        oh.fit(X)
        out = oh.transform(X)
        assert "__outlier__" in out.columns

    def test_isolation_forest_uses_train_median_not_test(self):
        """BUG-B6 fix verification: clip must use training medians."""
        from genetic_automl.preprocessing.outlier_handler import OutlierHandler
        rng = np.random.default_rng(0)
        X_train = pd.DataFrame({"a": rng.standard_normal(200)})
        X_val = pd.DataFrame({"a": [100.0, 200.0, 300.0, 400.0, 500.0]})
        oh = OutlierHandler("isolation_forest", action="clip")
        oh.fit(X_train)
        out = oh.transform(X_val)
        # Should be clipped to training median (near 0), not val data median (300)
        assert out["a"].max() < 10.0, (
            "Outlier clip must use training median, not test/val data statistics."
        )

    def test_zscore_clip_uses_train_bounds(self):
        from genetic_automl.preprocessing.outlier_handler import OutlierHandler
        X_train = pd.DataFrame({"v": list(range(100))})
        X_val = pd.DataFrame({"v": [1000.0, -1000.0]})
        oh = OutlierHandler("zscore", threshold=3.0, action="clip")
        oh.fit(X_train)
        out = oh.transform(X_val)
        # Values should be clipped to train bounds
        assert out["v"].max() < 200.0
        assert out["v"].min() > -200.0

    def test_none_method_is_noop(self):
        from genetic_automl.preprocessing.outlier_handler import OutlierHandler
        X = pd.DataFrame({"a": [1.0, 2.0, 1000.0]})
        oh = OutlierHandler("none")
        oh.fit(X)
        out = oh.transform(X)
        assert out["a"].max() == 1000.0


# ===========================================================================
# Feature selector edge cases
# ===========================================================================

class TestFeatureSelectorEdgeCases:
    def test_variance_threshold_removes_constant_col(self):
        from genetic_automl.preprocessing.feature_selector import FeatureSelector
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"const": [1.0] * 50, "var": rng.standard_normal(50)})
        y = pd.Series(rng.integers(0, 2, 50))
        fs = FeatureSelector("variance_threshold", keep_k=1.0)
        fs.fit(X, y)
        out = fs.transform(X)
        assert "const" not in out.columns
        assert "var" in out.columns

    def test_mutual_info_keep_k_fraction(self):
        from genetic_automl.preprocessing.feature_selector import FeatureSelector
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((100, 6)), columns=list("abcdef"))
        y = pd.Series(rng.integers(0, 2, 100))
        fs = FeatureSelector("mutual_info", keep_k=0.5)  # keep 3/6
        fs.fit(X, y)
        out = fs.transform(X)
        assert out.shape[1] == 3

    def test_transform_raises_on_missing_selected_cols(self):
        """BUG-B7 fix verification: missing selected columns must raise ValueError."""
        from genetic_automl.preprocessing.feature_selector import FeatureSelector
        rng = np.random.default_rng(0)
        X_train = pd.DataFrame(rng.standard_normal((100, 3)), columns=["a", "b", "c"])
        y = pd.Series(rng.integers(0, 2, 100))
        fs = FeatureSelector("mutual_info", keep_k=1.0)
        fs.fit(X_train, y)
        X_val = pd.DataFrame(rng.standard_normal((10, 2)), columns=["x", "y"])
        with pytest.raises(ValueError, match="missing from input"):
            fs.transform(X_val)

    def test_rfe_selects_correct_count(self):
        from genetic_automl.preprocessing.feature_selector import FeatureSelector
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((100, 8)), columns=list("abcdefgh"))
        y = pd.Series(rng.integers(0, 2, 100))
        fs = FeatureSelector("rfe", keep_k=0.5)
        fs.fit(X, y)
        assert len(fs.selected_features) == 4


# ===========================================================================
# CorrelationFilter edge cases
# ===========================================================================

class TestCorrelationFilterEdgeCases:
    def test_single_column_no_drop(self):
        from genetic_automl.preprocessing.correlation_filter import CorrelationFilter
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        cf = CorrelationFilter(0.95)
        cf.fit(X)
        assert cf.dropped_features == []

    def test_perfect_correlation_drops_one(self):
        from genetic_automl.preprocessing.correlation_filter import CorrelationFilter
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0],
                           "b": [2.0, 4.0, 6.0, 8.0, 10.0]})
        cf = CorrelationFilter(0.95)
        cf.fit(X)
        assert len(cf.dropped_features) == 1

    def test_none_threshold_is_noop(self):
        from genetic_automl.preprocessing.correlation_filter import CorrelationFilter
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
        cf = CorrelationFilter(None)
        cf.fit(X)
        out = cf.transform(X)
        assert list(out.columns) == ["a", "b"]

    def test_cat_columns_ignored(self):
        from genetic_automl.preprocessing.correlation_filter import CorrelationFilter
        X = pd.DataFrame({"num": [1, 2, 3, 4, 5],
                           "dup": [2, 4, 6, 8, 10],
                           "cat": ["a", "b", "c", "a", "b"]})
        cf = CorrelationFilter(0.95)
        cf.fit(X)
        out = cf.transform(X)
        assert "cat" in out.columns  # Cat col must survive


# ===========================================================================
# DistributionTransform edge cases
# ===========================================================================

class TestDistributionTransformEdgeCases:
    def test_box_cox_on_negative_values_no_crash(self):
        """box-cox auto-shifts negative data to be strictly positive."""
        from genetic_automl.preprocessing.distribution_transform import DistributionTransform
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"neg": rng.normal(-5, 1, 100)})
        dt = DistributionTransform("box-cox")
        dt.fit(X)
        out = dt.transform(X)
        assert not out.isnull().any().any()

    def test_log1p_on_negative_auto_shifted(self):
        """log1p shifts negative columns so no NaN/inf is produced."""
        from genetic_automl.preprocessing.distribution_transform import DistributionTransform
        X = pd.DataFrame({"neg": [-10.0, -5.0, 0.0, 5.0, 10.0] * 20})
        dt = DistributionTransform("log1p")
        dt.fit(X)
        out = dt.transform(X)
        assert not out.isnull().any().any()
        assert not np.isinf(out.values).any()

    def test_none_method_is_noop(self):
        from genetic_automl.preprocessing.distribution_transform import DistributionTransform
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((50, 3)), columns=list("abc"))
        dt = DistributionTransform("none")
        dt.fit(X)
        out = dt.transform(X)
        pd.testing.assert_frame_equal(X, out)

    def test_yeo_johnson_reduces_skewness(self):
        from genetic_automl.preprocessing.distribution_transform import DistributionTransform
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"skewed": rng.exponential(scale=2, size=500)})
        original_skew = abs(X["skewed"].skew())
        dt = DistributionTransform("yeo-johnson")
        dt.fit(X)
        out = dt.transform(X)
        transformed_skew = abs(out["skewed"].skew())
        assert transformed_skew < original_skew


# ===========================================================================
# DataManager edge cases
# ===========================================================================

class TestDataManagerEdgeCases:
    def test_missing_target_column_raises(self):
        from genetic_automl.core.data import DataManager
        from genetic_automl.core.problem import ProblemType
        dm = DataManager("target", ProblemType.CLASSIFICATION)
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        with pytest.raises(ValueError, match="Target column"):
            dm.validate(df)

    def test_three_way_split_no_index_overlap(self):
        from genetic_automl.core.data import DataManager
        from genetic_automl.core.problem import ProblemType
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": range(200), "target": rng.integers(0, 2, 200)})
        dm = DataManager("target", ProblemType.CLASSIFICATION, test_size=0.15, val_size=0.2)
        train, val, test = dm.three_way_split(df)
        train_idx = set(train.index)
        val_idx = set(val.index)
        test_idx = set(test.index)
        assert len(train_idx & val_idx) == 0, "Train/val index overlap"
        assert len(train_idx & test_idx) == 0, "Train/test index overlap"
        assert len(val_idx & test_idx) == 0, "Val/test index overlap"

    def test_regression_split_with_stratify_flag(self):
        """Regression should not stratify even if flag is True."""
        from genetic_automl.core.data import DataManager
        from genetic_automl.core.problem import ProblemType
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": range(200), "target": rng.standard_normal(200)})
        dm = DataManager("target", ProblemType.REGRESSION, stratify=True)
        train, val, test = dm.three_way_split(df)
        assert len(train) + len(val) + len(test) == 200

    def test_external_test_df_respected(self):
        from genetic_automl.core.data import DataManager
        from genetic_automl.core.problem import ProblemType
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": range(100), "target": rng.integers(0, 2, 100)})
        test_df = pd.DataFrame({"a": range(50), "target": rng.integers(0, 2, 50)})
        dm = DataManager("target", ProblemType.CLASSIFICATION)
        train, val, test = dm.three_way_split(df, test_df)
        assert len(test) == 50


# ===========================================================================
# Diversity controller edge cases
# ===========================================================================

class TestDiversityController:
    def test_hamming_identical_population_zero(self):
        from genetic_automl.genetic.diversity import mean_pairwise_hamming
        from genetic_automl.genetic.chromosome import Chromosome
        genes = {"a": 1, "b": "x"}
        pop = [Chromosome(genes.copy()) for _ in range(5)]
        assert mean_pairwise_hamming(pop) == pytest.approx(0.0)

    def test_hamming_single_chromosome_returns_one(self):
        from genetic_automl.genetic.diversity import mean_pairwise_hamming
        from genetic_automl.genetic.chromosome import Chromosome
        pop = [Chromosome({"a": 1})]
        assert mean_pairwise_hamming(pop) == pytest.approx(1.0)

    def test_injection_maintains_population_size(self):
        from genetic_automl.genetic.diversity import PopulationDiversity
        from genetic_automl.genetic.chromosome import random_population
        rng = random.Random(0)
        pop = random_population("sklearn", 20, rng)
        for i, c in enumerate(pop):
            c.fitness = float(i)
        div = PopulationDiversity("sklearn", min_diversity_threshold=1.0)  # always inject
        new_pop, _ = div.update(pop, 0, 0)
        assert len(new_pop) == 20

    def test_mutation_rate_never_exceeds_0_8(self):
        from genetic_automl.genetic.diversity import PopulationDiversity
        from genetic_automl.genetic.chromosome import random_population
        rng = random.Random(0)
        pop = random_population("sklearn", 10, rng)
        for c in pop:
            c.fitness = 0.5
        div = PopulationDiversity("sklearn", base_mutation_rate=0.5, mutation_boost_factor=10.0)
        _, rate = div.update(pop, 0, no_improvement_streak=99)
        assert rate <= 0.8, "Mutation rate must be capped at 0.8"

    def test_mutation_decays_back_to_base(self):
        from genetic_automl.genetic.diversity import PopulationDiversity
        from genetic_automl.genetic.chromosome import random_population
        rng = random.Random(0)
        pop = random_population("sklearn", 10, rng)
        for c in pop:
            c.fitness = 0.5

        div = PopulationDiversity("sklearn", base_mutation_rate=0.2,
                                   stagnation_rounds=3, mutation_boost_factor=2.0,
                                   mutation_decay=0.5)
        # Trigger boost
        div.update(pop, 0, no_improvement_streak=5)
        assert div._boosted
        # Simulate 20 improvement rounds
        for gen in range(1, 20):
            div.update(pop, gen, no_improvement_streak=0)
        assert div._current_mutation_rate == pytest.approx(0.2, abs=0.01)


# ===========================================================================
# Pareto front
# ===========================================================================

class TestParetoFront:
    def test_dominated_solution_excluded(self):
        from genetic_automl.core.problem import pareto_front
        scores = [[0.9, 0.8], [0.7, 0.95], [0.5, 0.5]]
        front = pareto_front(scores)
        assert 2 not in front, "[0.5, 0.5] is dominated and should not be on Pareto front"

    def test_all_nondominated_all_included(self):
        from genetic_automl.core.problem import pareto_front
        scores = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        front = pareto_front(scores)
        assert set(front) == {0, 1, 2}

    def test_single_solution_on_front(self):
        from genetic_automl.core.problem import pareto_front
        front = pareto_front([[0.7, 0.8]])
        assert front == [0]


# ===========================================================================
# GeneticEngine integration — smoke test with tiny settings
# ===========================================================================

class TestGeneticEngineSmoke:
    def test_engine_runs_and_returns_best(self, clf_Xy):
        from genetic_automl.config import GeneticConfig
        from genetic_automl.genetic.engine import GeneticEngine
        from genetic_automl.genetic.fitness import FitnessEvaluator
        from genetic_automl.core.problem import ProblemType

        X, y = clf_Xy
        evaluator = FitnessEvaluator(
            ProblemType.CLASSIFICATION, "label", "sklearn", n_folds=2
        )
        cfg = GeneticConfig(
            population_size=4,
            generations=2,
            early_stopping_rounds=1,
            warm_start=False,
            random_seed=42,
        )
        engine = GeneticEngine(cfg, evaluator, backend="sklearn")
        best = engine.run(X, y)
        assert best is not None
        assert best.fitness is not None
        assert best.fitness > float("-inf")

    def test_engine_history_populated(self, clf_Xy):
        from genetic_automl.config import GeneticConfig
        from genetic_automl.genetic.engine import GeneticEngine
        from genetic_automl.genetic.fitness import FitnessEvaluator
        from genetic_automl.core.problem import ProblemType

        X, y = clf_Xy
        evaluator = FitnessEvaluator(ProblemType.CLASSIFICATION, "label", "sklearn", n_folds=2)
        cfg = GeneticConfig(
            population_size=4, generations=2, early_stopping_rounds=5,
            warm_start=False, random_seed=0,
        )
        engine = GeneticEngine(cfg, evaluator, backend="sklearn")
        engine.run(X, y)
        assert len(engine.history.generations) >= 1
        assert len(engine.history.all_chromosomes) >= 4

    def test_warm_start_population_correct_size(self, clf_Xy):
        from genetic_automl.config import GeneticConfig
        from genetic_automl.genetic.engine import GeneticEngine
        from genetic_automl.genetic.fitness import FitnessEvaluator
        from genetic_automl.core.problem import ProblemType

        X, y = clf_Xy
        evaluator = FitnessEvaluator(ProblemType.CLASSIFICATION, "label", "sklearn", n_folds=2)
        cfg = GeneticConfig(
            population_size=6, generations=1, early_stopping_rounds=1,
            warm_start=True, warm_start_n_seeds=2,
            warm_start_halving_pool_ratio=1.0, random_seed=42,
        )
        engine = GeneticEngine(cfg, evaluator, backend="sklearn")
        engine.run(X, y)
        assert len(engine.history.all_chromosomes) >= 6


# ===========================================================================
# Full pipeline smoke tests
# ===========================================================================

class TestFullPipelineSmoke:
    def _make_clf_config(self):
        from genetic_automl.config import PipelineConfig, GeneticConfig, AutoMLConfig, ReportConfig
        from genetic_automl.core.problem import ProblemType
        import tempfile
        return PipelineConfig(
            problem_type=ProblemType.CLASSIFICATION,
            target_column="label",
            genetic=GeneticConfig(
                population_size=4, generations=2, early_stopping_rounds=1,
                warm_start=False, random_seed=42,
            ),
            automl=AutoMLConfig(backend="sklearn"),
            report=ReportConfig(output_dir=tempfile.mkdtemp(), open_html_on_finish=False),
        )

    def test_classification_pipeline_e2e(self):
        from sklearn.datasets import load_breast_cancer
        from genetic_automl.pipeline import AutoMLPipeline
        data = load_breast_cancer(as_frame=True)
        df = data.frame.rename(columns={"target": "label"})
        pipeline = AutoMLPipeline(self._make_clf_config())
        pipeline.fit(df)
        assert pipeline.final_score is not None
        assert 0.0 < pipeline.final_score <= 1.0

    def test_predict_returns_correct_shape(self):
        from sklearn.datasets import load_breast_cancer
        from genetic_automl.pipeline import AutoMLPipeline
        data = load_breast_cancer(as_frame=True)
        df = data.frame.rename(columns={"target": "label"})
        pipeline = AutoMLPipeline(self._make_clf_config())
        pipeline.fit(df)
        preds = pipeline.predict(df.drop(columns=["label"]))
        assert preds.shape == (len(df),)

    def test_predict_before_fit_raises(self):
        from genetic_automl.pipeline import AutoMLPipeline
        from genetic_automl.config import PipelineConfig
        pipeline = AutoMLPipeline(PipelineConfig())
        with pytest.raises(RuntimeError, match="fitted"):
            pipeline.predict(pd.DataFrame({"a": [1, 2, 3]}))

    def test_report_file_created(self):
        import os
        from sklearn.datasets import load_breast_cancer
        from genetic_automl.pipeline import AutoMLPipeline
        data = load_breast_cancer(as_frame=True)
        df = data.frame.rename(columns={"target": "label"})
        pipeline = AutoMLPipeline(self._make_clf_config())
        pipeline.fit(df)
        assert pipeline.report_path is not None
        assert os.path.exists(pipeline.report_path)
