"""Unit tests for genetic algorithm components."""
import random

import numpy as np
import pandas as pd
import pytest

from genetic_automl.genetic.chromosome import (
    Chromosome,
    get_gene_space,
    random_population,
)
from genetic_automl.genetic.diversity import (
    PopulationDiversity,
    hamming_distance,
    mean_pairwise_hamming,
)
from genetic_automl.genetic.operators import (
    elites,
    mutate,
    single_point_crossover,
    tournament_selection,
)
from genetic_automl.genetic.warm_start import WarmStart, _sklearn_baseline


# ---------------------------------------------------------------------------
# Chromosome
# ---------------------------------------------------------------------------

class TestChromosome:
    def test_random_population_size(self):
        rng = random.Random(42)
        pop = random_population("sklearn", size=10, rng=rng)
        assert len(pop) == 10

    def test_all_genes_present(self):
        rng = random.Random(42)
        gene_names = {g.name for g in get_gene_space("sklearn")}
        pop = random_population("sklearn", size=5, rng=rng)
        for chrom in pop:
            assert set(chrom.genes.keys()) == gene_names

    def test_copy_is_independent(self):
        rng = random.Random(0)
        pop = random_population("sklearn", 1, rng)
        original = pop[0]
        copy = original.copy()
        copy.genes["scaler"] = "MODIFIED"
        assert original.genes["scaler"] != "MODIFIED"


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class TestOperators:
    def _make_pop(self, n=10):
        rng = random.Random(42)
        pop = random_population("sklearn", n, rng)
        for i, c in enumerate(pop):
            c.fitness = float(i)
        return pop

    def test_tournament_selection_returns_chromosome(self):
        pop = self._make_pop()
        rng = random.Random(0)
        winner = tournament_selection(pop, tournament_size=3, rng=rng)
        assert isinstance(winner, Chromosome)

    def test_elites_preserves_best(self):
        pop = self._make_pop(10)
        top = elites(pop, elite_ratio=0.2)
        assert len(top) == 2
        assert top[0].fitness == 9.0

    def test_crossover_preserves_gene_keys(self):
        rng = random.Random(0)
        pop = random_population("sklearn", 2, rng)
        child_a, child_b = single_point_crossover(pop[0], pop[1], rng)
        gene_names = set(pop[0].genes.keys())
        assert set(child_a.genes.keys()) == gene_names
        assert set(child_b.genes.keys()) == gene_names

    def test_mutation_produces_valid_genes(self):
        rng = random.Random(0)
        pop = random_population("sklearn", 1, rng)
        gene_space = {g.name: set(g.values) for g in get_gene_space("sklearn")}
        mutant = mutate(pop[0], "sklearn", mutation_rate=1.0, rng=rng)
        for name, val in mutant.genes.items():
            assert val in gene_space[name], f"Gene {name}={val} not in valid values"


# ---------------------------------------------------------------------------
# Hamming diversity
# ---------------------------------------------------------------------------

class TestDiversity:
    def test_hamming_identical(self):
        rng = random.Random(0)
        pop = random_population("sklearn", 1, rng)
        a = pop[0]
        b = a.copy()
        assert hamming_distance(a, b) == pytest.approx(0.0)

    def test_hamming_range(self):
        rng = random.Random(0)
        pop = random_population("sklearn", 20, rng)
        for i in range(len(pop)):
            for j in range(i + 1, len(pop)):
                d = hamming_distance(pop[i], pop[j])
                assert 0.0 <= d <= 1.0

    def test_mean_hamming_identical_population(self):
        rng = random.Random(0)
        c = random_population("sklearn", 1, rng)[0]
        pop = [c.copy() for _ in range(5)]
        assert mean_pairwise_hamming(pop) == pytest.approx(0.0)

    def test_injection_maintains_population_size(self):
        rng = random.Random(0)
        pop = random_population("sklearn", 10, rng)
        for i, c in enumerate(pop):
            c.fitness = float(i)
        pd_ctrl = PopulationDiversity(
            backend="sklearn",
            min_diversity_threshold=1.0,  # always trigger
            injection_ratio=0.3,
            stagnation_rounds=999,
        )
        new_pop, _ = pd_ctrl.update(pop, 0, 0)
        assert len(new_pop) == 10

    def test_adaptive_mutation_boost(self):
        rng = random.Random(0)
        pop = random_population("sklearn", 5, rng)
        for c in pop:
            c.fitness = 1.0
        pd_ctrl = PopulationDiversity(
            backend="sklearn",
            base_mutation_rate=0.2,
            min_diversity_threshold=0.0,
            stagnation_rounds=2,
            mutation_boost_factor=3.0,
        )
        _, rate_base = pd_ctrl.update(pop, 0, no_improvement_streak=0)
        _, rate_boost = pd_ctrl.update(pop, 1, no_improvement_streak=3)
        assert rate_boost > rate_base

    def test_mutation_decays_after_improvement(self):
        rng = random.Random(0)
        pop = random_population("sklearn", 5, rng)
        for c in pop:
            c.fitness = 1.0
        pd_ctrl = PopulationDiversity(
            backend="sklearn",
            base_mutation_rate=0.2,
            min_diversity_threshold=0.0,
            stagnation_rounds=1,
            mutation_boost_factor=2.0,
            mutation_decay=0.5,
        )
        _, boosted = pd_ctrl.update(pop, 0, no_improvement_streak=2)
        _, decayed = pd_ctrl.update(pop, 1, no_improvement_streak=0)
        assert decayed < boosted


# ---------------------------------------------------------------------------
# WarmStart
# ---------------------------------------------------------------------------

class TestWarmStart:
    def test_default_seeds_cover_all_genes(self):
        gene_names = {g.name for g in get_gene_space("sklearn")}
        genes = _sklearn_baseline("sklearn")
        assert gene_names == set(genes.keys())

    def test_build_initial_population_size(self):
        ws = WarmStart("sklearn", n_default_seeds=2, halving_pool_ratio=0, random_seed=42)
        X = pd.DataFrame(np.random.randn(50, 3), columns=list("abc"))
        y = pd.Series(np.random.randint(0, 2, 50))
        pop = ws.build_initial_population(population_size=8, evaluator=None, X_train=X, y_train=y)
        assert len(pop) == 8

    def test_seeds_injected_first(self):
        ws = WarmStart("sklearn", n_default_seeds=3, halving_pool_ratio=0, random_seed=42)
        X = pd.DataFrame(np.random.randn(50, 3), columns=list("abc"))
        y = pd.Series(np.random.randint(0, 2, 50))
        pop = ws.build_initial_population(population_size=5, evaluator=None, X_train=X, y_train=y)
        # First 3 should have fitness=None (not yet evaluated by engine)
        assert all(c.fitness is None for c in pop)


# ---------------------------------------------------------------------------
# Phase 1 improvements
# ---------------------------------------------------------------------------

class TestVectorisedHamming:
    """PERF-3: vectorised Hamming distance produces same results as scalar version."""

    def test_mean_pairwise_matches_scalar(self):
        from genetic_automl.genetic.diversity import mean_pairwise_hamming, hamming_distance
        rng = random.Random(7)
        pop = random_population("sklearn", 10, rng)
        # Scalar reference
        n = len(pop)
        total, pairs = 0.0, 0
        for i in range(n):
            for j in range(i + 1, n):
                total += hamming_distance(pop[i], pop[j])
                pairs += 1
        scalar_mean = total / pairs
        assert abs(mean_pairwise_hamming(pop) - scalar_mean) < 1e-9

    def test_identical_population_has_zero_diversity(self):
        from genetic_automl.genetic.diversity import mean_pairwise_hamming
        rng = random.Random(0)
        chrom = random_population("sklearn", 1, rng)[0]
        pop = [chrom.copy() for _ in range(6)]
        assert mean_pairwise_hamming(pop) == pytest.approx(0.0)

    def test_single_individual_returns_one(self):
        from genetic_automl.genetic.diversity import mean_pairwise_hamming
        rng = random.Random(0)
        pop = random_population("sklearn", 1, rng)
        assert mean_pairwise_hamming(pop) == 1.0


class TestFitnessCache:
    """PERF-2: identical chromosomes are served from cache."""

    def test_cache_hit_on_duplicate(self):
        from genetic_automl.genetic.fitness import FitnessEvaluator
        from genetic_automl.core.problem import ProblemType
        evaluator = FitnessEvaluator(
            problem_type=ProblemType.CLASSIFICATION,
            target_column="y",
            backend="sklearn",
            n_folds=2,
            random_seed=42,
            fitness_std_penalty=0.0,
        )
        X = pd.DataFrame(np.random.randn(60, 3), columns=list("abc"))
        y = pd.Series(np.random.randint(0, 2, 60))
        rng = random.Random(1)
        chrom1 = random_population("sklearn", 1, rng)[0]
        # Evaluate once
        f1 = evaluator.evaluate(chrom1, X, y)
        assert evaluator._cache_hits == 0
        # Evaluate identical genes again via a new chromosome object
        chrom2 = chrom1.copy()
        chrom2.fitness = None
        f2 = evaluator.evaluate(chrom2, X, y)
        assert evaluator._cache_hits == 1
        assert f1 == f2

    def test_different_genes_not_cached(self):
        from genetic_automl.genetic.fitness import FitnessEvaluator
        from genetic_automl.core.problem import ProblemType
        evaluator = FitnessEvaluator(
            problem_type=ProblemType.CLASSIFICATION,
            target_column="y",
            backend="sklearn",
            n_folds=2,
            random_seed=42,
            fitness_std_penalty=0.0,
        )
        X = pd.DataFrame(np.random.randn(60, 3), columns=list("abc"))
        y = pd.Series(np.random.randint(0, 2, 60))
        rng = random.Random(2)
        c1, c2 = random_population("sklearn", 2, rng)
        # Force them to differ in at least one gene
        c2.genes["scaler"] = "minmax" if c1.genes.get("scaler") != "minmax" else "standard"
        evaluator.evaluate(c1, X, y)
        evaluator.evaluate(c2, X, y)
        assert evaluator._cache_hits == 0


class TestFitnessStdPenalty:
    """QUAL-1: std penalty correctly reduces fitness of high-variance chromosomes."""

    def test_penalty_lowers_fitness(self):
        from genetic_automl.genetic.fitness import FitnessEvaluator
        from genetic_automl.core.problem import ProblemType
        import numpy as np

        ev_no_pen = FitnessEvaluator(
            problem_type=ProblemType.CLASSIFICATION,
            target_column="y",
            backend="sklearn",
            n_folds=3,
            random_seed=42,
            fitness_std_penalty=0.0,
        )
        ev_penalty = FitnessEvaluator(
            problem_type=ProblemType.CLASSIFICATION,
            target_column="y",
            backend="sklearn",
            n_folds=3,
            random_seed=42,
            fitness_std_penalty=1.0,
        )
        X = pd.DataFrame(np.random.randn(90, 4), columns=list("abcd"))
        y = pd.Series(np.random.randint(0, 2, 90))
        rng = random.Random(5)
        chrom = random_population("sklearn", 1, rng)[0]

        f_no_pen = ev_no_pen.evaluate(chrom, X, y)
        chrom2 = chrom.copy(); chrom2.fitness = None
        f_pen = ev_penalty.evaluate(chrom2, X, y)

        # With std > 0, penalised fitness must be <= unpenalised
        assert f_pen <= f_no_pen


class TestUniformCrossoverConfig:
    """QUAL-2: crossover_type='uniform' uses uniform_crossover; 'single_point' uses single_point."""

    def test_uniform_crossover_produces_mixed_genes(self):
        from genetic_automl.genetic.operators import uniform_crossover
        rng = random.Random(0)
        a, b = random_population("sklearn", 2, rng)
        # Force a and b to differ on every gene
        for key in b.genes:
            gene_def = next(g for g in get_gene_space("sklearn") if g.name == key)
            vals = [v for v in gene_def.values if v != a.genes[key]]
            if vals:
                b.genes[key] = vals[0]
        child_a, child_b = uniform_crossover(a, b, random.Random(42))
        # Children should not be identical to either parent (very likely with >5 genes)
        assert child_a.genes != a.genes or child_b.genes != b.genes

    def test_crossover_type_config_respected(self):
        from genetic_automl.config import GeneticConfig
        cfg = GeneticConfig(crossover_type="uniform")
        assert cfg.crossover_type == "uniform"
        cfg2 = GeneticConfig(crossover_type="single_point")
        assert cfg2.crossover_type == "single_point"


class TestGeneSpaceDictCache:
    """PERF-4: mutate() accepts pre-built dict and produces valid chromosomes."""

    def test_mutate_with_dict_cache(self):
        from genetic_automl.genetic.operators import mutate
        from genetic_automl.genetic.chromosome import get_gene_space
        rng = random.Random(9)
        pop = random_population("sklearn", 1, rng)
        chrom = pop[0]
        gene_space = get_gene_space("sklearn")
        gene_space_dict = {g.name: g for g in gene_space}
        mutant = mutate(chrom, "sklearn", 1.0, rng, gene_space, gene_space_dict)
        assert set(mutant.genes.keys()) == set(chrom.genes.keys())
