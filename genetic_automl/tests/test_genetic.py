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
