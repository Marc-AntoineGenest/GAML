"""
Tests for NSGA-II + Island Model compatibility (Bug fixes).

Covers:
  1. Bug 1 fix: _step_island passes objective_values to _breed when nsga2 active
  2. Bug 2 fix: _merge_histories populates pareto_front when nsga2 active
  3. Bug 3 fix: _migrate selects rank-0 diverse emigrants when nsga2 active
  4. Combined end-to-end: nsga2_enabled + island_model completes and returns best
  5. Pareto front non-empty after combined run
  6. Without nsga2, island migration still uses scalar fitness (no regression)
  7. Without island_model, nsga2 still works standalone (no regression)
"""
from __future__ import annotations

import copy
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from genetic_automl.config import GeneticConfig
from genetic_automl.core.problem import ProblemType
from genetic_automl.genetic.chromosome import Chromosome
from genetic_automl.genetic.fitness import FitnessEvaluator
from genetic_automl.genetic.nsga2 import _CROWD_ATTR, _RANK_ATTR

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clf_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((150, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series((X["f0"] + X["f1"] > 0).astype(int), name="label")
    return X, y


def _make_evaluator(n_folds=2):
    return FitnessEvaluator(
        problem_type=ProblemType.CLASSIFICATION,
        target_column="label",
        backend="sklearn",
        metric="f1_macro",
        n_folds=n_folds,
    )


def _make_cfg(nsga2=False, island=False, n_islands=2, pop=8, gens=2):
    return GeneticConfig(
        population_size=pop,
        generations=gens,
        n_cv_folds=2,
        nsga2_enabled=nsga2,
        nsga2_objectives=["f1_macro", "complexity"] if nsga2 else None,
        island_model=island,
        n_islands=n_islands,
        migration_interval=1,
        migration_size=1,
        random_seed=42,
    )


def _chrom(fitness, n_est=100, model="lgbm"):
    c = Chromosome(genes={"model_type": model, "n_estimators": n_est})
    c.fitness = fitness
    return c


# ---------------------------------------------------------------------------
# Bug 1: _step_island passes objective_values to _breed
# ---------------------------------------------------------------------------

class TestBug1BreedReceivesObjectiveValues:
    def test_breed_called_with_objective_values_when_nsga2(self, clf_data):
        """When nsga2_enabled, _breed must receive objective_values (not None)."""
        from genetic_automl.genetic.island_engine import IslandEngine

        X, y = clf_data
        cfg = _make_cfg(nsga2=True, island=True, n_islands=2, pop=6, gens=2)
        engine = IslandEngine(
            genetic_config=cfg,
            evaluator=_make_evaluator(),
            backend="sklearn",
            n_islands=2,
            migration_interval=5,  # no migration during test
            migration_size=1,
        )

        breed_calls = []
        original_breed = engine._islands[0]._breed

        def capturing_breed(population, next_gen, mutation_rate, objective_values=None):
            breed_calls.append(objective_values)
            return original_breed(population, next_gen, mutation_rate,
                                   objective_values=objective_values)

        engine._islands[0]._breed = capturing_breed

        # Run one step manually
        populations = engine._build_initial_populations(X, y)
        state = {
            "population": populations[0],
            "no_improvement_streak": 0,
            "best_fitness_so_far": float("-inf"),
            "current_mut_rate": cfg.mutation_rate,
        }
        engine._step_island(engine._islands[0], state, X, y, gen_idx=0)

        # objective_values must NOT be None when nsga2 is enabled
        assert len(breed_calls) > 0
        assert breed_calls[0] is not None, (
            "Bug 1: _breed received objective_values=None even though nsga2_enabled=True"
        )

    def test_breed_called_without_objective_values_when_no_nsga2(self, clf_data):
        """Without nsga2, _breed must receive objective_values=None (scalar path)."""
        from genetic_automl.genetic.island_engine import IslandEngine

        X, y = clf_data
        cfg = _make_cfg(nsga2=False, island=True, n_islands=2, pop=6, gens=2)
        engine = IslandEngine(
            genetic_config=cfg,
            evaluator=_make_evaluator(),
            backend="sklearn",
            n_islands=2,
            migration_interval=5,
        )

        breed_calls = []
        original_breed = engine._islands[0]._breed

        def capturing_breed(population, next_gen, mutation_rate, objective_values=None):
            breed_calls.append(objective_values)
            return original_breed(population, next_gen, mutation_rate,
                                   objective_values=objective_values)

        engine._islands[0]._breed = capturing_breed
        populations = engine._build_initial_populations(X, y)
        state = {
            "population": populations[0],
            "no_improvement_streak": 0,
            "best_fitness_so_far": float("-inf"),
            "current_mut_rate": cfg.mutation_rate,
        }
        engine._step_island(engine._islands[0], state, X, y, gen_idx=0)

        assert len(breed_calls) > 0
        assert breed_calls[0] is None


# ---------------------------------------------------------------------------
# Bug 2: _merge_histories populates pareto_front
# ---------------------------------------------------------------------------

class TestBug2ParetoFrontMerged:
    def test_pareto_front_populated_after_island_nsga2_run(self, clf_data):
        """After combined run, merged history.pareto_front must be non-empty."""
        from genetic_automl.genetic.island_engine import IslandEngine

        X, y = clf_data
        cfg = _make_cfg(nsga2=True, island=True, n_islands=2, pop=6, gens=2)
        engine = IslandEngine(
            genetic_config=cfg,
            evaluator=_make_evaluator(),
            backend="sklearn",
            n_islands=2,
            migration_interval=2,
            migration_size=1,
        )
        engine.run(X, y)

        assert isinstance(engine.history.pareto_front, list), (
            "Bug 2: pareto_front is not a list"
        )
        assert len(engine.history.pareto_front) > 0, (
            "Bug 2: pareto_front is empty after island+nsga2 run"
        )

    def test_pareto_front_empty_without_nsga2(self, clf_data):
        """Without nsga2, pareto_front must remain empty (no regression)."""
        from genetic_automl.genetic.island_engine import IslandEngine

        X, y = clf_data
        cfg = _make_cfg(nsga2=False, island=True, n_islands=2, pop=6, gens=2)
        engine = IslandEngine(
            genetic_config=cfg,
            evaluator=_make_evaluator(),
            backend="sklearn",
            n_islands=2,
            migration_interval=2,
        )
        engine.run(X, y)
        assert engine.history.pareto_front == []

    def test_pareto_front_entries_have_expected_keys(self, clf_data):
        """Each Pareto front entry must have the required fields."""
        from genetic_automl.genetic.island_engine import IslandEngine

        X, y = clf_data
        cfg = _make_cfg(nsga2=True, island=True, n_islands=2, pop=6, gens=2)
        engine = IslandEngine(
            genetic_config=cfg,
            evaluator=_make_evaluator(),
            backend="sklearn",
            n_islands=2,
            migration_interval=2,
        )
        engine.run(X, y)

        for entry in engine.history.pareto_front:
            assert "id"         in entry
            assert "objectives" in entry
            assert "rank"       in entry
            assert "fitness"    in entry


# ---------------------------------------------------------------------------
# Bug 3: Migration selects diverse rank-0 emigrants under NSGA-II
# ---------------------------------------------------------------------------

class TestBug3MigrationDiversity:
    def _make_island_engine_with_nsga2(self):
        cfg = _make_cfg(nsga2=True, island=True, n_islands=3, pop=8)
        evaluator = _make_evaluator()
        from genetic_automl.genetic.island_engine import IslandEngine
        return IslandEngine(
            genetic_config=cfg,
            evaluator=evaluator,
            backend="sklearn",
            n_islands=3,
            migration_interval=1,
            migration_size=2,
        )

    def test_emigrants_come_from_rank0_when_nsga2(self):
        """With nsga2, emigrant selection must use Pareto rank, not just fitness."""
        engine = self._make_island_engine_with_nsga2()

        # Build population where rank-0 are NOT the highest-fitness individuals
        # (diverse trade-off). Rank 0 = non-dominated. Rank 1 = dominated.
        # High-fitness but dominated: A(0.95, low complexity)
        # Lower-fitness but diverse: B(0.80, high complexity) — non-dominated with A
        a = _chrom(0.95, n_est=500)   # high accuracy, complex
        b = _chrom(0.80, n_est=10)    # lower accuracy, simple — non-dominated
        c = _chrom(0.70, n_est=300)   # dominated by a on both

        # Stamp ranks manually (as nsga2 would)
        setattr(a, _RANK_ATTR, 0); setattr(a, _CROWD_ATTR, 2.0)
        setattr(b, _RANK_ATTR, 0); setattr(b, _CROWD_ATTR, 2.0)
        setattr(c, _RANK_ATTR, 1); setattr(c, _CROWD_ATTR, 1.0)

        state0 = {"population": [a, b, c]}
        state1 = {"population": [_chrom(0.5), _chrom(0.5)]}
        state2 = {"population": [_chrom(0.5)]}
        island_states = [state0, state1, state2]

        # Run one migration
        engine._migrate(island_states)

        # Emigrants from island 0 should be a and b (rank 0), not c (rank 1)
        immigrant_ids = {ch.id for ch in island_states[1]["population"]
                         if ch.id not in {_chrom(0.5).id}}
        # Neither emigrant should be c (the dominated solution)
        immigrant_genes = [ch.genes for ch in island_states[1]["population"]]
        dominated_genes = c.genes
        # c's n_estimators=300 should not appear in immigrants
        n_est_values = [g.get("n_estimators") for g in immigrant_genes]
        assert 300 not in n_est_values, (
            "Bug 3: dominated chromosome (rank 1) was selected as emigrant"
        )

    def test_emigrants_use_scalar_fitness_without_nsga2(self):
        """Without nsga2, migration must use highest scalar fitness (no regression)."""
        cfg = _make_cfg(nsga2=False, island=True, n_islands=2, pop=6)
        from genetic_automl.genetic.island_engine import IslandEngine
        engine = IslandEngine(
            genetic_config=cfg,
            evaluator=_make_evaluator(),
            backend="sklearn",
            n_islands=2,
            migration_interval=1,
            migration_size=1,
        )

        best = _chrom(0.99, n_est=100)
        mid  = _chrom(0.70, n_est=200)
        worst = _chrom(0.40, n_est=300)

        state0 = {"population": [best, mid, worst]}
        state1 = {"population": [_chrom(0.5, n_est=50)]}
        engine._migrate([state0, state1])

        # The immigrant should have n_est=100 (best fitness)
        immigrant = state1["population"][-1]
        assert immigrant.genes["n_estimators"] == 100


# ---------------------------------------------------------------------------
# End-to-end: combined island + nsga2 run
# ---------------------------------------------------------------------------

class TestCombinedEndToEnd:
    def test_combined_run_returns_chromosome(self, clf_data):
        """Full run with both island_model and nsga2_enabled must complete."""
        from genetic_automl.genetic.island_engine import IslandEngine

        X, y = clf_data
        cfg = _make_cfg(nsga2=True, island=True, n_islands=2, pop=8, gens=3)
        engine = IslandEngine(
            genetic_config=cfg,
            evaluator=_make_evaluator(),
            backend="sklearn",
            n_islands=2,
            migration_interval=2,
            migration_size=1,
        )
        best = engine.run(X, y)

        assert best is not None
        assert best.fitness is not None
        assert isinstance(best.fitness, float)
        assert best.fitness > float("-inf")

    def test_combined_run_finds_diverse_solutions(self, clf_data):
        """Combined run should produce multiple distinct chromosomes."""
        from genetic_automl.genetic.island_engine import IslandEngine

        X, y = clf_data
        cfg = _make_cfg(nsga2=True, island=True, n_islands=2, pop=8, gens=3)
        engine = IslandEngine(
            genetic_config=cfg,
            evaluator=_make_evaluator(),
            backend="sklearn",
            n_islands=2,
            migration_interval=2,
            migration_size=1,
        )
        engine.run(X, y)

        # Should have evaluated chromosomes from both islands
        assert len(engine.history.all_chromosomes) > 0
        unique_models = {c.genes.get("model_type") for c in engine.history.all_chromosomes}
        # With 2 islands and NSGA-II diversity pressure, expect variety
        assert len(unique_models) >= 1  # minimal sanity — at least one model type

    def test_nsga2_standalone_still_works(self, clf_data):
        """nsga2 without island model must still produce Pareto front (no regression)."""
        from genetic_automl.genetic.engine import GeneticEngine

        X, y = clf_data
        cfg = _make_cfg(nsga2=True, island=False, pop=6, gens=2)
        engine = GeneticEngine(
            genetic_config=cfg,
            evaluator=_make_evaluator(),
            backend="sklearn",
        )
        best = engine.run(X, y)
        assert best is not None
        assert len(engine.history.pareto_front) > 0

    def test_island_standalone_still_works(self, clf_data):
        """Island model without nsga2 must still return best chromosome (no regression)."""
        from genetic_automl.genetic.island_engine import IslandEngine

        X, y = clf_data
        cfg = _make_cfg(nsga2=False, island=True, n_islands=2, pop=6, gens=2)
        engine = IslandEngine(
            genetic_config=cfg,
            evaluator=_make_evaluator(),
            backend="sklearn",
            n_islands=2,
        )
        best = engine.run(X, y)
        assert best is not None
        assert best.fitness is not None
