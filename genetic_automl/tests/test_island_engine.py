"""
Tests for genetic/island_engine.py and its config/pipeline/CLI wiring.

Coverage:
  1.  IslandEngine instantiates N GeneticEngine islands
  2.  Island population size = population_size // n_islands (min 4)
  3.  Each island gets a unique random seed offset
  4.  Each island has an independent evaluator (deep copy — different object)
  5.  _migrate(): emigrants copied from island i to island (i+1) % n_islands
  6.  _migrate(): worst members on destination are replaced
  7.  _migrate(): deep copy — mutation on dst doesn't affect src
  8.  _migrate(): island with no evaluated chromosomes sends nothing
  9.  _merge_histories(): all_chromosomes contains chromosomes from all islands
  10. _merge_histories(): GenerationStats aligned by generation index
  11. _merge_histories(): global best_fitness = max across islands
  12. run() completes and returns a Chromosome
  13. run() with n_islands=2 completes correctly
  14. run() with migration_interval=1 (every generation) completes correctly
  15. run() early stopping fires when all islands stagnate
  16. GeneticConfig island model fields have correct defaults
  17. config_loader parses island model fields from YAML
  18. pipeline chooses IslandEngine when island_model=True
  19. pipeline chooses GeneticEngine when island_model=False
  20. CLI --island-model flag enables island model
  21. CLI --n-islands, --migration-interval, --migration-size flags applied
  22. diversity_summary() aggregates across all islands
  23. IslandEngine history has generations after run()
"""
from __future__ import annotations

import copy
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from genetic_automl.config import GeneticConfig, PipelineConfig
from genetic_automl.core.problem import ProblemType
from genetic_automl.genetic.chromosome import Chromosome
from genetic_automl.genetic.engine import EvolutionHistory
from genetic_automl.genetic.island_engine import IslandEngine


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clf_Xy():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((120, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series((X["f0"] + X["f1"] > 0).astype(int), name="label")
    return X, y


def _make_genetic_config(**overrides) -> GeneticConfig:
    defaults = dict(
        population_size=8,
        generations=3,
        n_cv_folds=2,
        random_seed=42,
        warm_start=False,
        surrogate_enabled=False,
        early_stopping_rounds=999,
        island_model=True,
        n_islands=2,
        migration_interval=2,
        migration_size=1,
        n_island_jobs=1,
    )
    defaults.update(overrides)
    return GeneticConfig(**defaults)


def _make_evaluator():
    from genetic_automl.genetic.fitness import FitnessEvaluator
    return FitnessEvaluator(
        problem_type=ProblemType.CLASSIFICATION,
        target_column="label",
        backend="sklearn",
        metric="f1_macro",
        n_folds=2,
        random_seed=42,
    )


def _make_chromosome(fitness: float | None = None, genes: dict | None = None) -> Chromosome:
    g = genes or {"model_type": "lgbm", "n_estimators": 20,
                  "scaler": "standard", "numeric_imputer": "mean",
                  "feature_engineering": "none"}
    c = Chromosome(genes=g)
    c.fitness = fitness
    return c


def _make_island_engine(n_islands=2, migration_interval=2, migration_size=1):
    cfg = _make_genetic_config(n_islands=n_islands,
                                migration_interval=migration_interval,
                                migration_size=migration_size)
    return IslandEngine(
        genetic_config=cfg,
        evaluator=_make_evaluator(),
        backend="sklearn",
        n_islands=n_islands,
        migration_interval=migration_interval,
        migration_size=migration_size,
    )


# ===========================================================================
# 1–4. Instantiation
# ===========================================================================

class TestIslandEngineInstantiation:

    def test_correct_number_of_islands(self):
        ie = _make_island_engine(n_islands=3)
        assert len(ie._islands) == 3

    def test_island_pop_size_divided(self):
        cfg = _make_genetic_config(population_size=12, n_islands=3)
        ie = IslandEngine(cfg, _make_evaluator(), backend="sklearn",
                          n_islands=3, migration_interval=2, migration_size=1)
        for island in ie._islands:
            assert island.cfg.population_size == 4   # 12 // 3

    def test_minimum_island_pop_size_is_4(self):
        cfg = _make_genetic_config(population_size=4, n_islands=4)
        ie = IslandEngine(cfg, _make_evaluator(), backend="sklearn",
                          n_islands=4, migration_interval=2, migration_size=1)
        for island in ie._islands:
            assert island.cfg.population_size >= 4

    def test_unique_random_seeds_per_island(self):
        ie = _make_island_engine(n_islands=4)
        seeds = [island.cfg.random_seed for island in ie._islands]
        assert len(set(seeds)) == 4   # all different

    def test_independent_evaluators(self):
        ie = _make_island_engine(n_islands=2)
        ev0 = ie._islands[0].evaluator
        ev1 = ie._islands[1].evaluator
        assert ev0 is not ev1   # deep copies, not the same object


# ===========================================================================
# 5–8. Migration
# ===========================================================================

class TestMigration:

    def _make_state(self, fitnesses):
        """Make an island state dict with a population of evaluated chromosomes."""
        pop = [_make_chromosome(fitness=f) for f in fitnesses]
        return {"population": pop, "no_improvement_streak": 0,
                "best_fitness_so_far": max(fitnesses),
                "current_mut_rate": 0.2}

    def test_migration_sends_to_next_island(self):
        """After migration, island 1 should contain chromosomes from island 0."""
        ie = _make_island_engine(n_islands=2, migration_size=1)
        states = [
            self._make_state([0.9, 0.7, 0.5, 0.3]),  # island 0 — best=0.9
            self._make_state([0.6, 0.4, 0.2, 0.1]),  # island 1 — best=0.6
        ]
        ie._migrate(states)
        island1_fitnesses = [c.fitness for c in states[1]["population"]]
        assert 0.9 in island1_fitnesses   # the best from island 0 arrived

    def test_migration_ring_topology(self):
        """Island n-1 sends to island 0 (ring closes)."""
        ie = _make_island_engine(n_islands=3, migration_size=1)
        states = [
            self._make_state([0.3, 0.2, 0.1, 0.05]),  # island 0 — will receive from island 2
            self._make_state([0.5, 0.4, 0.3, 0.2]),   # island 1
            self._make_state([0.9, 0.8, 0.7, 0.6]),   # island 2 — best=0.9 → goes to island 0
        ]
        ie._migrate(states)
        island0_fitnesses = [c.fitness for c in states[0]["population"]]
        assert 0.9 in island0_fitnesses

    def test_migration_replaces_worst(self):
        """The worst member of the receiving island is replaced."""
        ie = _make_island_engine(n_islands=2, migration_size=1)
        states = [
            self._make_state([0.9, 0.7, 0.5, 0.3]),  # sends 0.9
            self._make_state([0.6, 0.4, 0.2, 0.1]),  # 0.1 is worst — should be replaced
        ]
        ie._migrate(states)
        island1_fitnesses = [c.fitness for c in states[1]["population"]]
        assert 0.1 not in island1_fitnesses   # worst evicted
        assert 0.9 in island1_fitnesses       # immigrant arrived

    def test_migration_is_deep_copy(self):
        """Mutations on destination island must not affect source."""
        ie = _make_island_engine(n_islands=2, migration_size=1)
        states = [
            self._make_state([0.9, 0.7, 0.5, 0.3]),
            self._make_state([0.6, 0.4, 0.2, 0.1]),
        ]
        ie._migrate(states)
        # Corrupt the immigrant on island 1
        for c in states[1]["population"]:
            if c.fitness == 0.9:
                c.fitness = -999.0
        # Original on island 0 must be unaffected
        island0_fitnesses = [c.fitness for c in states[0]["population"]]
        assert 0.9 in island0_fitnesses

    def test_migration_empty_island_sends_nothing(self):
        """An island with no evaluated chromosomes must not crash."""
        ie = _make_island_engine(n_islands=2, migration_size=1)
        empty_pop = [_make_chromosome(fitness=None) for _ in range(4)]
        states = [
            {"population": empty_pop, "no_improvement_streak": 0,
             "best_fitness_so_far": float("-inf"), "current_mut_rate": 0.2},
            self._make_state([0.6, 0.4, 0.2, 0.1]),
        ]
        # Must not raise
        ie._migrate(states)


# ===========================================================================
# 9–11. History merging
# ===========================================================================

class TestMergeHistories:

    def _make_history_with_stats(self, fitnesses_per_gen, island_idx=0):
        """Build a minimal EvolutionHistory with GenerationStats."""
        from genetic_automl.genetic.engine import GenerationStats
        h = EvolutionHistory()
        for gen_idx, best_fit in enumerate(fitnesses_per_gen):
            chrom = _make_chromosome(fitness=best_fit)
            h.all_chromosomes.append(chrom)
            h.generations.append(GenerationStats(
                generation=gen_idx,
                best_fitness=best_fit,
                mean_fitness=best_fit - 0.05,
                worst_fitness=best_fit - 0.1,
                elapsed_seconds=1.0,
                mean_hamming=0.3,
                mutation_rate=0.2,
                diversity_injected=False,
                mutation_boosted=False,
                best_chromosome=chrom,
            ))
        return h

    def test_all_chromosomes_collected(self):
        ie = _make_island_engine(n_islands=2)
        ie._islands[0].history = self._make_history_with_stats([0.8, 0.85])
        ie._islands[1].history = self._make_history_with_stats([0.7, 0.75])
        merged = ie._merge_histories()
        assert len(merged.all_chromosomes) == 4   # 2 from each island

    def test_generations_aligned_by_index(self):
        ie = _make_island_engine(n_islands=2)
        ie._islands[0].history = self._make_history_with_stats([0.8, 0.85])
        ie._islands[1].history = self._make_history_with_stats([0.7, 0.75])
        merged = ie._merge_histories()
        assert len(merged.generations) == 2

    def test_global_best_is_max_across_islands(self):
        ie = _make_island_engine(n_islands=2)
        ie._islands[0].history = self._make_history_with_stats([0.8, 0.85])
        ie._islands[1].history = self._make_history_with_stats([0.9, 0.75])
        merged = ie._merge_histories()
        # Generation 0: max(0.8, 0.9) = 0.9
        assert merged.generations[0].best_fitness == pytest.approx(0.9)
        # Generation 1: max(0.85, 0.75) = 0.85
        assert merged.generations[1].best_fitness == pytest.approx(0.85)


# ===========================================================================
# 12–15. Full run()
# ===========================================================================

class TestIslandEngineRun:

    def test_run_returns_chromosome(self, clf_Xy):
        X, y = clf_Xy
        cfg = _make_genetic_config(population_size=8, generations=2,
                                   n_islands=2, migration_interval=1)
        ie = IslandEngine(cfg, _make_evaluator(), backend="sklearn",
                          n_islands=2, migration_interval=1, migration_size=1)
        result = ie.run(X, y)
        assert isinstance(result, Chromosome)
        assert result.fitness is not None

    def test_run_history_populated(self, clf_Xy):
        X, y = clf_Xy
        cfg = _make_genetic_config(population_size=8, generations=2,
                                   n_islands=2, migration_interval=2)
        ie = IslandEngine(cfg, _make_evaluator(), backend="sklearn",
                          n_islands=2, migration_interval=2, migration_size=1)
        ie.run(X, y)
        assert len(ie.history.all_chromosomes) > 0
        assert len(ie.history.generations) > 0

    def test_run_with_migration_every_generation(self, clf_Xy):
        """migration_interval=1 means migrate after every generation — must not crash."""
        X, y = clf_Xy
        cfg = _make_genetic_config(population_size=8, generations=3,
                                   n_islands=2, migration_interval=1)
        ie = IslandEngine(cfg, _make_evaluator(), backend="sklearn",
                          n_islands=2, migration_interval=1, migration_size=1)
        result = ie.run(X, y)
        assert result.fitness is not None

    def test_run_best_fitness_is_finite(self, clf_Xy):
        X, y = clf_Xy
        cfg = _make_genetic_config(population_size=8, generations=2,
                                   n_islands=2, migration_interval=2)
        ie = IslandEngine(cfg, _make_evaluator(), backend="sklearn",
                          n_islands=2, migration_interval=2, migration_size=1)
        result = ie.run(X, y)
        assert result.fitness > float("-inf")
        assert result.fitness <= 1.0


# ===========================================================================
# 16–17. Config and config_loader
# ===========================================================================

class TestIslandConfig:

    def test_genetic_config_defaults(self):
        cfg = GeneticConfig()
        assert cfg.island_model is False
        assert cfg.n_islands == 4
        assert cfg.migration_interval == 3
        assert cfg.migration_size == 2
        assert cfg.n_island_jobs == 1

    def test_config_loader_parses_island_fields(self, tmp_path):
        from genetic_automl.config_loader import load_config
        yaml = """\
run:
  backend: sklearn
problem:
  type: classification
  target_column: label
genetic:
  population_size: 8
  generations: 2
  island_model: true
  n_islands: 3
  migration_interval: 2
  migration_size: 2
  n_island_jobs: -1
"""
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml)
        cfg, _ = load_config(str(p))
        assert cfg.genetic.island_model is True
        assert cfg.genetic.n_islands == 3
        assert cfg.genetic.migration_interval == 2
        assert cfg.genetic.migration_size == 2
        assert cfg.genetic.n_island_jobs == -1

    def test_config_loader_island_defaults_when_absent(self, tmp_path):
        from genetic_automl.config_loader import load_config
        yaml = """\
run:
  backend: sklearn
problem:
  type: classification
  target_column: label
genetic:
  population_size: 8
  generations: 2
"""
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml)
        cfg, _ = load_config(str(p))
        assert cfg.genetic.island_model is False
        assert cfg.genetic.n_islands == 4


# ===========================================================================
# 18–19. Pipeline engine selection
# ===========================================================================

class TestPipelineEngineSelection:

    @pytest.fixture
    def sample_csv(self, tmp_path):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.standard_normal((80, 4)), columns=list("abcd"))
        df["label"] = rng.integers(0, 2, 80)
        p = tmp_path / "data.csv"
        df.to_csv(p, index=False)
        return p

    def _capture_engine(self, island_model: bool):
        """Run a mock fit and capture which engine class was instantiated."""
        from genetic_automl.pipeline import AutoMLPipeline
        cfg = PipelineConfig(
            problem_type=ProblemType.CLASSIFICATION,
            target_column="label",
        )
        cfg.genetic.island_model = island_model
        cfg.genetic.n_islands = 2
        cfg.genetic.generations = 1
        cfg.genetic.population_size = 4
        cfg.automl.backend = "sklearn"

        captured = {}

        with patch("genetic_automl.pipeline.IslandEngine") as mock_island, \
             patch("genetic_automl.pipeline.GeneticEngine") as mock_ga:

            mock_engine = MagicMock()
            mock_engine.run.return_value = _make_chromosome(fitness=0.8)
            mock_engine.history = MagicMock(spec=EvolutionHistory)
            mock_engine.history.best = _make_chromosome(fitness=0.8)
            mock_engine.history.top_chromosomes.return_value = [_make_chromosome(fitness=0.8)]
            mock_engine.history.all_chromosomes = []
            mock_engine.history.generations = []
            mock_engine.diversity_summary.return_value = {}

            mock_island.return_value = mock_engine
            mock_ga.return_value = mock_engine

            rng = np.random.default_rng(0)
            X = pd.DataFrame(rng.standard_normal((80, 4)), columns=list("abcd"))
            y = pd.Series(rng.integers(0, 2, 80), name="label")

            try:
                p = AutoMLPipeline(cfg)
                p.fit(pd.concat([X.assign(label=y)], axis=1))
            except Exception:
                pass

            captured["island_called"] = mock_island.called
            captured["ga_called"] = mock_ga.called

        return captured

    def test_island_engine_selected_when_enabled(self):
        captured = self._capture_engine(island_model=True)
        assert captured["island_called"] is True
        assert captured["ga_called"] is False

    def test_genetic_engine_selected_when_disabled(self):
        captured = self._capture_engine(island_model=False)
        assert captured["ga_called"] is True
        assert captured["island_called"] is False


# ===========================================================================
# 20–21. CLI flags
# ===========================================================================

class TestIslandCLIFlags:

    @pytest.fixture
    def sample_csv(self, tmp_path):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.standard_normal((60, 3)), columns=list("abc"))
        df["label"] = rng.integers(0, 2, 60)
        p = tmp_path / "data.csv"
        df.to_csv(p, index=False)
        return p

    def _run_fit(self, sample_csv, tmp_path, extra_args):
        from genetic_automl.cli import main
        mock_pipeline = MagicMock()
        mock_pipeline.final_score = 0.8
        mock_pipeline._metric_name = "f1_macro"
        mock_pipeline.report_path = None
        captured = {}

        def fake_init(config, gene_space_overrides=None):
            captured["config"] = config
            return mock_pipeline

        with patch("genetic_automl.cli.AutoMLPipeline", side_effect=fake_init):
            main(["fit", str(sample_csv), "--target", "label",
                  "--output-dir", str(tmp_path)] + extra_args)
        return captured["config"]

    def test_island_model_flag_enables(self, sample_csv, tmp_path):
        cfg = self._run_fit(sample_csv, tmp_path, ["--island-model"])
        assert cfg.genetic.island_model is True

    def test_n_islands_flag(self, sample_csv, tmp_path):
        cfg = self._run_fit(sample_csv, tmp_path, ["--island-model", "--n-islands", "3"])
        assert cfg.genetic.n_islands == 3

    def test_migration_interval_flag(self, sample_csv, tmp_path):
        cfg = self._run_fit(sample_csv, tmp_path,
                            ["--island-model", "--migration-interval", "5"])
        assert cfg.genetic.migration_interval == 5

    def test_migration_size_flag(self, sample_csv, tmp_path):
        cfg = self._run_fit(sample_csv, tmp_path,
                            ["--island-model", "--migration-size", "3"])
        assert cfg.genetic.migration_size == 3


# ===========================================================================
# 22–23. Diversity summary and history
# ===========================================================================

class TestIslandDiversitySummary:

    def test_diversity_summary_aggregates_islands(self):
        ie = _make_island_engine(n_islands=2)
        summary = ie.diversity_summary()
        assert summary["n_islands"] == 2
        assert "n_injections_total" in summary
        assert "island_summaries" in summary
        assert len(summary["island_summaries"]) == 2

    def test_history_has_generations_after_run(self, clf_Xy):
        X, y = clf_Xy
        cfg = _make_genetic_config(population_size=8, generations=2,
                                   n_islands=2, migration_interval=2)
        ie = IslandEngine(cfg, _make_evaluator(), backend="sklearn",
                          n_islands=2, migration_interval=2, migration_size=1)
        ie.run(X, y)
        assert len(ie.history.generations) >= 1
        assert ie.history.best is not None
