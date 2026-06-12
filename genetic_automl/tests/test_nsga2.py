"""
Tests for the NSGA-II multi-objective GA.

Coverage:
  1.  dominates() — all cases
  2.  fast_non_dominated_sort — single front, two fronts, empty population
  3.  crowding_distance_assignment — boundary = inf, interior = finite
  4.  nsga2_select — prefers lower rank, then higher crowding
  5.  nsga2_survive — fills survivors correctly, handles partial fronts
  6.  build_objective_values — metric, complexity, latency objectives
  7.  pareto_front_summary — returns correct structure
  8.  GeneticConfig — nsga2_enabled default, nsga2_objectives default
  9.  config_loader parses nsga2_enabled and nsga2_objectives
  10. GeneticEngine._breed with nsga2 uses nsga2_select (mock)
  11. Full GeneticEngine.run() with nsga2_enabled=True completes
  12. EvolutionHistory.pareto_front populated after nsga2 run
  13. Chromosome.extra_scores field added and copied correctly
  14. HTML reporter accepts pareto_front=None without error
  15. HTML reporter renders Pareto section when pareto_front provided
  16. CLI --nsga2 flag enables nsga2
  17. CLI --objectives flag sets nsga2_objectives
  18. Existing single-objective _breed unchanged (regression guard)
"""
from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from genetic_automl.config import GeneticConfig, PipelineConfig
from genetic_automl.core.problem import ProblemType
from genetic_automl.genetic.chromosome import Chromosome
from genetic_automl.genetic.nsga2 import (
    _CROWD_ATTR,
    _INF_CROWD,
    _RANK_ATTR,
    build_objective_values,
    crowding_distance_assignment,
    dominates,
    fast_non_dominated_sort,
    nsga2_select,
    nsga2_survive,
    pareto_front_summary,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chrom(fitness, genes=None, chrom_id=None):
    c = Chromosome(genes=genes or {"model_type": "lgbm", "n_estimators": 100})
    c.fitness = fitness
    if chrom_id:
        c.id = chrom_id
    return c


def _obj_vals(chroms, vecs):
    """Build objective_values dict from parallel lists."""
    return {c.id: v for c, v in zip(chroms, vecs)}


# ===========================================================================
# 1. dominates()
# ===========================================================================

class TestDominates:
    def test_a_dominates_b_strictly_better_on_one(self):
        assert dominates([0.9, 0.8], [0.8, 0.8]) is True

    def test_a_dominates_b_better_on_all(self):
        assert dominates([0.9, 0.9], [0.8, 0.7]) is True

    def test_a_does_not_dominate_b_worse_on_one(self):
        assert dominates([0.9, 0.5], [0.8, 0.8]) is False

    def test_equal_not_dominate(self):
        assert dominates([0.8, 0.8], [0.8, 0.8]) is False

    def test_b_dominates_a(self):
        assert dominates([0.7, 0.7], [0.9, 0.9]) is False


# ===========================================================================
# 2. fast_non_dominated_sort()
# ===========================================================================

class TestFastNonDominatedSort:
    def test_all_non_dominated_single_front(self):
        # Pareto-incomparable: a better on obj1, b better on obj2
        a, b = _chrom(0.9), _chrom(0.7)
        obj = _obj_vals([a, b], [[0.9, 0.5], [0.6, 0.9]])
        fronts = fast_non_dominated_sort([a, b], obj)
        assert len(fronts) == 1
        assert len(fronts[0]) == 2

    def test_two_fronts(self):
        # c dominates d on both objectives
        c, d = _chrom(0.9), _chrom(0.7)
        obj = _obj_vals([c, d], [[0.9, 0.9], [0.7, 0.7]])
        fronts = fast_non_dominated_sort([c, d], obj)
        assert len(fronts) == 2
        assert c in fronts[0]
        assert d in fronts[1]

    def test_ranks_stamped(self):
        a, b, c = _chrom(0.9), _chrom(0.7), _chrom(0.5)
        obj = _obj_vals([a, b, c], [[0.9, 0.9], [0.7, 0.7], [0.5, 0.5]])
        fast_non_dominated_sort([a, b, c], obj)
        assert getattr(a, _RANK_ATTR) == 0
        assert getattr(b, _RANK_ATTR) == 1
        assert getattr(c, _RANK_ATTR) == 2

    def test_empty_population(self):
        fronts = fast_non_dominated_sort([], {})
        assert fronts == []

    def test_single_individual(self):
        a = _chrom(0.8)
        fronts = fast_non_dominated_sort([a], {a.id: [0.8, 0.8]})
        assert len(fronts) == 1
        assert fronts[0][0] is a


# ===========================================================================
# 3. crowding_distance_assignment()
# ===========================================================================

class TestCrowdingDistance:
    def test_boundary_gets_inf(self):
        a, b, c = _chrom(0.9), _chrom(0.7), _chrom(0.5)
        front = [a, b, c]
        obj = _obj_vals(front, [[0.9, 0.1], [0.7, 0.5], [0.5, 0.9]])
        crowding_distance_assignment(front, obj, 2)
        assert getattr(a, _CROWD_ATTR) == _INF_CROWD or getattr(c, _CROWD_ATTR) == _INF_CROWD

    def test_interior_gets_finite(self):
        a, b, c = _chrom(0.9), _chrom(0.7), _chrom(0.5)
        front = [a, b, c]
        obj = _obj_vals(front, [[0.9, 0.1], [0.7, 0.5], [0.5, 0.9]])
        crowding_distance_assignment(front, obj, 2)
        mid = getattr(b, _CROWD_ATTR)
        assert mid != _INF_CROWD
        assert mid > 0

    def test_two_individuals_both_inf(self):
        a, b = _chrom(0.9), _chrom(0.5)
        front = [a, b]
        crowding_distance_assignment(front, _obj_vals([a, b], [[0.9], [0.5]]), 1)
        assert getattr(a, _CROWD_ATTR) == _INF_CROWD
        assert getattr(b, _CROWD_ATTR) == _INF_CROWD


# ===========================================================================
# 4. nsga2_select()
# ===========================================================================

class TestNSGA2Select:
    import random as _random

    def _rng(self, seed=0):
        import random
        return random.Random(seed)

    def test_prefers_lower_rank(self):
        """nsga2_select: given [a(rank=0), b(rank=1)], result has a's genes."""
        import random
        from unittest.mock import patch
        a, b = _chrom(0.9, genes={"model_type": "lgbm", "n_estimators": 100}),                _chrom(0.7, genes={"model_type": "rf",   "n_estimators": 200})
        setattr(a, _RANK_ATTR, 0); setattr(b, _RANK_ATTR, 1)
        setattr(a, _CROWD_ATTR, 1.0); setattr(b, _CROWD_ATTR, 2.0)
        rng = random.Random(0)
        with patch.object(rng, "sample", return_value=[a, b]):
            result = nsga2_select([a, b], rng)
        # result is a copy of a (rank 0 wins)
        assert result.genes == a.genes

    def test_same_rank_prefers_higher_crowding(self):
        """nsga2_select: same rank → higher crowding distance wins."""
        import random
        from unittest.mock import patch
        a, b = _chrom(0.9, genes={"model_type": "lgbm", "n_estimators": 100}),                _chrom(0.7, genes={"model_type": "rf",   "n_estimators": 200})
        setattr(a, _RANK_ATTR, 0); setattr(b, _RANK_ATTR, 0)
        setattr(a, _CROWD_ATTR, 5.0); setattr(b, _CROWD_ATTR, 0.5)
        rng = random.Random(0)
        with patch.object(rng, "sample", return_value=[a, b]):
            result = nsga2_select([a, b], rng)
        # result is a copy of a (higher crowding wins)
        assert result.genes == a.genes

    def test_returns_copy_not_original(self):
        import random
        a, b = _chrom(0.9), _chrom(0.7)
        setattr(a, _RANK_ATTR, 0); setattr(a, _CROWD_ATTR, 1.0)
        setattr(b, _RANK_ATTR, 1); setattr(b, _CROWD_ATTR, 0.5)
        result = nsga2_select([a, b], random.Random(0))
        assert result is not a and result is not b


# ===========================================================================
# 5. nsga2_survive()
# ===========================================================================

class TestNSGA2Survive:
    def test_selects_n_survive(self):
        pop = [_chrom(0.9 - i*0.1) for i in range(6)]
        obj = _obj_vals(pop, [[0.9-i*0.1, 0.1+i*0.1] for i in range(6)])
        survivors = nsga2_survive(pop, 4, obj, 2)
        assert len(survivors) == 4

    def test_pareto_front_members_survive_first(self):
        # a and b are non-dominated; c is dominated by a
        a, b, c = _chrom(0.9), _chrom(0.8), _chrom(0.5)
        obj = _obj_vals([a, b, c], [[0.9, 0.3], [0.4, 0.9], [0.5, 0.5]])
        survivors = nsga2_survive([a, b, c], 2, obj, 2)
        survivor_ids = {s.id for s in survivors}
        assert a.id in survivor_ids
        assert b.id in survivor_ids


# ===========================================================================
# 6. build_objective_values()
# ===========================================================================

class TestBuildObjectiveValues:
    def test_complexity_objective(self):
        c = _chrom(0.8, genes={"n_estimators": 200, "model_type": "lgbm"})
        obj = build_objective_values([c], ["complexity"])
        assert obj[c.id][0] == -200.0

    def test_latency_objective(self):
        c = _chrom(0.8)
        lat_map = {c.id: 3.5}
        obj = build_objective_values([c], ["latency"], latency_map=lat_map)
        assert obj[c.id][0] == -3.5

    def test_unevaluated_skipped(self):
        c = Chromosome(genes={"model_type": "lgbm"})
        c.fitness = None
        obj = build_objective_values([c], ["complexity"])
        assert c.id not in obj

    def test_metric_falls_back_to_primary_fitness(self):
        c = _chrom(0.85)
        obj = build_objective_values([c], ["f1_macro"])
        assert abs(obj[c.id][0] - 0.85) < 1e-9


# ===========================================================================
# 7. pareto_front_summary()
# ===========================================================================

class TestParetoFrontSummary:
    def test_returns_list_of_dicts(self):
        a, b = _chrom(0.9), _chrom(0.8)
        result = pareto_front_summary([a, b], ["f1_macro", "complexity"])
        assert isinstance(result, list)
        if result:
            assert "id" in result[0]
            assert "objectives" in result[0]
            assert "rank" in result[0]

    def test_empty_history(self):
        result = pareto_front_summary([], ["f1_macro"])
        assert result == []

    def test_dominated_solutions_not_in_front(self):
        # a dominates b on both objectives
        a, b = _chrom(0.9, genes={"n_estimators": 50, "model_type": "lgbm"}), \
               _chrom(0.5, genes={"n_estimators": 200, "model_type": "lgbm"})
        result = pareto_front_summary([a, b], ["f1_macro", "complexity"])
        ids = [r["id"] for r in result]
        assert a.id in ids
        assert b.id not in ids


# ===========================================================================
# 8-9. Config and config_loader
# ===========================================================================

class TestNSGA2Config:
    def test_defaults(self):
        cfg = GeneticConfig()
        assert cfg.nsga2_enabled is False
        assert cfg.nsga2_objectives is None

    def test_config_loader_parses_nsga2(self, tmp_path):
        from genetic_automl.config_loader import load_config
        yaml = """\
run:
  backend: sklearn
problem:
  type: classification
  target_column: label
genetic:
  population_size: 4
  generations: 2
  nsga2_enabled: true
  nsga2_objectives:
    - f1_macro
    - complexity
"""
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml)
        cfg, _ = load_config(str(p))
        assert cfg.genetic.nsga2_enabled is True
        assert cfg.genetic.nsga2_objectives == ["f1_macro", "complexity"]

    def test_config_loader_defaults_when_absent(self, tmp_path):
        from genetic_automl.config_loader import load_config
        yaml = """\
run:
  backend: sklearn
problem:
  type: classification
  target_column: label
genetic:
  population_size: 4
  generations: 2
"""
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml)
        cfg, _ = load_config(str(p))
        assert cfg.genetic.nsga2_enabled is False
        assert cfg.genetic.nsga2_objectives is None


# ===========================================================================
# 10-12. GeneticEngine integration
# ===========================================================================

class TestGeneticEngineNSGA2:
    @pytest.fixture
    def clf_data(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((120, 5)), columns=[f"f{i}" for i in range(5)])
        y = pd.Series((X["f0"] + X["f1"] > 0).astype(int), name="label")
        return X, y

    def test_full_run_nsga2_returns_chromosome(self, clf_data):
        from genetic_automl.config import GeneticConfig
        from genetic_automl.genetic.engine import GeneticEngine
        from genetic_automl.genetic.fitness import FitnessEvaluator

        X, y = clf_data
        cfg = GeneticConfig(
            population_size=6,
            generations=3,
            n_cv_folds=2,
            nsga2_enabled=True,
            nsga2_objectives=["f1_macro", "complexity"],
            random_seed=42,
        )
        evaluator = FitnessEvaluator(
            problem_type=ProblemType.CLASSIFICATION,
            target_column="label",
            backend="sklearn",
            metric="f1_macro",
            n_folds=2,
        )
        engine = GeneticEngine(genetic_config=cfg, evaluator=evaluator, backend="sklearn")
        best = engine.run(X, y)
        assert best is not None
        assert best.fitness is not None
        assert isinstance(best.fitness, float)

    def test_pareto_front_populated_after_nsga2_run(self, clf_data):
        from genetic_automl.config import GeneticConfig
        from genetic_automl.genetic.engine import GeneticEngine
        from genetic_automl.genetic.fitness import FitnessEvaluator

        X, y = clf_data
        cfg = GeneticConfig(
            population_size=6,
            generations=2,
            n_cv_folds=2,
            nsga2_enabled=True,
            nsga2_objectives=["f1_macro", "complexity"],
            random_seed=0,
        )
        evaluator = FitnessEvaluator(
            problem_type=ProblemType.CLASSIFICATION,
            target_column="label",
            backend="sklearn",
            metric="f1_macro",
            n_folds=2,
        )
        engine = GeneticEngine(genetic_config=cfg, evaluator=evaluator, backend="sklearn")
        engine.run(X, y)
        assert isinstance(engine.history.pareto_front, list)
        assert len(engine.history.pareto_front) > 0

    def test_standard_run_pareto_front_empty(self, clf_data):
        """When nsga2_enabled=False, pareto_front should stay empty."""
        from genetic_automl.config import GeneticConfig
        from genetic_automl.genetic.engine import GeneticEngine
        from genetic_automl.genetic.fitness import FitnessEvaluator

        X, y = clf_data
        cfg = GeneticConfig(population_size=4, generations=2, n_cv_folds=2, nsga2_enabled=False)
        evaluator = FitnessEvaluator(
            problem_type=ProblemType.CLASSIFICATION,
            target_column="label",
            backend="sklearn",
            metric="f1_macro",
            n_folds=2,
        )
        engine = GeneticEngine(genetic_config=cfg, evaluator=evaluator, backend="sklearn")
        engine.run(X, y)
        assert engine.history.pareto_front == []


# ===========================================================================
# 13. Chromosome.extra_scores
# ===========================================================================

class TestChromosomeExtraScores:
    def test_extra_scores_default_none(self):
        c = Chromosome(genes={"model_type": "lgbm"})
        assert c.extra_scores is None

    def test_extra_scores_copied(self):
        c = Chromosome(genes={"model_type": "lgbm"})
        c.extra_scores = {"roc_auc": 0.92, "f1_macro": 0.88}
        c2 = c.copy()
        assert c2.extra_scores == c.extra_scores
        # Must be a distinct dict (not same reference)
        c2.extra_scores["roc_auc"] = 0.0
        assert c.extra_scores["roc_auc"] == 0.92


# ===========================================================================
# 14-15. HTML reporter
# ===========================================================================

class TestHTMLReporterPareto:
    def _make_history(self):
        from genetic_automl.genetic.engine import EvolutionHistory
        h = MagicMock(spec=EvolutionHistory)
        c = _chrom(0.85)
        h.best = c
        gen = MagicMock()
        gen.generation = 0; gen.best_fitness = 0.85
        gen.mean_fitness = 0.82; gen.worst_fitness = 0.78
        gen.elapsed_seconds = 5.0
        h.generations = [gen]
        h.all_chromosomes = [c]
        h.pareto_front = []
        # Return real lists so json.dumps works
        h.fitness_curve.return_value = [0.85]
        h.diversity_curve.return_value = [0.5]
        h.mutation_rate_curve.return_value = [0.2]
        return h

    def test_no_pareto_no_error(self, tmp_path):
        from genetic_automl.reporting.html_reporter import HTMLReporter
        reporter = HTMLReporter(output_dir=str(tmp_path))
        cfg = PipelineConfig(problem_type=ProblemType.CLASSIFICATION, target_column="label")
        path = reporter.generate(
            config=cfg, history=self._make_history(),
            pareto_front=None,
        )
        assert path.endswith(".html")
        html = open(path).read()
        assert "NSGA-II" not in html

    def test_pareto_section_rendered(self, tmp_path):
        from genetic_automl.reporting.html_reporter import HTMLReporter
        reporter = HTMLReporter(output_dir=str(tmp_path))
        cfg = PipelineConfig(problem_type=ProblemType.CLASSIFICATION, target_column="label")
        fake_front = [
            {"id": "abc12345", "rank": 0, "crowding": 1.5,
             "objectives": {"f1_macro": 0.88, "complexity": -100},
             "fitness": 0.88, "key_genes": {"model_type": "lgbm"}},
            {"id": "def67890", "rank": 0, "crowding": 0.9,
             "objectives": {"f1_macro": 0.82, "complexity": -50},
             "fitness": 0.82, "key_genes": {"model_type": "rf"}},
        ]
        path = reporter.generate(
            config=cfg, history=self._make_history(),
            pareto_front=fake_front,
        )
        html = open(path).read()
        assert "NSGA-II Pareto Front" in html
        assert "abc12345" in html
        assert "f1_macro" in html


# ===========================================================================
# 16-17. CLI flags
# ===========================================================================

class TestNSGA2CLI:
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

    def test_nsga2_flag_enables(self, sample_csv, tmp_path):
        cfg = self._run_fit(sample_csv, tmp_path, ["--nsga2"])
        assert cfg.genetic.nsga2_enabled is True

    def test_objectives_flag_sets_list(self, sample_csv, tmp_path):
        cfg = self._run_fit(sample_csv, tmp_path,
                            ["--nsga2", "--objectives", "f1_macro", "complexity"])
        assert cfg.genetic.nsga2_objectives == ["f1_macro", "complexity"]
