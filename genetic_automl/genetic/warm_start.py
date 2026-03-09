"""
WarmStart
---------
Produces a partially pre-evaluated initial population rather than
starting from pure random individuals.

Two complementary strategies are combined:

Strategy A — Default config seeding
    A small set of hand-crafted "known-good" configurations is injected
    into gen-0. These are validated by domain knowledge and cover the
    most common real-world tabular data scenarios.

    Three archetypes:
      - "sklearn_baseline"  : median imputer, standard scaler, onehot,
                              no outlier/transform. Reliable on clean data.
      - "robust_tabular"    : robust scaler, IQR outlier, yeo-johnson,
                              ordinal. Good for messy real-world data.
      - "tree_friendly"     : no scaling, ordinal, knn imputer. Optimal
                              for tree-based models (GBM, RF) that are
                              invariant to scaling.

Strategy B — Successive Halving pre-screen
    Generates a candidate pool of (pool_ratio × population_size) random
    individuals, evaluates them cheaply with n_folds=1 (single fold),
    then keeps only the top fraction as gen-0 survivors.

    This is data-driven: the "warm start" adapts to the actual dataset
    rather than relying on fixed human assumptions.

Combined gen-0 composition:
    ┌── n_seeds default configs   (Strategy A)
    ├── n_survivors from halving  (Strategy B)
    └── remainder random          (exploration / diversity)
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional

import pandas as pd

from genetic_automl.genetic.chromosome import (
    Chromosome,
    _random_id,
    get_gene_space,
    random_population,
)
from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Default seed configurations
# ---------------------------------------------------------------------------

def _sklearn_baseline(backend: str) -> Dict[str, Any]:
    """Safe, clean-data baseline."""
    base = {
        "numeric_imputer": "median",
        "outlier_method": "none",
        "outlier_threshold": 1.5,
        "outlier_action": "clip",
        "correlation_threshold": 0.95,
        "categorical_encoder": "onehot",
        "distribution_transform": "none",
        "scaler": "standard",
        "missing_indicator": False,
        "feature_selection_method": "none",
        "feature_selection_k": 1.0,
        "imbalance_method": "none",
    }
    if backend == "sklearn":
        base.update({"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1})
    elif backend == "autogluon":
        base.update({"presets": "medium_quality", "time_limit": 60, "ag_metric": None})
    return base


def _robust_tabular(backend: str) -> Dict[str, Any]:
    """Handles messy data: outliers, skewness, correlations."""
    base = {
        "numeric_imputer": "median",
        "outlier_method": "iqr",
        "outlier_threshold": 1.5,
        "outlier_action": "clip",
        "correlation_threshold": 0.90,
        "categorical_encoder": "ordinal",
        "distribution_transform": "yeo-johnson",
        "scaler": "robust",
        "missing_indicator": True,
        "feature_selection_method": "none",
        "feature_selection_k": 1.0,
        "imbalance_method": "none",
    }
    if backend == "sklearn":
        base.update({"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05})
    elif backend == "autogluon":
        base.update({"presets": "good_quality", "time_limit": 120, "ag_metric": None})
    return base


def _tree_friendly(backend: str) -> Dict[str, Any]:
    """Optimal for gradient boosting: no scaling, ordinal encoding."""
    base = {
        "numeric_imputer": "median",
        "outlier_method": "none",
        "outlier_threshold": 1.5,
        "outlier_action": "clip",
        "correlation_threshold": None,
        "categorical_encoder": "ordinal",
        "distribution_transform": "none",
        "scaler": "none",
        "missing_indicator": True,
        "feature_selection_method": "none",
        "feature_selection_k": 1.0,
        "imbalance_method": "class_weight",
    }
    if backend == "sklearn":
        base.update({"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1})
    elif backend == "autogluon":
        base.update({"presets": "medium_quality", "time_limit": 60, "ag_metric": None})
    return base


_DEFAULT_SEEDS = [_sklearn_baseline, _robust_tabular, _tree_friendly]


# ---------------------------------------------------------------------------
# WarmStart class
# ---------------------------------------------------------------------------

class WarmStart:
    """
    Produces a warm-started initial population.

    Parameters
    ----------
    backend : str
    n_default_seeds : int
        How many default archetype configs to inject into gen-0.
        Capped at len(_DEFAULT_SEEDS) = 3.
    halving_pool_ratio : float
        Pool size = halving_pool_ratio × population_size random candidates
        pre-screened with 1-fold CV. Set 0 to disable halving.
    halving_keep_ratio : float
        Fraction of the pool to keep as survivors (default 0.5).
    random_seed : int
    """

    def __init__(
        self,
        backend: str,
        n_default_seeds: int = 3,
        halving_pool_ratio: float = 2.0,
        halving_keep_ratio: float = 0.5,
        random_seed: int = 42,
    ) -> None:
        self.backend = backend
        self.n_default_seeds = min(n_default_seeds, len(_DEFAULT_SEEDS))
        self.halving_pool_ratio = halving_pool_ratio
        self.halving_keep_ratio = halving_keep_ratio
        self.random_seed = random_seed
        self._rng = random.Random(random_seed)

    # ------------------------------------------------------------------

    def build_initial_population(
        self,
        population_size: int,
        evaluator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> List[Chromosome]:
        """
        Build a warm-started initial population of *population_size* individuals.

        Steps:
          1. Create default-seed chromosomes (Strategy A)
          2. Pre-screen random pool with 1-fold halving (Strategy B)
          3. Fill remainder with fresh random individuals
          4. Return combined population (no fitness values yet — evaluated in engine)
        """
        population: List[Chromosome] = []

        # ── Strategy A: inject default seeds ──────────────────────────
        seeds = self._build_default_seeds()
        for seed in seeds[:self.n_default_seeds]:
            population.append(seed)
        log.info("WarmStart: injected %d default seed configs", len(population))

        # ── Strategy B: halving pre-screen ────────────────────────────
        survivors = []
        if self.halving_pool_ratio > 0 and evaluator is not None:
            n_pool = max(
                1,
                int(population_size * self.halving_pool_ratio) - len(population)
            )
            survivors = self._halving_prescreen(
                n_pool=n_pool,
                n_keep=max(1, int(n_pool * self.halving_keep_ratio)),
                evaluator=evaluator,
                X_train=X_train,
                y_train=y_train,
            )
            # Avoid duplicating any default seeds
            existing_genes = {str(c.genes) for c in population}
            for s in survivors:
                if str(s.genes) not in existing_genes:
                    population.append(s)
                    existing_genes.add(str(s.genes))
                if len(population) >= population_size:
                    break
            log.info(
                "WarmStart: added %d halving survivors (pool=%d, keep_ratio=%.1f)",
                len(survivors), n_pool, self.halving_keep_ratio,
            )

        # ── Fill remainder with random individuals ─────────────────────
        n_random = population_size - len(population)
        if n_random > 0:
            random_pop = random_population(
                backend=self.backend,
                size=n_random,
                rng=self._rng,
                generation=0,
            )
            population.extend(random_pop)
            log.info("WarmStart: filled %d random individuals", n_random)

        log.info(
            "WarmStart: initial population ready | %d seeds | %d survivors | %d random | total=%d",
            min(self.n_default_seeds, len(_DEFAULT_SEEDS)),
            len(survivors),
            n_random,
            len(population),
        )
        return population[:population_size]

    # ------------------------------------------------------------------

    def _build_default_seeds(self) -> List[Chromosome]:
        """Build Chromosome objects from each archetype factory."""
        seeds = []
        for factory in _DEFAULT_SEEDS[:self.n_default_seeds]:
            genes = factory(self.backend)
            # Fill any missing genes with random values
            gene_space = {g.name: g for g in get_gene_space(self.backend)}
            for name, gdef in gene_space.items():
                if name not in genes:
                    genes[name] = gdef.random_value(self._rng)
            seeds.append(Chromosome(genes=genes, generation=0))
        return seeds

    def _halving_prescreen(
        self,
        n_pool: int,
        n_keep: int,
        evaluator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> List[Chromosome]:
        """
        Evaluate n_pool random chromosomes with a 1-fold fast eval,
        return the top n_keep survivors.
        """
        from genetic_automl.genetic.fitness import FitnessEvaluator

        # Build a cheap 1-fold evaluator
        fast_evaluator = FitnessEvaluator(
            problem_type=evaluator.problem_type,
            target_column=evaluator.target_column,
            backend=evaluator.backend,
            metric=evaluator.metric,
            n_folds=1,                 # single fold — fast
            random_seed=evaluator.random_seed,
        )

        pool = random_population(
            backend=self.backend,
            size=n_pool,
            rng=self._rng,
            generation=0,
        )

        log.info("WarmStart halving: evaluating %d candidates (1-fold fast)…", n_pool)
        for chrom in pool:
            fast_evaluator.evaluate(chrom, X_train, y_train)

        # Sort by fitness, keep top n_keep
        evaluated = [c for c in pool if c.fitness is not None and c.fitness != float("-inf")]
        evaluated.sort(key=lambda c: c.fitness, reverse=True)
        survivors = evaluated[:n_keep]

        # Reset fitness so they get re-evaluated properly in the GA loop
        for s in survivors:
            s.fitness = None

        return survivors
