"""
WarmStart
---------
Produces a pre-screened initial population rather than starting from pure random.

Two strategies are combined:

Strategy A — Default config seeding
    Three hand-crafted archetypes are injected into gen-0:
      - sklearn_baseline  : median imputer, standard scaler, onehot. Reliable on clean data.
      - robust_tabular    : robust scaler, IQR outlier, yeo-johnson. Handles messy real-world data.
      - tree_friendly     : no scaling, ordinal, knn imputer. Optimal for GBM/RF.

Strategy B — Successive Halving pre-screen
    Generates pool_ratio × population_size random candidates, evaluates each with a
    fast 80/20 split, and keeps the top fraction as gen-0 survivors.
    Resets their fitness to None so the GA re-evaluates them with full CV.

Combined gen-0:
    ┌── n_seeds default configs   (Strategy A)
    ├── n_survivors from halving  (Strategy B)
    └── remainder random          (diversity)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

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
    base = {
        "numeric_imputer": "median", "outlier_method": "none",
        "outlier_threshold": 1.5, "outlier_action": "clip",
        "correlation_threshold": 0.95, "categorical_encoder": "onehot",
        "distribution_transform": "none", "scaler": "standard",
        "missing_indicator": False, "feature_selection_method": "none",
        "feature_selection_k": 1.0, "imbalance_method": "none",
    }
    if backend == "sklearn":
        base.update({"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1})
    elif backend == "autogluon":
        base.update({"presets": "medium_quality", "time_limit": 60, "ag_metric": None})
    return base


def _robust_tabular(backend: str) -> Dict[str, Any]:
    base = {
        "numeric_imputer": "median", "outlier_method": "iqr",
        "outlier_threshold": 1.5, "outlier_action": "clip",
        "correlation_threshold": 0.90, "categorical_encoder": "ordinal",
        "distribution_transform": "yeo-johnson", "scaler": "robust",
        "missing_indicator": True, "feature_selection_method": "none",
        "feature_selection_k": 1.0, "imbalance_method": "none",
    }
    if backend == "sklearn":
        base.update({"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05})
    elif backend == "autogluon":
        base.update({"presets": "good_quality", "time_limit": 120, "ag_metric": None})
    return base


def _tree_friendly(backend: str) -> Dict[str, Any]:
    base = {
        "numeric_imputer": "median", "outlier_method": "none",
        "outlier_threshold": 1.5, "outlier_action": "clip",
        "correlation_threshold": None, "categorical_encoder": "ordinal",
        "distribution_transform": "none", "scaler": "none",
        "missing_indicator": True, "feature_selection_method": "none",
        "feature_selection_k": 1.0, "imbalance_method": "class_weight",
    }
    if backend == "sklearn":
        base.update({"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1})
    elif backend == "autogluon":
        base.update({"presets": "medium_quality", "time_limit": 60, "ag_metric": None})
    return base


_DEFAULT_SEEDS = [_sklearn_baseline, _robust_tabular, _tree_friendly]


# ---------------------------------------------------------------------------
# WarmStart
# ---------------------------------------------------------------------------

class WarmStart:
    """
    Parameters
    ----------
    backend : str
    n_default_seeds : int
        Archetype configs injected into gen-0 (max 3).
    halving_pool_ratio : float
        Pool size = ratio × population_size. Set 0 to disable.
    halving_keep_ratio : float
        Fraction of pool kept as survivors.
    random_seed : int
    """

    def __init__(
        self,
        backend: str,
        n_default_seeds: int = 3,
        halving_pool_ratio: float = 2.0,
        halving_keep_ratio: float = 0.5,
        random_seed: int = 42,
        gene_space=None,
    ) -> None:
        self.backend = backend
        self.n_default_seeds = min(n_default_seeds, len(_DEFAULT_SEEDS))
        self.halving_pool_ratio = halving_pool_ratio
        self.halving_keep_ratio = halving_keep_ratio
        self.random_seed = random_seed
        self._rng = random.Random(random_seed)
        self._gene_space = gene_space  # None → default space used by random_population

    def build_initial_population(
        self,
        population_size: int,
        evaluator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> List[Chromosome]:
        """Build a warm-started initial population of population_size individuals."""
        population: List[Chromosome] = []

        # Strategy A: inject default seeds
        seeds = self._build_default_seeds()
        population.extend(seeds[:self.n_default_seeds])
        log.info("WarmStart: injected %d default seed configs", len(population))

        # Strategy B: halving pre-screen
        survivors = []
        if self.halving_pool_ratio > 0 and evaluator is not None:
            n_pool = max(1, int(population_size * self.halving_pool_ratio) - len(population))
            n_keep = max(1, int(n_pool * self.halving_keep_ratio))
            survivors = self._halving_prescreen(n_pool, n_keep, evaluator, X_train, y_train)
            existing = {str(c.genes) for c in population}
            for s in survivors:
                if str(s.genes) not in existing and len(population) < population_size:
                    population.append(s)
                    existing.add(str(s.genes))
            log.info("WarmStart: added %d halving survivors (pool=%d)", len(survivors), n_pool)

        # Fill remainder with random individuals
        n_random = population_size - len(population)
        if n_random > 0:
            population.extend(random_population(self.backend, n_random, self._rng, generation=0, gene_space=self._gene_space))
            log.info("WarmStart: filled %d random individuals", n_random)

        log.info(
            "WarmStart ready | seeds=%d | survivors=%d | random=%d | total=%d",
            self.n_default_seeds, len(survivors), n_random, len(population),
        )
        return population[:population_size]

    def _build_default_seeds(self) -> List[Chromosome]:
        seeds = []
        gene_space = {g.name: g for g in get_gene_space(self.backend)}
        for factory in _DEFAULT_SEEDS[:self.n_default_seeds]:
            genes = factory(self.backend)
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
        """Evaluate n_pool candidates with a fast 80/20 split, return top n_keep."""
        from sklearn.model_selection import train_test_split as _tts
        from genetic_automl.core.problem import fitness_sign, compute_metric
        from genetic_automl.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline
        from genetic_automl.automl import build_automl
        from genetic_automl.genetic.fitness import _split_genes

        stratify = y_train if evaluator.problem_type.value == "classification" else None
        try:
            X_tr, X_hld, y_tr, y_hld = _tts(
                X_train, y_train, test_size=0.20,
                random_state=self.random_seed, stratify=stratify,
            )
        except Exception:
            X_tr, X_hld, y_tr, y_hld = _tts(
                X_train, y_train, test_size=0.20, random_state=self.random_seed,
            )

        pool = random_population(self.backend, n_pool, self._rng, generation=0, gene_space=self._gene_space)
        sign = fitness_sign(evaluator.metric)
        log.info("WarmStart halving: evaluating %d candidates…", n_pool)

        for chrom in pool:
            try:
                pp_genes, model_genes = _split_genes(chrom.genes)
                pp = PreprocessingPipeline(
                    config=PreprocessingConfig.from_genes(pp_genes),
                    problem_type=evaluator.problem_type,
                    random_seed=self.random_seed,
                )
                X_tr_pp, y_tr_pp = pp.fit_transform_train(X_tr, y_tr)
                X_hld_pp = pp.transform(X_hld)
                model = build_automl(
                    backend=evaluator.backend,
                    problem_type=evaluator.problem_type,
                    target_column=evaluator.target_column,
                    random_seed=self.random_seed,
                    **{k: v for k, v in model_genes.items() if v is not None},
                )
                model.fit(X_tr_pp, y_tr_pp)
                chrom.fitness = sign * compute_metric(evaluator.metric, y_hld.values, model.predict(X_hld_pp))
            except Exception as exc:
                log.debug("WarmStart halving: candidate failed (%s)", exc)
                chrom.fitness = float("-inf")

        evaluated = sorted(
            [c for c in pool if c.fitness not in (None, float("-inf"))],
            key=lambda c: c.fitness, reverse=True,
        )[:n_keep]

        log.info(
            "WarmStart halving: %d/%d survived (best=%.4f)",
            len(evaluated), n_pool,
            evaluated[0].fitness if evaluated else float("nan"),
        )

        # Reset fitness — GA will re-evaluate with full CV
        for s in evaluated:
            s.fitness = None
        return evaluated
