"""
WarmStart — builds a pre-screened initial population for generation 0.

Two strategies are combined:

Strategy A — Archetype seeding
    Up to three hand-crafted chromosomes are injected at the start:
      sklearn_baseline : median imputer, standard scaler, onehot encoding.
      robust_tabular   : robust scaler, IQR outlier handling, yeo-johnson.
      tree_friendly    : no scaling, ordinal encoding, knn imputer.

Strategy B — Successive halving pre-screen
    A pool of random candidates is generated and evaluated with a fast 80/20
    split. The top fraction survive into generation 0. Their fitness is reset
    so the GA re-evaluates them with full CV on equal footing.

Generation 0 composition:
    n_seeds archetypes  +  halving survivors  +  random fill
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


class WarmStart:
    """
    Parameters
    ----------
    backend : str
    n_default_seeds : int
        Number of archetype chromosomes to inject (max 3).
    halving_pool_ratio : float
        Pool size = ratio × population_size. Set 0 to disable halving.
    halving_keep_ratio : float
        Fraction of the pool kept after halving.
    random_seed : int
    gene_space : list | None
        Pre-built gene space. None = use default for backend.
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
        self._gene_space = gene_space

    def build_initial_population(
        self,
        population_size: int,
        evaluator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> List[Chromosome]:
        """Build a warm-started initial population of population_size individuals."""
        population: List[Chromosome] = []

        seeds = self._build_default_seeds()
        population.extend(seeds[:self.n_default_seeds])
        log.info("WarmStart: injected %d archetype configs", len(population))

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

        n_random = population_size - len(population)
        if n_random > 0:
            population.extend(random_population(self.backend, n_random, self._rng, generation=0, gene_space=self._gene_space))
            log.info("WarmStart: filled %d random individuals", n_random)

        log.info(
            "WarmStart ready | archetypes=%d | survivors=%d | random=%d | total=%d",
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
        """Evaluate n_pool random candidates with an 80/20 split; return top n_keep."""
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
        log.info("WarmStart halving: evaluating %d candidates", n_pool)

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

        for s in evaluated:
            s.fitness = None
        return evaluated
