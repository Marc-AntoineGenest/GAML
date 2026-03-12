"""
FitnessEvaluator — scores a Chromosome via k-fold cross-validation.

Design principles:
  - Zero leakage: PreprocessingPipeline is fit fresh on each fold's train split.
    Val/test data only ever sees transform(), never fit().
  - ImbalanceHandler is applied to fold training data only.
  - The GA always maximises fitness:
      classification metrics (F1, accuracy, AUC) — returned as-is
      regression metrics (MSE, MAE)              — negated
  - Chromosomes with identical genes reuse a cached result (no redundant CV).
  - Fitness includes a configurable std penalty to favour stable pipelines:
      fitness = mean_cv - penalty * std_cv
"""

from __future__ import annotations

import traceback
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

from genetic_automl.automl import build_automl
from genetic_automl.core.problem import (
    ProblemType,
    fitness_sign,
    get_default_metric,
)
from genetic_automl.genetic.chromosome import Chromosome
from genetic_automl.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline
from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)

_PREPROCESSING_GENE_KEYS = {
    "numeric_imputer", "outlier_method", "outlier_threshold", "outlier_action",
    "correlation_threshold", "categorical_encoder", "distribution_transform",
    "scaler", "missing_indicator", "feature_selection_method",
    "feature_selection_k", "imbalance_method",
}


def _split_genes(genes: dict):
    """Partition chromosome genes into preprocessing genes and model genes."""
    pp_genes = {k: v for k, v in genes.items() if k in _PREPROCESSING_GENE_KEYS}
    model_genes = {k: v for k, v in genes.items() if k not in _PREPROCESSING_GENE_KEYS}
    return pp_genes, model_genes


class FitnessEvaluator:
    """
    Evaluate a Chromosome via stratified k-fold CV on the training set.

    Parameters
    ----------
    problem_type : ProblemType
    target_column : str
    backend : str
    metric : str | None
        Scoring metric. None = default for problem_type.
    n_folds : int
        Number of CV folds. 3 is a good default for speed; use 5 for production.
    fitness_std_penalty : float
        Coefficient for the std penalty term. 0.0 = pure mean CV score.
    random_seed : int
    """

    def __init__(
        self,
        problem_type: ProblemType,
        target_column: str,
        backend: str = "autogluon",
        metric: Optional[str] = None,
        n_folds: int = 3,
        multi_objective_metrics: Optional[List[str]] = None,
        multi_objective_weights: Optional[List[float]] = None,
        random_seed: int = 42,
        fitness_std_penalty: float = 0.5,
    ) -> None:
        self.problem_type = problem_type
        self.target_column = target_column
        self.backend = backend
        self.metric = metric or get_default_metric(problem_type)
        self.n_folds = n_folds
        self.multi_objective_metrics = multi_objective_metrics
        self.multi_objective_weights = multi_objective_weights
        self.random_seed = random_seed
        self.fitness_std_penalty = fitness_std_penalty
        self._cache: dict = {}
        self._cache_hits: int = 0

    def evaluate(
        self,
        chromosome: Chromosome,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
    ) -> float:
        """
        Score chromosome via k-fold CV. Returns fitness (float).
        Returns float('-inf') on failure.
        """
        cache_key = tuple(sorted(chromosome.genes.items()))
        if cache_key in self._cache:
            cached_fitness, cached_std = self._cache[cache_key]
            chromosome.fitness = cached_fitness
            chromosome.fitness_std = cached_std
            self._cache_hits += 1
            log.debug(
                "Chromosome %s | cache hit (fitness=%.6f) | total_hits=%d",
                chromosome.id, cached_fitness, self._cache_hits,
            )
            return cached_fitness

        try:
            pp_genes, model_genes = _split_genes(chromosome.genes)
            fold_scores = []

            cv = self._build_cv(y_train)

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                X_fold_train = X_train.iloc[train_idx].reset_index(drop=True)
                y_fold_train = y_train.iloc[train_idx].reset_index(drop=True)
                X_fold_val   = X_train.iloc[val_idx].reset_index(drop=True)
                y_fold_val   = y_train.iloc[val_idx].reset_index(drop=True)

                pp_config = PreprocessingConfig.from_genes(pp_genes)
                pp = PreprocessingPipeline(
                    config=pp_config,
                    problem_type=self.problem_type,
                    random_seed=self.random_seed,
                )
                X_tr_pp, y_tr_pp = pp.fit_transform_train(X_fold_train, y_fold_train)
                X_vl_pp = pp.transform(X_fold_val)

                if X_tr_pp.shape[1] == 0:
                    log.warning("Chromosome %s fold %d: all features dropped", chromosome.id, fold_idx)
                    fold_scores.append(float("-inf"))
                    continue

                model = build_automl(
                    backend=self.backend,
                    problem_type=self.problem_type,
                    target_column=self.target_column,
                    random_seed=self.random_seed,
                    **{k: v for k, v in model_genes.items() if v is not None},
                )
                model.fit(X_tr_pp, y_tr_pp, X_vl_pp, y_fold_val)

                if self.problem_type == ProblemType.MULTI_OBJECTIVE:
                    score = self._multi_objective_score(model, X_vl_pp, y_fold_val)
                else:
                    raw = model.score(X_vl_pp, y_fold_val, metric=self.metric)
                    score = raw * fitness_sign(self.metric)

                fold_scores.append(score)
                log.debug(
                    "Chromosome %s | fold %d/%d | score=%.6f",
                    chromosome.id, fold_idx + 1, self.n_folds, score,
                )

            valid = [s for s in fold_scores if s != float("-inf")]
            if not valid:
                chromosome.fitness = float("-inf")
                return float("-inf")

            fitness = float(np.mean(valid))
            fitness_std = float(np.std(valid)) if len(valid) > 1 else 0.0
            penalised_fitness = fitness - self.fitness_std_penalty * fitness_std

            chromosome.fitness = penalised_fitness
            chromosome.fitness_std = fitness_std
            self._cache[cache_key] = (penalised_fitness, fitness_std)
            chromosome._pp_genes = pp_genes  # noqa: SLF001

            log.info(
                "Chromosome %s | CV mean=%.6f | std=%.6f | penalty=%.6f | fitness=%.6f | genes=%s",
                chromosome.id, fitness, fitness_std,
                self.fitness_std_penalty * fitness_std, penalised_fitness,
                {**pp_genes, **model_genes},
            )
            return penalised_fitness

        except Exception as exc:
            log.warning(
                "Chromosome %s failed: %s\n%s",
                chromosome.id, exc, traceback.format_exc(),
            )
            chromosome.fitness = float("-inf")
            return float("-inf")

    def _build_cv(self, y: pd.Series):
        """Stratified KFold for classification, regular KFold for regression."""
        if self.problem_type == ProblemType.REGRESSION:
            return KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
        return StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)

    def _multi_objective_score(self, model, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        metrics = self.multi_objective_metrics or [self.metric]
        weights = self.multi_objective_weights or [1.0 / len(metrics)] * len(metrics)
        scores = [
            model.score(X_val, y_val, metric=m) * fitness_sign(m)
            for m in metrics
        ]
        return sum(w * s for w, s in zip(weights, scores))
