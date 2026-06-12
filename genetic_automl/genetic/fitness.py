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
  - ASHA median-stop pruning: fold loop is cut short when a chromosome's
    running mean score is already below the population fold-score median.
"""

from __future__ import annotations

import traceback
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

from genetic_automl.automl import build_automl
from genetic_automl.core.problem import (
    ProblemType,
    fitness_sign,
    get_default_metric,
)
from genetic_automl.genetic.chromosome import Chromosome
from genetic_automl.genetic.surrogate import SurrogateModel
from genetic_automl.preprocessing.pipeline import (
    PreprocessingConfig,
    PreprocessingPipeline,
)
from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)

_PREPROCESSING_GENE_KEYS = {
    "numeric_imputer", "outlier_method", "outlier_threshold", "outlier_action",
    "correlation_threshold", "categorical_encoder", "distribution_transform",
    "scaler", "missing_indicator", "feature_selection_method",
    "feature_selection_k", "imbalance_method",
    "feature_engineering", "max_interaction_features",
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
        surrogate: Optional[SurrogateModel] = None,
        asha_enabled: bool = True,
        asha_min_folds_before_prune: int = 1,
        asha_prune_margin: float = 0.0,
        cv_strategy: str = "stratified",
        group_column: Optional[str] = None,
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
        self.surrogate: Optional[SurrogateModel] = surrogate
        self.asha_enabled = asha_enabled
        self.asha_min_folds_before_prune = asha_min_folds_before_prune
        self.asha_prune_margin = asha_prune_margin
        self.cv_strategy = cv_strategy
        self.group_column = group_column
        self._cache: dict = {}
        self._cache_hits: int = 0
        # Shared pool of all individual fold scores seen across every chromosome
        # this run.  Used as the ASHA pruning reference distribution.
        self._all_fold_scores: list = []
        self._asha_prunes: int = 0

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

        # Surrogate skip: predict fitness cheaply before paying for full CV.
        # Only fires when the surrogate is trained and confident.
        if self.surrogate is not None:
            evaluated = [c for c in [chromosome]
                         if c.fitness is None]  # always true here, but guard anyway
            population_median = self._current_population_median
            skip, pred_fitness = self.surrogate.should_skip(chromosome, population_median)
            if skip:
                chromosome.fitness = pred_fitness
                chromosome.fitness_std = 0.0
                log.debug(
                    "Surrogate skipped chromosome %s | pred=%.5f | median=%.5f",
                    chromosome.id, pred_fitness, population_median,
                )
                return pred_fitness

        try:
            pp_genes, model_genes = _split_genes(chromosome.genes)
            fold_scores = []

            cv = self._build_cv(y_train)

            # For group CV: extract group array; drop group col from features.
            groups = None
            if self.cv_strategy == "group" and self.group_column:
                if self.group_column in X_train.columns:
                    groups = X_train[self.group_column].values
                else:
                    log.warning(
                        "group_column='%s' not found in X_train columns — "
                        "falling back to stratified split.", self.group_column,
                    )

            # TimeSeriesSplit.split() does not accept a y argument.
            if self.cv_strategy == "timeseries":
                split_iter = cv.split(X_train)
            elif self.cv_strategy == "group" and groups is not None:
                split_iter = cv.split(X_train, y_train, groups=groups)
            else:
                split_iter = cv.split(X_train, y_train)

            for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
                X_fold_train = X_train.iloc[train_idx].reset_index(drop=True)
                y_fold_train = y_train.iloc[train_idx].reset_index(drop=True)
                X_fold_val   = X_train.iloc[val_idx].reset_index(drop=True)
                y_fold_val   = y_train.iloc[val_idx].reset_index(drop=True)

                # Drop group column from features — it must not be used in training.
                if self.group_column and self.group_column in X_fold_train.columns:
                    X_fold_train = X_fold_train.drop(columns=[self.group_column])
                    X_fold_val   = X_fold_val.drop(columns=[self.group_column])

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
                # Record this fold score in the shared pool for future ASHA decisions.
                if score != float("-inf"):
                    self._all_fold_scores.append(score)

                log.debug(
                    "Chromosome %s | fold %d/%d | score=%.6f",
                    chromosome.id, fold_idx + 1, self.n_folds, score,
                )

                # ASHA median-stop pruning: after min_folds have completed,
                # check whether this chromosome is already trailing the field.
                if (
                    self.asha_enabled
                    and fold_idx + 1 >= self.asha_min_folds_before_prune
                    and fold_idx + 1 < self.n_folds   # don't prune on the last fold
                    and len(self._all_fold_scores) >= self.n_folds * 2  # need a reference
                ):
                    valid_so_far = [s for s in fold_scores if s != float("-inf")]
                    if valid_so_far:
                        running_mean = float(np.mean(valid_so_far))
                        reference_median = float(np.median(self._all_fold_scores))
                        if running_mean < reference_median - self.asha_prune_margin:
                            self._asha_prunes += 1
                            log.debug(
                                "ASHA prune | chromosome %s | running_mean=%.5f "
                                "< median=%.5f - margin=%.3f | folds_done=%d/%d",
                                chromosome.id, running_mean, reference_median,
                                self.asha_prune_margin, fold_idx + 1, self.n_folds,
                            )
                            # Assign penalised fitness from the folds we did run,
                            # then break out of the fold loop early.
                            fitness = running_mean
                            fitness_std = float(np.std(valid_so_far)) if len(valid_so_far) > 1 else 0.0
                            penalised_fitness = fitness - self.fitness_std_penalty * fitness_std
                            chromosome.fitness = penalised_fitness
                            chromosome.fitness_std = fitness_std
                            # Do NOT cache pruned results — a full evaluation
                            # could score higher; we don't want to lock in a low value.
                            return penalised_fitness

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

    @property
    def _current_population_median(self) -> float:
        """Median fitness of all evaluated chromosomes seen so far (from cache)."""
        cached_fitnesses = [v[0] for v in self._cache.values()
                            if v[0] != float("-inf")]
        if not cached_fitnesses:
            return float("-inf")
        return float(np.median(cached_fitnesses))

    def evaluator_summary(self) -> dict:
        """Return combined stats for surrogate and ASHA pruning."""
        result = {
            "asha_enabled": self.asha_enabled,
            "asha_prunes": self._asha_prunes,
            "asha_fold_pool_size": len(self._all_fold_scores),
        }
        if self.surrogate is not None:
            result["surrogate"] = self.surrogate.summary()
        return result

    def surrogate_summary(self) -> dict:
        """Return surrogate performance stats, or empty dict if disabled.
        Deprecated: use evaluator_summary() instead."""
        if self.surrogate is None:
            return {}
        return self.surrogate.summary()

    def _build_cv(self, y: pd.Series):
        """
        Build the appropriate CV splitter based on cv_strategy.

        Strategy       Splitter                           Use case
        ----------     --------------------------------   ----------------------------
        stratified     StratifiedKFold / KFold            Default; works for all data
        group          StratifiedGroupKFold / GroupKFold   Grouped data (patients, stores)
        timeseries     TimeSeriesSplit                    Temporal data; no shuffling
        """
        strategy = self.cv_strategy

        if strategy == "timeseries":
            # TimeSeriesSplit preserves order — no shuffle, no random_state needed.
            return TimeSeriesSplit(n_splits=self.n_folds)

        if strategy == "group":
            if self.problem_type == ProblemType.REGRESSION:
                return GroupKFold(n_splits=self.n_folds)
            return StratifiedGroupKFold(n_splits=self.n_folds, shuffle=True,
                                        random_state=self.random_seed)

        # Default: "stratified"
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
