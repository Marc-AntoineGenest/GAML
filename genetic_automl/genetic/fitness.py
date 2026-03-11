"""
FitnessEvaluator
----------------
Evaluates a Chromosome using k-fold cross-validation on the training set,
returning the mean CV score as fitness.

Why k-fold instead of a single val split:
  - Prevents the GA from exploiting a lucky val split
  - Fitness signal is less noisy — GA converges to genuinely good configs
  - The test set remains completely untouched until final evaluation

Zero-leakage per fold:
  - PreprocessingPipeline is fit fresh on each fold's train portion
  - No data from the validation fold touches any fit step
  - ImbalanceHandler is applied only to the fold's training portion

The GA always *maximizes* fitness:
  - Classification metrics (F1, accuracy, AUC) → returned as-is
  - Regression metrics (MSE, MAE) → negated
  - Multi-objective → weighted scalarization
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

# Genes that belong to the preprocessing layer (not forwarded to the AutoML model)
_PREPROCESSING_GENE_KEYS = {
    "numeric_imputer",
    "outlier_method",
    "outlier_threshold",
    "outlier_action",
    "correlation_threshold",
    "categorical_encoder",
    "distribution_transform",
    "scaler",
    "missing_indicator",
    "feature_selection_method",
    "feature_selection_k",
    "imbalance_method",
}


def _split_genes(genes: dict):
    """Split chromosome genes into preprocessing genes and model genes."""
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
    n_folds : int
        Number of CV folds (default 3 — balances quality vs speed in GA context).
    multi_objective_metrics : list[str] | None
    multi_objective_weights : list[float] | None
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
    ) -> None:
        self.problem_type = problem_type
        self.target_column = target_column
        self.backend = backend
        self.metric = metric or get_default_metric(problem_type)
        self.n_folds = n_folds
        self.multi_objective_metrics = multi_objective_metrics
        self.multi_objective_weights = multi_objective_weights
        self.random_seed = random_seed

    # ------------------------------------------------------------------

    def evaluate(
        self,
        chromosome: Chromosome,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,   # kept for API compat — not used in CV mode
        y_val: pd.Series = None,
    ) -> float:
        """
        Evaluate via k-fold CV. Returns mean fitness across folds.
        Returns float('-inf') on any exception.
        """
        try:
            pp_genes, model_genes = _split_genes(chromosome.genes)
            fold_scores = []

            cv = self._build_cv(y_train)

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                X_fold_train = X_train.iloc[train_idx].reset_index(drop=True)
                y_fold_train = y_train.iloc[train_idx].reset_index(drop=True)
                X_fold_val   = X_train.iloc[val_idx].reset_index(drop=True)
                y_fold_val   = y_train.iloc[val_idx].reset_index(drop=True)

                # Fresh preprocessor per fold — zero leakage
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

            # Reject chromosome if any fold failed completely
            valid = [s for s in fold_scores if s != float("-inf")]
            if not valid:
                chromosome.fitness = float("-inf")
                return float("-inf")

            fitness = float(np.mean(valid))
            fitness_std = float(np.std(valid)) if len(valid) > 1 else 0.0

            chromosome.fitness = fitness
            chromosome.fitness_std = fitness_std

            # Store preprocessing gene config on chromosome for lineage tracking
            chromosome._pp_genes = pp_genes  # noqa: SLF001  # intentional dynamic attr

            log.info(
                "Chromosome %s | CV fitness=%.6f ± %.6f | genes=%s",
                chromosome.id,
                fitness,
                fitness_std,
                {**pp_genes, **model_genes},
            )
            return fitness

        except Exception as exc:
            log.warning(
                "Chromosome %s failed: %s\n%s",
                chromosome.id, exc, traceback.format_exc(),
            )
            chromosome.fitness = float("-inf")
            return float("-inf")

    # ------------------------------------------------------------------

    def _build_cv(self, y: pd.Series):
        """Stratified for classification, regular KFold for regression."""
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
