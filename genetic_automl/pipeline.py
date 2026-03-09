"""
AutoMLPipeline — top-level entry point.

Full execution flow:
  1. DataManager.validate() + split()  →  train / val / test
  2. GeneticEngine.run()               →  evolve preprocessing + model config
     └─ Each chromosome evaluation:
          a. PreprocessingPipeline.fit_transform_train(X_train)
          b. PreprocessingPipeline.transform(X_val)
          c. AutoMLModel.fit(X_train_pp, y_train_pp)
          d. AutoMLModel.score(X_val_pp, y_val)   → fitness
  3. Best chromosome selected
  4. PreprocessingPipeline refit on full training split (best config)
  5. AutoMLModel retrained on preprocessed full training split
  6. Final evaluation on preprocessed test set
  7. MLflow logging + JSON export
  8. HTML report generation
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import pandas as pd

from genetic_automl.automl import build_automl
from genetic_automl.config import PipelineConfig
from genetic_automl.core.base_automl import BaseAutoML
from genetic_automl.core.data import DataManager
from genetic_automl.core.problem import get_default_metric, ProblemType
from genetic_automl.genetic.engine import EvolutionHistory, GeneticEngine
from genetic_automl.genetic.fitness import FitnessEvaluator, _split_genes
from genetic_automl.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline
from genetic_automl.reporting.html_reporter import HTMLReporter
from genetic_automl.reporting.mlflow_logger import MLflowLogger
from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)


class AutoMLPipeline:
    """
    Orchestrates the full Genetic AutoML pipeline.

    Parameters
    ----------
    config : PipelineConfig

    Usage
    -----
    pipeline = AutoMLPipeline(config)
    pipeline.fit(train_df)
    predictions = pipeline.predict(test_df)
    score = pipeline.score(test_df)
    print(pipeline.report_path)
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._data_manager: Optional[DataManager] = None
        self._best_preprocessor: Optional[PreprocessingPipeline] = None
        self._best_model: Optional[BaseAutoML] = None
        self._history: Optional[EvolutionHistory] = None
        self._report_path: Optional[str] = None
        self._final_score: Optional[float] = None
        self._metric_name: str = get_default_metric(config.problem_type)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
    ) -> "AutoMLPipeline":
        """
        Run the full genetic AutoML pipeline.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data including the target column.
        test_df : pd.DataFrame | None
            Optional explicit hold-out set. If None, a random split is used.
        """
        cfg = self.config
        t0 = time.perf_counter()
        log.info("=" * 60)
        log.info("AutoMLPipeline.fit() | run=%s | backend=%s", cfg.run_name, cfg.automl.backend)
        log.info("=" * 60)

        # ── 1. Data ────────────────────────────────────────────────────
        self._data_manager = DataManager(
            target_column=cfg.target_column,
            problem_type=cfg.problem_type,
            test_size=cfg.data.test_size,
            val_size=cfg.data.val_size,
            stratify=cfg.data.stratify,
            random_seed=cfg.data.random_seed,
        )
        validated = self._data_manager.validate(train_df)
        train_split, val_split, test_split = self._data_manager.three_way_split(validated, test_df)

        # GA evolves on train_split using k-fold CV — test_split is NEVER touched here
        X_train = self._data_manager.features(train_split)
        y_train = self._data_manager.labels(train_split)
        X_test  = self._data_manager.features(test_split)
        y_test  = self._data_manager.labels(test_split)

        # ── 2. Genetic search ─────────────────────────────────────────
        evaluator = FitnessEvaluator(
            problem_type=cfg.problem_type,
            target_column=cfg.target_column,
            backend=cfg.automl.backend,
            metric=self._metric_name,
            n_folds=cfg.genetic.n_cv_folds,
            random_seed=cfg.genetic.random_seed,
        )
        engine = GeneticEngine(
            genetic_config=cfg.genetic,
            evaluator=evaluator,
            backend=cfg.automl.backend,
        )
        # GA uses k-fold CV on X_train only — X_test is never passed in
        best_chrom = engine.run(X_train, y_train)
        self._history = engine.history

        # ── 3. Refit best preprocessing on full training split ────────
        log.info("Refitting best preprocessing config on full training split…")
        pp_genes, model_genes = _split_genes(best_chrom.genes)
        pp_config = PreprocessingConfig.from_genes(pp_genes)
        self._best_preprocessor = PreprocessingPipeline(
            config=pp_config,
            problem_type=cfg.problem_type,
            random_seed=cfg.genetic.random_seed,
        )
        X_train_pp, y_train_pp = self._best_preprocessor.fit_transform_train(X_train, y_train)
        X_test_pp = self._best_preprocessor.transform(X_test)

        # ── 4. Retrain best model on preprocessed full training set ───
        log.info("Retraining best model config on preprocessed training data…")
        self._best_model = build_automl(
            backend=cfg.automl.backend,
            problem_type=cfg.problem_type,
            target_column=cfg.target_column,
            random_seed=cfg.genetic.random_seed,
            time_limit=cfg.automl.time_limit_per_eval * 2,
            **{k: v for k, v in model_genes.items() if v is not None},
        )
        self._best_model.fit(X_train_pp, y_train_pp, X_test_pp, y_test)

        # ── 5. Final score on preprocessed test set ───────────────────
        raw = self._best_model.score(X_test_pp, y_test, metric=self._metric_name)
        self._final_score = raw
        log.info(
            "Final test %s: %.6f | elapsed: %.1fs",
            self._metric_name, raw, time.perf_counter() - t0,
        )

        # ── 6. MLflow logging ─────────────────────────────────────────
        os.makedirs(cfg.report.output_dir, exist_ok=True)
        mlflow_logger = MLflowLogger(
            tracking_uri=cfg.report.mlflow_tracking_uri,
            experiment_name=cfg.run_name,
        )
        mlflow_logger.log_evolution(self._history, cfg.run_name)
        json_path = os.path.join(cfg.report.output_dir, f"run_{cfg.run_id}.json")
        mlflow_logger.save_json(self._history, json_path)

        # ── 7. HTML report ────────────────────────────────────────────
        reporter = HTMLReporter(output_dir=cfg.report.output_dir)
        self._report_path = reporter.generate(
            config=cfg,
            history=self._history,
            final_test_score=self._final_score,
            final_metric_name=self._metric_name,
            preprocessing_summary=self._best_preprocessor.summary(),
            diversity_summary=engine.diversity_summary(),
            open_browser=cfg.report.open_html_on_finish,
        )
        log.info("Report: %s", self._report_path)
        return self

    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess then predict."""
        self._check_fitted()
        X = self._drop_target_if_present(df)
        X_pp = self._best_preprocessor.transform(X)
        return self._best_model.predict(X_pp)

    def predict_proba(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Preprocess then predict class probabilities (classification only)."""
        self._check_fitted()
        X = self._drop_target_if_present(df)
        X_pp = self._best_preprocessor.transform(X)
        return self._best_model.predict_proba(X_pp)

    def score(self, df: pd.DataFrame, metric: Optional[str] = None) -> float:
        """Preprocess df then evaluate the final model."""
        self._check_fitted()
        X = self._data_manager.features(df)
        y = self._data_manager.labels(df)
        X_pp = self._best_preprocessor.transform(X)
        return self._best_model.score(X_pp, y, metric=metric)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def best_model(self) -> Optional[BaseAutoML]:
        return self._best_model

    @property
    def best_preprocessor(self) -> Optional[PreprocessingPipeline]:
        return self._best_preprocessor

    @property
    def history(self) -> Optional[EvolutionHistory]:
        return self._history

    @property
    def report_path(self) -> Optional[str]:
        return self._report_path

    @property
    def final_score(self) -> Optional[float]:
        return self._final_score

    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._best_model is None:
            raise RuntimeError("Pipeline has not been fitted. Call fit() first.")

    def _drop_target_if_present(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.target_column in df.columns:
            return df.drop(columns=[self.config.target_column])
        return df
