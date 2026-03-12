"""
AutoMLPipeline — top-level entry point.

Execution flow:
  1. DataManager: validate and 3-way split into train / val / test
  2. GeneticEngine: evolve preprocessing + model config via k-fold CV on train
  3. Refit best preprocessor on train + val
  4. Retrain best model on preprocessed train + val
  5. Evaluate on the locked test set
  6. Log to MLflow and export JSON
  7. Generate HTML report
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import pandas as pd
from joblib import dump as _jdump, load as _jload

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
        Build with load_config("gaml_config.yaml") or construct manually.
    gene_space_overrides : dict, optional
        {gene_name: [candidate_values]} overrides returned by load_config().

    Examples
    --------
    YAML-driven (recommended)::

        from genetic_automl import load_config, AutoMLPipeline
        config, overrides = load_config("gaml_config.yaml")
        pipeline = AutoMLPipeline(config, overrides)
        pipeline.fit(df)

    Code-only::

        pipeline = AutoMLPipeline(PipelineConfig(...))
        pipeline.fit(df)
    """

    def __init__(
        self,
        config: PipelineConfig,
        gene_space_overrides: Optional[dict] = None,
    ) -> None:
        self.config = config
        self._gene_space_overrides: dict = gene_space_overrides or {}
        self._data_manager: Optional[DataManager] = None
        self._best_preprocessor: Optional[PreprocessingPipeline] = None
        self._best_model: Optional[BaseAutoML] = None
        self._history: Optional[EvolutionHistory] = None
        self._report_path: Optional[str] = None
        self._final_score: Optional[float] = None
        metric_override = getattr(config, "_metric_override", None)
        self._metric_name: str = metric_override or get_default_metric(config.problem_type)

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
            Optional hold-out set. If None, carved from train_df automatically.
        """
        cfg = self.config
        t0 = time.perf_counter()
        log.info("=" * 60)
        log.info("AutoMLPipeline.fit() | run=%s | backend=%s", cfg.run_name, cfg.automl.backend)
        log.info("=" * 60)

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

        X_train = self._data_manager.features(train_split)
        y_train = self._data_manager.labels(train_split)
        X_val   = self._data_manager.features(val_split)
        y_val   = self._data_manager.labels(val_split)
        X_test  = self._data_manager.features(test_split)
        y_test  = self._data_manager.labels(test_split)

        evaluator = FitnessEvaluator(
            problem_type=cfg.problem_type,
            target_column=cfg.target_column,
            backend=cfg.automl.backend,
            metric=self._metric_name,
            n_folds=cfg.genetic.n_cv_folds,
            random_seed=cfg.genetic.random_seed,
            fitness_std_penalty=cfg.genetic.fitness_std_penalty,
        )
        engine = GeneticEngine(
            genetic_config=cfg.genetic,
            evaluator=evaluator,
            backend=cfg.automl.backend,
            gene_space_overrides=self._gene_space_overrides,
        )
        best_chrom = engine.run(X_train, y_train)
        self._history = engine.history

        log.info("Refitting best preprocessing on train + val")
        X_dev = pd.concat([X_train, X_val], ignore_index=True)
        y_dev = pd.concat([y_train, y_val], ignore_index=True)

        pp_genes, model_genes = _split_genes(best_chrom.genes)
        pp_config = PreprocessingConfig.from_genes(pp_genes)
        self._best_preprocessor = PreprocessingPipeline(
            config=pp_config,
            problem_type=cfg.problem_type,
            random_seed=cfg.genetic.random_seed,
        )
        X_dev_pp, y_dev_pp = self._best_preprocessor.fit_transform_train(X_dev, y_dev)
        X_test_pp = self._best_preprocessor.transform(X_test)

        log.info("Retraining best model on preprocessed train + val")
        self._best_model = build_automl(
            backend=cfg.automl.backend,
            problem_type=cfg.problem_type,
            target_column=cfg.target_column,
            random_seed=cfg.genetic.random_seed,
            time_limit=cfg.automl.time_limit_per_eval * 2,
            **{k: v for k, v in model_genes.items() if v is not None},
        )
        self._best_model.fit(X_dev_pp, y_dev_pp, None, None)

        self._final_score = self._best_model.score(X_test_pp, y_test, metric=self._metric_name)
        log.info(
            "Final test %s: %.6f | elapsed: %.1fs",
            self._metric_name, self._final_score, time.perf_counter() - t0,
        )

        os.makedirs(cfg.report.output_dir, exist_ok=True)
        mlflow_logger = MLflowLogger(
            tracking_uri=cfg.report.mlflow_tracking_uri,
            experiment_name=cfg.run_name,
        )
        mlflow_logger.log_evolution(self._history, cfg.run_name)
        json_path = os.path.join(cfg.report.output_dir, f"run_{cfg.run_id}.json")
        mlflow_logger.save_json(self._history, json_path)

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

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess and return predictions."""
        self._check_fitted()
        X = self._drop_target_if_present(df)
        return self._best_model.predict(self._best_preprocessor.transform(X))

    def predict_proba(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Preprocess and return class probabilities (classification only)."""
        self._check_fitted()
        X = self._drop_target_if_present(df)
        return self._best_model.predict_proba(self._best_preprocessor.transform(X))

    def score(self, df: pd.DataFrame, metric: Optional[str] = None) -> float:
        """Preprocess and evaluate the final model on df."""
        self._check_fitted()
        X = self._data_manager.features(df)
        y = self._data_manager.labels(df)
        return self._best_model.score(self._best_preprocessor.transform(X), y, metric=metric)

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

    def save(self, path: str) -> str:
        """
        Persist the fitted pipeline to disk using joblib.

        Parameters
        ----------
        path : str
            File path to write. The .joblib extension is recommended.

        Returns
        -------
        str
            Resolved absolute path of the saved file.

        Raises
        ------
        RuntimeError
            If the pipeline has not been fitted yet.
        """
        self._check_fitted()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "preprocessor": self._best_preprocessor,
            "model": self._best_model,
            "config": self.config,
            "metric_name": self._metric_name,
            "final_score": self._final_score,
        }
        _jdump(payload, path)
        log.info("Pipeline saved to %s", os.path.abspath(path))
        return os.path.abspath(path)

    @classmethod
    def load(cls, path: str) -> "AutoMLPipeline":
        """
        Reload a pipeline saved with save(). Ready for predict() immediately
        — no re-fitting required.

        Parameters
        ----------
        path : str
            Path to a file previously written by save().
        """
        payload = _jload(path)
        instance = cls.__new__(cls)
        instance.config = payload["config"]
        instance._gene_space_overrides = {}
        instance._data_manager = None
        instance._best_preprocessor = payload["preprocessor"]
        instance._best_model = payload["model"]
        instance._history = None
        instance._report_path = None
        instance._final_score = payload.get("final_score")
        instance._metric_name = payload["metric_name"]
        log.info("Pipeline loaded from %s", os.path.abspath(path))
        return instance

    def _check_fitted(self) -> None:
        if self._best_model is None:
            raise RuntimeError("Pipeline has not been fitted. Call fit() first.")

    def _drop_target_if_present(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.target_column in df.columns:
            return df.drop(columns=[self.config.target_column])
        return df
