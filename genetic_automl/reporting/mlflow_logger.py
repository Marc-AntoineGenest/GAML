"""
MLflow logger — local tracking only.

Designed as a standalone class so it can be replaced with a remote
MLflow server, W&B, or any other tracker without touching the pipeline.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from genetic_automl.genetic.engine import EvolutionHistory
from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)


class MLflowLogger:
    """
    Logs GA runs to a local MLflow tracking store.

    If MLflow is not installed the logger degrades gracefully to a
    JSON-based fallback so the pipeline never breaks.
    """

    def __init__(self, tracking_uri: str = "mlflow_runs", experiment_name: str = "genetic_automl") -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._client = None
        self._run = None
        self._run_id: Optional[str] = None
        self._available = self._try_import()

    # ------------------------------------------------------------------

    def _try_import(self) -> bool:
        try:
            import mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._mlflow = mlflow
            log.info("MLflow available. Tracking URI: %s", self.tracking_uri)
            return True
        except ImportError:
            log.warning("MLflow not installed. Falling back to JSON logging.")
            return False

    def start_run(self, run_name: str) -> None:
        if self._available:
            self._run = self._mlflow.start_run(run_name=run_name)
            self._run_id = self._run.info.run_id
            log.info("MLflow run started: %s", self._run_id)

    def log_params(self, params: Dict[str, Any]) -> None:
        if self._available and self._run:
            self._mlflow.log_params(
                {k: str(v) for k, v in params.items()}
            )

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        if self._available and self._run:
            kwargs = {} if step is None else {"step": step}
            self._mlflow.log_metric(key, value, **kwargs)

    def log_artifact(self, path: str) -> None:
        if self._available and self._run and os.path.exists(path):
            self._mlflow.log_artifact(path)

    def end_run(self) -> None:
        if self._available and self._run:
            self._mlflow.end_run()
            log.info("MLflow run ended: %s", self._run_id)

    # ------------------------------------------------------------------
    # Convenience: log full evolution history
    # ------------------------------------------------------------------

    def log_evolution(self, history: EvolutionHistory, run_name: str) -> None:
        """Log all generation stats and the best chromosome."""
        self.start_run(run_name)
        try:
            if history.best:
                self.log_params(history.best.genes)
                self.log_metric("best_fitness", history.best.fitness)

            for gen_stats in history.generations:
                step = gen_stats.generation
                self.log_metric("gen_best_fitness", gen_stats.best_fitness, step=step)
                self.log_metric("gen_mean_fitness", gen_stats.mean_fitness, step=step)
        finally:
            self.end_run()

    # ------------------------------------------------------------------
    # JSON fallback
    # ------------------------------------------------------------------

    def save_json(self, history: EvolutionHistory, path: str) -> None:
        """Always available fallback: write evolution history to JSON."""
        data = {
            "best": history.best.as_dict() if history.best else None,
            "generations": [
                {
                    "generation": g.generation,
                    "best_fitness": g.best_fitness,
                    "mean_fitness": g.mean_fitness,
                    "worst_fitness": g.worst_fitness,
                    "elapsed_seconds": g.elapsed_seconds,
                }
                for g in history.generations
            ],
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        log.info("Evolution history saved to %s", path)
