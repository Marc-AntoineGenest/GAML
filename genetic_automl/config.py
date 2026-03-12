"""
Configuration dataclasses for GAML.

All tuneable settings live here. Construct them directly in Python or use
load_config() to populate them from gaml_config.yaml.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from genetic_automl.core.problem import ProblemType


@dataclass
class GeneticConfig:
    """Genetic algorithm settings."""

    population_size: int = 20
    """Chromosomes (pipeline configs) evaluated per generation."""

    generations: int = 15
    """Maximum number of evolution cycles."""

    mutation_rate: float = 0.2
    """Probability that any single gene changes value during reproduction."""

    crossover_rate: float = 0.7
    """Probability that two parents recombine instead of cloning."""

    crossover_type: str = "uniform"
    """
    Crossover operator. Options:
      uniform      — each gene drawn independently from either parent (p=0.5). Default.
      single_point — genes split at one random cut point.
    """

    elite_ratio: float = 0.1
    """Fraction of top individuals preserved unchanged each generation."""

    tournament_size: int = 3
    """Candidates compared per tournament selection draw."""

    early_stopping_rounds: int = 5
    """Stop if best fitness does not improve for this many consecutive generations."""

    n_cv_folds: int = 3
    """CV folds per chromosome evaluation. 3 balances quality and speed."""

    # Warm-start
    warm_start: bool = True
    """Seed generation 0 with archetype configs and halving survivors."""

    warm_start_n_seeds: int = 3
    """Number of archetype configs injected into generation 0 (max 3)."""

    warm_start_halving_pool_ratio: float = 2.0
    """Pool size = ratio × population_size. Set 0 to disable halving pre-screen."""

    warm_start_halving_keep_ratio: float = 0.5
    """Fraction of the halving pool kept as generation 0 survivors."""

    # Diversity
    diversity_threshold: float = 0.15
    """Mean Hamming distance below which diversity injection fires."""

    diversity_injection_ratio: float = 0.2
    """Fraction of worst individuals replaced on diversity injection."""

    # Adaptive mutation
    adaptive_mutation: bool = True
    """Boost mutation rate on stagnation; decay back on improvement."""

    adaptive_mutation_stagnation_rounds: int = 3
    """No-improvement generations required to trigger a mutation boost."""

    adaptive_mutation_boost_factor: float = 2.5
    """Multiply base mutation_rate by this factor when boosting."""

    adaptive_mutation_decay: float = 0.85
    """Per-generation decay coefficient back toward the base rate after a boost."""

    # Fitness
    fitness_std_penalty: float = 0.5
    """
    Stability penalty coefficient: fitness = mean_cv - penalty * std_cv.
    0.0 = pure mean CV score. Increase to favour consistent pipelines.
    """

    # Parallelism
    n_jobs: int = 1
    """
    Parallel workers for chromosome evaluation.
    1 = sequential (default, safe for all backends).
    -1 = all CPU cores. Use with the sklearn backend only — AutoGluon
    manages its own thread pool and can oversubscribe when n_jobs != 1.
    """

    random_seed: int = 42


@dataclass
class AutoMLConfig:
    """AutoML backend settings."""

    backend: str = "autogluon"
    """Backend to use. Options: 'autogluon', 'sklearn'."""

    time_limit_per_eval: int = 60
    """Wall-clock seconds allowed per individual fitness evaluation."""

    autogluon_presets: str = "medium_quality"
    """AutoGluon presets string (ignored for other backends)."""

    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Additional kwargs forwarded verbatim to the backend constructor."""


@dataclass
class DataConfig:
    """Data split settings."""

    backend: str = "pandas"
    """Data loading backend. Currently only 'pandas' is supported."""

    test_size: float = 0.15
    """Fraction of total data locked as the final test set (never seen by the GA)."""

    val_size: float = 0.2
    """Fraction of dev data used as validation during the final refit."""

    stratify: bool = True
    """Stratify train/test splits on the label column (classification only)."""

    random_seed: int = 42


@dataclass
class ReportConfig:
    """Reporting settings."""

    output_dir: str = "reports"
    """Directory where HTML reports and JSON run summaries are written."""

    mlflow_tracking_uri: str = "mlflow_runs"
    """Local MLflow tracking store directory. Set to None to disable MLflow."""

    open_html_on_finish: bool = False
    """Open the HTML report in the default browser when the run completes."""


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    problem_type: ProblemType = ProblemType.CLASSIFICATION
    """Task type: CLASSIFICATION, REGRESSION, or MULTI_OBJECTIVE."""

    target_column: str = "target"
    """Name of the target column in the input DataFrame."""

    objectives: Optional[List[str]] = None
    """For MULTI_OBJECTIVE: list of target column names."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    """Short unique identifier for this run (auto-generated)."""

    run_name: Optional[str] = None
    """Human-readable run name shown in reports and MLflow. Auto-generated if None."""

    genetic: GeneticConfig = field(default_factory=GeneticConfig)
    automl: AutoMLConfig = field(default_factory=AutoMLConfig)
    data: DataConfig = field(default_factory=DataConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    def __post_init__(self) -> None:
        if self.run_name is None:
            self.run_name = f"{self.problem_type.value}_{self.run_id}"
        if self.problem_type == ProblemType.MULTI_OBJECTIVE and not self.objectives:
            raise ValueError(
                "ProblemType.MULTI_OBJECTIVE requires 'objectives' to be set."
            )
