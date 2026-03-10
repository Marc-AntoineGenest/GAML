"""
Central configuration dataclasses for the Genetic AutoML framework.
All tuneable knobs live here — easy to serialize to JSON/YAML.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from genetic_automl.core.problem import ProblemType


# ---------------------------------------------------------------------------
# Genetic algorithm hyper-parameters
# ---------------------------------------------------------------------------

@dataclass
class GeneticConfig:
    population_size: int = 20
    """Number of chromosomes (pipeline configs) per generation."""

    generations: int = 15
    """Maximum number of generations to evolve."""

    mutation_rate: float = 0.2
    """Probability that any single gene mutates during reproduction."""

    crossover_rate: float = 0.7
    """Probability that two parents produce offspring via crossover."""

    elite_ratio: float = 0.1
    """Fraction of top individuals preserved unchanged each generation."""

    tournament_size: int = 3
    """Number of candidates compared in tournament selection."""

    early_stopping_rounds: int = 5
    """Stop if best fitness does not improve for this many generations."""

    n_cv_folds: int = 3
    """Number of cross-validation folds per chromosome evaluation. 3 balances quality vs speed."""

    # --- Warm-start ---
    warm_start: bool = True
    """Whether to use warm-start population seeding (default seeds + halving pre-screen)."""

    warm_start_n_seeds: int = 3
    """Number of default archetype configs injected into gen-0 (max 3)."""

    warm_start_halving_pool_ratio: float = 2.0
    """Random pool size = ratio × population_size. Pre-screened with 1-fold CV. 0 = disable."""
    warm_start_halving_keep_ratio: float = 0.5
    """Fraction of the halving pool kept as survivors."""

    # --- Diversity ---
    diversity_threshold: float = 0.15
    """Mean Hamming distance below which diversity injection is triggered."""
    diversity_injection_ratio: float = 0.2
    """Fraction of worst individuals replaced on injection."""
    # --- Adaptive mutation ---
    adaptive_mutation: bool = True
    """Boost mutation rate when stagnation is detected, decay back on improvement."""
    adaptive_mutation_stagnation_rounds: int = 3
    """No-improvement generations that trigger mutation boost."""
    adaptive_mutation_boost_factor: float = 2.5
    """Multiply base mutation_rate by this factor on stagnation."""
    adaptive_mutation_decay: float = 0.85
    """Per-generation decay toward base_rate after a boost."""
    random_seed: int = 42


# ---------------------------------------------------------------------------
# AutoML back-end options
# ---------------------------------------------------------------------------

@dataclass
class AutoMLConfig:
    backend: str = "autogluon"
    """Which AutoML backend to use. Options: 'autogluon', 'sklearn'."""

    time_limit_per_eval: int = 60
    """Wall-clock seconds allowed per individual fitness evaluation."""

    autogluon_presets: str = "medium_quality"
    """AutoGluon presets string (ignored for other backends)."""

    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Additional kwargs forwarded verbatim to the backend constructor."""


# ---------------------------------------------------------------------------
# Data options
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    backend: str = "pandas"
    """Data loading backend. Currently only 'pandas' is implemented."""

    test_size: float = 0.15
    """Fraction of total data locked as final test set (never touches GA loop)."""

    val_size: float = 0.2
    """Fraction of dev data used as validation in final refit. GA itself uses k-fold CV."""

    stratify: bool = True
    """Stratify train/test split on the label (classification only)."""

    random_seed: int = 42


# ---------------------------------------------------------------------------
# Reporting options
# ---------------------------------------------------------------------------

@dataclass
class ReportConfig:
    output_dir: str = "reports"
    """Directory where HTML reports and MLflow artifacts are stored."""

    mlflow_tracking_uri: str = "mlflow_runs"
    """Local directory used as the MLflow tracking store."""

    open_html_on_finish: bool = False
    """Automatically open the HTML report in the browser when done."""


# ---------------------------------------------------------------------------
# Top-level pipeline config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    problem_type: ProblemType = ProblemType.CLASSIFICATION
    """Classification, Regression, or MultiObjective."""

    target_column: str = "target"
    """Name of the target column in the DataFrame."""

    objectives: Optional[List[str]] = None
    """For multi-objective: list of target column names."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    """Short unique identifier for this run (auto-generated)."""

    run_name: Optional[str] = None
    """Human-friendly run name. Defaults to '<problem_type>_<run_id>'."""

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
