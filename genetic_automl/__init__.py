"""Genetic AutoML -- top-level package."""
from __future__ import annotations

__version__ = "0.1.0"
from genetic_automl.config import (
    AutoMLConfig,
    DataConfig,
    EnsembleConfig,
    GeneticConfig,
    PipelineConfig,
    ReportConfig,
)
from genetic_automl.config_loader import load_config
from genetic_automl.core.problem import ProblemType
from genetic_automl.pipeline import AutoMLPipeline

__all__ = [
    "AutoMLPipeline",   # includes .save() and .load()
    "PipelineConfig",
    "GeneticConfig",
    "AutoMLConfig",
    "DataConfig",
    "ReportConfig",
    "ProblemType",
    "load_config",
    "EnsembleConfig",
    "__version__",
]
