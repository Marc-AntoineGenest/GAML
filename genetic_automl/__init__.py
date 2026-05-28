"""Genetic AutoML -- top-level package."""

__version__ = "0.1.0"
from genetic_automl.pipeline import AutoMLPipeline
from genetic_automl.config import PipelineConfig, GeneticConfig, AutoMLConfig, DataConfig, ReportConfig, EnsembleConfig
from genetic_automl.config_loader import load_config
from genetic_automl.core.problem import ProblemType

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
