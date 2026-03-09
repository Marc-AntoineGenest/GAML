"""Genetic AutoML — top-level package."""
from genetic_automl.pipeline import AutoMLPipeline
from genetic_automl.config import PipelineConfig, GeneticConfig, AutoMLConfig, DataConfig, ReportConfig
from genetic_automl.core.problem import ProblemType

__all__ = [
    "AutoMLPipeline",
    "PipelineConfig",
    "GeneticConfig",
    "AutoMLConfig",
    "DataConfig",
    "ReportConfig",
    "ProblemType",
]
