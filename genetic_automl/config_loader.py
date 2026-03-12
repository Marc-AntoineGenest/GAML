"""
GAML YAML loader
----------------
Reads ``gaml_config.yaml`` (or any compatible YAML file) and returns a fully
constructed :class:`PipelineConfig` together with the gene-space overrides
that reflect the user's chosen search spaces.

Typical usage
-------------
::

    from genetic_automl import load_config, AutoMLPipeline

    config, gene_overrides = load_config("gaml_config.yaml")
    pipeline = AutoMLPipeline(config, gene_space_overrides=gene_overrides)
    pipeline.fit(df)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from genetic_automl.config import (
    AutoMLConfig,
    DataConfig,
    GeneticConfig,
    PipelineConfig,
    ReportConfig,
)
from genetic_automl.core.problem import ProblemType
from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)

# Gene names that live under preprocessing_search_space in the YAML
_PREPROCESSING_GENE_NAMES = {
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

# Gene names that live under sklearn_search_space / autogluon_search_space
_MODEL_GENE_NAMES = {
    "sklearn": {"n_estimators", "max_depth", "learning_rate"},
    "autogluon": {"presets", "time_limit"},
}


def load_config(
    path: str = "gaml_config.yaml",
) -> Tuple[PipelineConfig, Dict[str, List[Any]]]:
    """
    Parse a GAML YAML configuration file and return a ``(PipelineConfig,
    gene_overrides)`` tuple.

    Parameters
    ----------
    path : str
        Path to the YAML file. Defaults to ``gaml_config.yaml`` in the
        current working directory.

    Returns
    -------
    config : PipelineConfig
        Fully populated pipeline configuration ready to pass to
        :class:`~genetic_automl.pipeline.AutoMLPipeline`.
    gene_overrides : dict
        Flat dict of ``{gene_name: [candidate_values]}`` combining
        preprocessing and model search spaces. Pass this to
        ``AutoMLPipeline(config, gene_space_overrides=gene_overrides)``.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist at *path*.
    ValueError
        If any search-space entry is not a non-empty list.
    """
    try:
        import yaml  # PyYAML
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load gaml_config.yaml. "
            "Install it with:  pip install pyyaml"
        ) from exc

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"GAML config file not found: '{path}'. "
            "Make sure gaml_config.yaml is in your working directory, "
            "or pass the correct path to load_config()."
        )

    with open(path, "r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    log.info("Loading GAML config from '%s'", path)

    # ------------------------------------------------------------------
    # Run section
    # ------------------------------------------------------------------
    run_cfg = raw.get("run", {})
    problem_type_str = str(run_cfg.get("problem_type", "classification")).lower()
    problem_type = _parse_problem_type(problem_type_str)
    target_column = str(run_cfg.get("target_column", "target"))
    backend = str(run_cfg.get("backend", "sklearn")).lower()
    metric: Optional[str] = run_cfg.get("metric") or None
    run_name: Optional[str] = run_cfg.get("name") or None

    # ------------------------------------------------------------------
    # Data section
    # ------------------------------------------------------------------
    data_cfg = raw.get("data", {})
    data = DataConfig(
        test_size=float(data_cfg.get("test_size", 0.15)),
        val_size=float(data_cfg.get("val_size", 0.20)),
        stratify=bool(data_cfg.get("stratify", True)),
        random_seed=int(data_cfg.get("random_seed", 42)),
    )

    # ------------------------------------------------------------------
    # Genetic section
    # ------------------------------------------------------------------
    gen_cfg = raw.get("genetic", {})
    genetic = GeneticConfig(
        population_size=int(gen_cfg.get("population_size", 20)),
        generations=int(gen_cfg.get("generations", 15)),
        n_cv_folds=int(gen_cfg.get("n_cv_folds", 3)),
        early_stopping_rounds=int(gen_cfg.get("early_stopping_rounds", 5)),
        mutation_rate=float(gen_cfg.get("mutation_rate", 0.20)),
        crossover_rate=float(gen_cfg.get("crossover_rate", 0.70)),
        elite_ratio=float(gen_cfg.get("elite_ratio", 0.10)),
        tournament_size=int(gen_cfg.get("tournament_size", 3)),
        warm_start=bool(gen_cfg.get("warm_start", True)),
        warm_start_n_seeds=int(gen_cfg.get("warm_start_n_seeds", 3)),
        warm_start_halving_pool_ratio=float(gen_cfg.get("warm_start_halving_pool_ratio", 2.0)),
        warm_start_halving_keep_ratio=float(gen_cfg.get("warm_start_halving_keep_ratio", 0.50)),
        diversity_threshold=float(gen_cfg.get("diversity_threshold", 0.15)),
        diversity_injection_ratio=float(gen_cfg.get("diversity_injection_ratio", 0.20)),
        adaptive_mutation=bool(gen_cfg.get("adaptive_mutation", True)),
        adaptive_mutation_stagnation_rounds=int(gen_cfg.get("adaptive_mutation_stagnation_rounds", 3)),
        adaptive_mutation_boost_factor=float(gen_cfg.get("adaptive_mutation_boost_factor", 2.5)),
        adaptive_mutation_decay=float(gen_cfg.get("adaptive_mutation_decay", 0.85)),
        random_seed=int(gen_cfg.get("random_seed", 42)),
    )

    # ------------------------------------------------------------------
    # AutoML backend section
    # ------------------------------------------------------------------
    automl = AutoMLConfig(
        backend=backend,
        autogluon_presets="medium_quality",  # overridden by gene search if autogluon
    )
    if backend == "autogluon":
        ag_cfg = raw.get("autogluon", {})
        automl.time_limit_per_eval = int(ag_cfg.get("time_limit_per_eval", 60))

    # ------------------------------------------------------------------
    # Report section
    # ------------------------------------------------------------------
    rep_cfg = raw.get("report", {})
    report = ReportConfig(
        output_dir=str(rep_cfg.get("output_dir", "reports")),
        mlflow_tracking_uri=str(rep_cfg.get("mlflow_tracking_uri", "mlflow_runs")),
        open_html_on_finish=bool(rep_cfg.get("open_html_on_finish", False)),
    )

    # ------------------------------------------------------------------
    # Build PipelineConfig
    # ------------------------------------------------------------------
    config = PipelineConfig(
        problem_type=problem_type,
        target_column=target_column,
        run_name=run_name,
        genetic=genetic,
        automl=automl,
        data=data,
        report=report,
    )

    # Apply metric override if provided
    if metric:
        config._metric_override = metric  # picked up by AutoMLPipeline

    # ------------------------------------------------------------------
    # Gene space overrides — combine preprocessing + model search spaces
    # ------------------------------------------------------------------
    pp_raw: Dict[str, Any] = raw.get("preprocessing_search_space", {})
    model_key = f"{backend}_search_space"
    model_raw: Dict[str, Any] = raw.get(model_key, {})

    gene_overrides: Dict[str, List[Any]] = {}

    for name, values in pp_raw.items():
        if name not in _PREPROCESSING_GENE_NAMES:
            log.warning("Unknown preprocessing gene '%s' in config — skipping.", name)
            continue
        gene_overrides[name] = _coerce_values(name, values)

    known_model_genes = _MODEL_GENE_NAMES.get(backend, set())
    for name, values in model_raw.items():
        if name not in known_model_genes:
            log.warning("Unknown model gene '%s' for backend '%s' — skipping.", name, backend)
            continue
        gene_overrides[name] = _coerce_values(name, values)

    _validate_gene_overrides(gene_overrides)

    log.info(
        "Config loaded | problem=%s | backend=%s | pop=%d | gens=%d | "
        "search_space_genes=%d",
        problem_type.value, backend,
        genetic.population_size, genetic.generations,
        len(gene_overrides),
    )
    return config, gene_overrides


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_problem_type(value: str) -> ProblemType:
    mapping = {
        "classification": ProblemType.CLASSIFICATION,
        "regression": ProblemType.REGRESSION,
        "multi_objective": ProblemType.MULTI_OBJECTIVE,
    }
    if value not in mapping:
        raise ValueError(
            f"Unknown problem_type '{value}'. "
            f"Options: {list(mapping.keys())}"
        )
    return mapping[value]


def _coerce_values(name: str, raw: Any) -> List[Any]:
    """Ensure the value is a list; convert YAML nulls to Python None."""
    if not isinstance(raw, list):
        raw = [raw]
    # YAML null → Python None (used for correlation_threshold: [null, 0.95])
    return [None if v == "null" else v for v in raw]


def _validate_gene_overrides(overrides: Dict[str, List[Any]]) -> None:
    for name, values in overrides.items():
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(
                f"Search space for gene '{name}' must be a non-empty list. "
                f"Got: {values!r}"
            )
