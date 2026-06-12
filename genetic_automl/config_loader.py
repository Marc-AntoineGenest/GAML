"""
GAML YAML configuration loader.

Reads gaml_config.yaml and returns a (PipelineConfig, gene_overrides) tuple
ready to pass directly to AutoMLPipeline.

Usage::

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
    CalibrationConfig,
    DataConfig,
    EnsembleConfig,
    GeneticConfig,
    OptunaConfig,
    PipelineConfig,
    ReportConfig,
)
from genetic_automl.core.problem import ProblemType
from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)

_PREPROCESSING_GENE_NAMES = {
    "numeric_imputer", "outlier_method", "outlier_threshold", "outlier_action",
    "correlation_threshold", "categorical_encoder", "distribution_transform",
    "scaler", "missing_indicator", "feature_selection_method",
    "feature_selection_k", "imbalance_method",
}

_MODEL_GENE_NAMES = {
    "sklearn": {"model_type", "n_estimators", "max_depth", "learning_rate", "min_samples_leaf", "max_features"},
    "_preprocessing": {"feature_engineering", "max_interaction_features"},
    "autogluon": {"presets", "time_limit"},
}


def load_config(
    path: str = "gaml_config.yaml",
) -> Tuple[PipelineConfig, Dict[str, List[Any]]]:
    """
    Parse a GAML YAML config file and return a (PipelineConfig, gene_overrides) tuple.

    Parameters
    ----------
    path : str
        Path to the YAML file. Defaults to gaml_config.yaml in the current directory.

    Returns
    -------
    config : PipelineConfig
    gene_overrides : dict
        {gene_name: [candidate_values]} for all search space entries in the YAML.
        Pass to AutoMLPipeline(config, gene_space_overrides=gene_overrides).

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If any search-space entry is not a non-empty list.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to use gaml_config.yaml. "
            "Install it with:  pip install pyyaml"
        ) from exc

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config file not found: '{path}'. "
            "Ensure gaml_config.yaml is in your working directory, "
            "or pass the correct path to load_config()."
        )

    with open(path, "r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    log.info("Loading GAML config from '%s'", path)

    run_cfg = raw.get("run", {})
    problem_type_str = str(run_cfg.get("problem_type", "classification")).lower()
    problem_type = _parse_problem_type(problem_type_str)
    target_column = str(run_cfg.get("target_column", "target"))
    backend = str(run_cfg.get("backend", "sklearn")).lower()
    metric: Optional[str] = run_cfg.get("metric") or None
    run_name: Optional[str] = run_cfg.get("name") or None

    data_cfg = raw.get("data", {})
    data = DataConfig(
        backend=str(data_cfg.get("backend", "pandas")),
        test_size=float(data_cfg.get("test_size", 0.15)),
        val_size=float(data_cfg.get("val_size", 0.20)),
        stratify=bool(data_cfg.get("stratify", True)),
        random_seed=int(data_cfg.get("random_seed", 42)),
    )

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
        fitness_std_penalty=float(gen_cfg.get("fitness_std_penalty", 0.5)),
        crossover_type=str(gen_cfg.get("crossover_type", "uniform")),
        n_jobs=int(gen_cfg.get("n_jobs", 1)),
        surrogate_enabled=bool(gen_cfg.get("surrogate_enabled", True)),
        surrogate_model_type=str(gen_cfg.get("surrogate_model_type", "rf")),
        surrogate_min_samples=int(gen_cfg.get("surrogate_min_samples", 10)),
        surrogate_uncertainty_threshold=float(
            gen_cfg.get("surrogate_uncertainty_threshold", 0.05)
        ),
        asha_enabled=bool(gen_cfg.get("asha_enabled", True)),
        asha_min_folds_before_prune=int(gen_cfg.get("asha_min_folds_before_prune", 1)),
        asha_prune_margin=float(gen_cfg.get("asha_prune_margin", 0.0)),
        checkpoint_dir=gen_cfg.get("checkpoint_dir") or None,
        checkpoint_every=int(gen_cfg.get("checkpoint_every", 1)),
        resume_from_checkpoint=gen_cfg.get("resume_from_checkpoint") or None,
        island_model=bool(gen_cfg.get("island_model", False)),
        n_islands=int(gen_cfg.get("n_islands", 4)),
        migration_interval=int(gen_cfg.get("migration_interval", 3)),
        migration_size=int(gen_cfg.get("migration_size", 2)),
        n_island_jobs=int(gen_cfg.get("n_island_jobs", 1)),
        cv_strategy=str(gen_cfg.get("cv_strategy", "stratified")),
        group_column=gen_cfg.get("group_column") or None,
        nsga2_enabled=bool(gen_cfg.get("nsga2_enabled", False)),
        nsga2_objectives=gen_cfg.get("nsga2_objectives") or None,
        random_seed=int(gen_cfg.get("random_seed", 42)),
    )

    automl = AutoMLConfig(backend=backend, autogluon_presets="medium_quality")
    if backend == "autogluon":
        ag_cfg = raw.get("autogluon", {})
        automl.time_limit_per_eval = int(ag_cfg.get("time_limit_per_eval", 60))

    ens_cfg = raw.get("ensemble", {})
    automl.ensemble = EnsembleConfig(
        enabled=bool(ens_cfg.get("enabled", True)),
        top_k=int(ens_cfg.get("top_k", 3)),
        weight_by_fitness=bool(ens_cfg.get("weight_by_fitness", True)),
    )

    opt_raw = raw.get("optuna", {})
    timeout_val = opt_raw.get("timeout", None)
    automl.optuna = OptunaConfig(
        enabled=bool(opt_raw.get("enabled", False)),
        n_trials=int(opt_raw.get("n_trials", 30)),
        timeout=float(timeout_val) if timeout_val is not None else None,
        use_cv=bool(opt_raw.get("use_cv", False)),
        n_cv_folds=int(opt_raw.get("n_cv_folds", 3)),
        verbose=bool(opt_raw.get("verbose", False)),
    )

    cal_raw = raw.get("calibration", {})
    automl.calibration = CalibrationConfig(
        enabled=bool(cal_raw.get("enabled", False)),
        method=str(cal_raw.get("method", "sigmoid")),
        cv=int(cal_raw.get("cv", 5)),
    )

    rep_cfg = raw.get("report", {})
    report = ReportConfig(
        output_dir=str(rep_cfg.get("output_dir", "reports")),
        mlflow_tracking_uri=rep_cfg.get("mlflow_tracking_uri", "mlflow_runs") or None,
        open_html_on_finish=bool(rep_cfg.get("open_html_on_finish", False)),
        shap_enabled=bool(rep_cfg.get("shap_enabled", True)),
        shap_max_samples=int(rep_cfg.get("shap_max_samples", 200)),
        drift_enabled=bool(rep_cfg.get("drift_enabled", False)),
        drift_pvalue_threshold=float(rep_cfg.get("drift_pvalue_threshold", 0.05)),
        drift_psi_threshold=float(rep_cfg.get("drift_psi_threshold", 0.20)),
    )

    config = PipelineConfig(
        problem_type=problem_type,
        target_column=target_column,
        run_name=run_name,
        genetic=genetic,
        automl=automl,
        data=data,
        report=report,
    )

    if metric:
        config._metric_override = metric

    pp_raw: Dict[str, Any] = raw.get("preprocessing_search_space", {})
    model_raw: Dict[str, Any] = raw.get(f"{backend}_search_space", {})

    gene_overrides: Dict[str, List[Any]] = {}

    for name, values in pp_raw.items():
        if name not in _PREPROCESSING_GENE_NAMES:
            log.warning("Unknown preprocessing gene '%s' in config — skipping.", name)
            continue
        gene_overrides[name] = _coerce_values(values)

    known_model_genes = _MODEL_GENE_NAMES.get(backend, set())
    for name, values in model_raw.items():
        if name not in known_model_genes:
            log.warning("Unknown model gene '%s' for backend '%s' — skipping.", name, backend)
            continue
        gene_overrides[name] = _coerce_values(values)

    _validate_gene_overrides(gene_overrides)

    log.info(
        "Config loaded | problem=%s | backend=%s | pop=%d | gens=%d | search_space_genes=%d",
        problem_type.value, backend,
        genetic.population_size, genetic.generations,
        len(gene_overrides),
    )
    return config, gene_overrides


def _parse_problem_type(value: str) -> ProblemType:
    mapping = {
        "classification": ProblemType.CLASSIFICATION,
        "regression": ProblemType.REGRESSION,
        "multi_objective": ProblemType.MULTI_OBJECTIVE,
    }
    if value not in mapping:
        raise ValueError(f"Unknown problem_type '{value}'. Options: {list(mapping.keys())}")
    return mapping[value]


def _coerce_values(raw: Any) -> List[Any]:
    """Ensure value is a list and convert YAML null strings to Python None."""
    if not isinstance(raw, list):
        raw = [raw]
    return [None if v == "null" else v for v in raw]


def _validate_gene_overrides(overrides: Dict[str, List[Any]]) -> None:
    for name, values in overrides.items():
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(
                f"Search space for gene '{name}' must be a non-empty list. Got: {values!r}"
            )
