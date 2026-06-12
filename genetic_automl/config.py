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

    # Surrogate-assisted GA
    surrogate_enabled: bool = True
    """
    When True, a surrogate model is trained on accumulated fitness history
    and used to skip CV for chromosomes predicted to be below the population
    median.  Saves 30-50% of CV evaluations with no meaningful accuracy loss.
    """

    surrogate_model_type: str = "rf"
    """
    Model type used as the fitness surrogate.  Any value accepted by
    build_automl(backend='sklearn', model_type=...) works:
      'rf'   — RandomForest (default): fast, uncertainty-aware via per-tree std.
      'lgbm' — LightGBM: slightly more accurate on large histories.
      'xgb'  — XGBoost: good alternative to lgbm.
      'gbm'  — sklearn GBM: slowest, use only for debugging.
    """

    surrogate_min_samples: int = 10
    """
    Minimum number of fully-evaluated chromosomes before the surrogate
    starts making skip decisions.  Below this threshold every chromosome
    is CV-evaluated regardless.
    """

    surrogate_uncertainty_threshold: float = 0.05
    """
    RF-surrogate only: skip a chromosome only when the per-tree prediction
    std is below this value (i.e. the surrogate is confident).
    Set to float('inf') to disable uncertainty gating (more aggressive skipping).
    """

    # ASHA / median-stop pruning
    asha_enabled: bool = True
    """
    When True, apply median-stop pruning inside each chromosome's CV loop.
    After each fold completes, the running mean score is compared against
    the median of all scores recorded so far (across all chromosomes and
    folds this run).  If the chromosome is already clearly below median,
    the remaining folds are skipped and a pruned fitness is assigned.

    Expected savings: 20-30% of total CV fold evaluations with no
    meaningful impact on the quality of the final model.
    """

    asha_min_folds_before_prune: int = 1
    """
    Number of folds that must complete before pruning can fire.
    1 = prune after fold 1 (most aggressive).
    2 = require 2 folds of evidence first (safer for noisy metrics).
    Must be < n_cv_folds, otherwise pruning never fires.
    """

    asha_prune_margin: float = 0.0
    """
    A chromosome is pruned only when its running mean is at least this
    many units below the population fold-score median.
    0.0  = prune anything below the median (default, balanced).
    0.05 = require 5% clearance below median before pruning (conservative).
    Negative values = prune more aggressively (not recommended).
    """

    # Generation checkpointing
    checkpoint_dir: Optional[str] = None
    """
    Directory where generation checkpoints are saved.
    None (default) = checkpointing disabled.
    When set, the engine saves a .joblib snapshot after every
    checkpoint_every generations so a run can be resumed after a crash.
    """

    checkpoint_every: int = 1
    """
    Save a checkpoint every N completed generations.
    1 = save after every generation (safest, small overhead).
    5 = save every 5 generations (lower I/O for long runs).
    """

    resume_from_checkpoint: Optional[str] = None
    """
    Path to a .joblib checkpoint file to resume from.
    When set, the engine restores the saved population and history, then
    continues evolution from the next generation.
    None (default) = start fresh.
    """

    # NSGA-II multi-objective
    nsga2_enabled: bool = False
    """
    Enable NSGA-II multi-objective selection.

    When True, the GA maintains a Pareto front instead of ranking by a single
    scalar fitness score.  Solutions are ranked by non-domination (front 0 is
    the Pareto-optimal set) and crowding distance (prefer diverse solutions
    within the same front).

    The final model is still selected by the primary metric (best scalar score
    on the first objective), so ensemble, calibration, and SHAP are unaffected.

    Typical objective combinations:
      [f1_macro, roc_auc]         — accuracy vs. discrimination
      [f1_macro, complexity]      — accuracy vs. model size
      [roc_auc, latency]          — discrimination vs. speed
    """

    nsga2_objectives: Optional[List[str]] = None
    """
    List of objective names for NSGA-II.
    First entry = primary metric (used to select final model).
    Remaining entries = secondary objectives.

    Built-in special objectives (no CV evaluation needed):
      complexity  — negated n_estimators (fewer = simpler = better)
      latency     — negated fit duration measured during evaluation

    Any metric name from the registry also works as a secondary objective
    if it was computed during fitness evaluation and stored in
    Chromosome.extra_scores.

    Defaults to [primary_metric, complexity] when nsga2_enabled=True and
    this field is None.
    """

    # Island model GA
    island_model: bool = False
    """
    Enable the island model GA: N independent sub-populations evolve in
    parallel, exchanging their best chromosomes every migration_interval
    generations (ring topology).

    Benefits over a single population:
      - Maintains diversity by keeping sub-populations isolated between
        migration events.
      - Typically finds solutions 5-15% better than a single population of
        the same total size.
      - Total chromosomes evaluated ≈ population_size (divided among islands)
        so wall-clock cost is similar.

    Set enabled=True and tune n_islands / migration_interval / migration_size.
    Requires backend='sklearn'.
    """

    n_islands: int = 4
    """
    Number of sub-populations.  2 = minimal diversity benefit; 4 = sweet spot
    for most datasets; 8 for large compute budgets.
    population_size is divided equally among islands (minimum 4 per island).
    """

    migration_interval: int = 3
    """
    Generations between ring-topology migration events.
    Lower = more gene flow = faster convergence but less diversity.
    Typical range: 2-5. Must be >= 1.
    """

    migration_size: int = 2
    """
    Number of chromosomes sent from each island to the next at each migration.
    Must be < island population size (population_size // n_islands).
    Typical: 1-3.
    """

    n_island_jobs: int = 1
    """
    Worker threads for parallel island evolution.
    1  = sequential (default, safe, easiest to debug).
    -1 = all available threads (faster on multi-core, but LGBM/XGB are
         already multi-threaded so gains diminish quickly).
    """

    # CV split strategy
    cv_strategy: str = "stratified"
    """
    Cross-validation split strategy used when evaluating each chromosome.

    Options:
      "stratified"  — StratifiedKFold for classification, KFold for regression
                      (default; safe for all datasets).
      "group"       — StratifiedGroupKFold / GroupKFold; requires
                      data_config.group_column to identify group membership.
                      Prevents the same group from appearing in both train and
                      val folds — essential for patient/customer/store datasets.
      "timeseries"  — TimeSeriesSplit; preserves temporal ordering, training
                      always precedes validation. No shuffling. Use for
                      any dataset where rows have a meaningful time axis.
    """

    group_column: Optional[str] = None
    """
    Name of the group column in the input DataFrame.
    Required when cv_strategy="group"; ignored otherwise.
    The column is used only for fold assignment and is automatically
    excluded from model features.
    """

    random_seed: int = 42


@dataclass
class EnsembleConfig:
    """Ensemble settings for the final model (applied after the GA finishes)."""

    enabled: bool = True
    """
    When True, refit and combine the top-k unique chromosomes into a
    soft-voting / averaging ensemble.  When False, only the single best
    chromosome is used.
    """

    top_k: int = 3
    """
    Number of unique top chromosomes to include in the ensemble.
    Must be >= 1.  If fewer unique chromosomes exist, all of them are used.
    """

    weight_by_fitness: bool = True
    """
    When True, weight each ensemble member proportionally to its CV fitness.
    When False, use equal weights.
    """


@dataclass
class OptunaConfig:
    """
    Optuna Bayesian HPO settings — applied after the GA finishes.

    The GA discovers the best structural pipeline (which preprocessor, model
    family, feature engineering step).  Optuna then fine-tunes the continuous
    model hyperparameters (learning rate, depth, regularisation) using
    Tree-structured Parzen Estimation (TPE).

    Set enabled=False to skip HPO and use the GA chromosome's hyperparameters
    as-is.
    """

    enabled: bool = False
    """
    Enable Optuna HPO after the GA run.
    Requires optuna to be installed (pip install optuna).
    Default False so existing runs are unaffected until explicitly opted in.
    """

    n_trials: int = 30
    """
    Number of Optuna trials.
    20–30 gives a good speed/accuracy trade-off for most datasets.
    Use 50–100 for production runs where wall-clock time allows.
    """

    timeout: Optional[float] = None
    """
    Hard wall-clock limit in seconds for the entire Optuna study.
    None = no time limit (runs until n_trials are completed).
    Set e.g. 300.0 to cap HPO at 5 minutes regardless of n_trials.
    """

    use_cv: bool = False
    """
    If True, evaluate each Optuna trial with full k-fold CV (slower, more
    accurate — matches the GA's own evaluation).
    If False (default), use a single 80/20 stratified split per trial (5–10×
    faster; sufficient for most HPO tasks).
    """

    n_cv_folds: int = 3
    """Number of CV folds when use_cv=True. Ignored when use_cv=False."""

    verbose: bool = False
    """Log Optuna's per-trial output. Default False keeps the GAML log clean."""


@dataclass
class CalibrationConfig:
    """
    Post-hoc probability calibration applied to the final classification model.

    Tree-based models (GBM, RF, XGBoost, LightGBM) often produce uncalibrated
    probabilities — a predicted score of 0.9 may not correspond to a 90% true
    positive rate.  Calibration corrects this using cross-validated held-out
    predictions, improving log-loss, reliability diagrams, and downstream
    decision-making.

    Only applied for classification tasks; silently skipped for regression.
    Requires backend="sklearn".

    Methods:
      sigmoid  — Platt scaling (logistic regression on model outputs).
                 Works well when the model is already fairly well-calibrated.
      isotonic — Non-parametric isotonic regression.
                 More powerful but requires more data (>1000 samples recommended).
    """

    enabled: bool = False
    """
    Enable post-hoc probability calibration.
    Default False so existing runs are unaffected until explicitly opted in.
    Requires: pip install scikit-learn (always available).
    """

    method: str = "sigmoid"
    """
    Calibration method: 'sigmoid' (Platt scaling) or 'isotonic'.
    sigmoid  — fast, works well for most datasets.
    isotonic — more flexible; needs >= 1000 training samples to be reliable.
    """

    cv: int = 5
    """
    Number of cross-validation folds used internally by CalibratedClassifierCV
    to fit the calibrator.  Higher values = more accurate calibration but slower.
    """


@dataclass
class AutoMLConfig:
    """AutoML backend settings."""

    backend: str = "autogluon"
    """Backend to use. Options: 'autogluon', 'sklearn'."""

    time_limit_per_eval: int = 60
    """Wall-clock seconds allowed per individual fitness evaluation."""

    autogluon_presets: str = "medium_quality"
    """AutoGluon presets string (ignored for other backends)."""

    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    """Ensemble configuration for the final model."""

    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    """Optuna Bayesian HPO configuration — applied after the GA finishes."""

    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    """Post-hoc probability calibration — applied to the final classification model."""

    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Additional kwargs forwarded verbatim to the backend constructor."""


@dataclass
class DataConfig:
    """Data split settings."""

    backend: str = "pandas"
    """
    Data loading backend for CSV/Parquet ingestion.
    Options:
      "pandas" (default) — battle-tested, works everywhere.
      "polars"           — 2-10x faster for large files (>100k rows).
                           Requires: pip install polars pyarrow
                           Falls back to pandas if polars is not installed.
    The rest of GAML always uses pandas DataFrames internally; this only
    affects the initial file-loading step.
    """

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

    mlflow_tracking_uri: Optional[str] = "mlflow_runs"
    """Local MLflow tracking store directory. Set to None to disable MLflow."""

    open_html_on_finish: bool = False
    """Open the HTML report in the default browser when the run completes."""

    shap_enabled: bool = True
    """
    Compute SHAP feature attributions after fitting the final model and
    embed a summary bar chart in the HTML report.
    Requires: pip install shap  (gracefully skipped if not installed).
    Only supported for backend='sklearn' tree models (lgbm, xgb, gbm, rf).
    Set to False to skip SHAP and keep report generation fast.
    """

    drift_enabled: bool = False
    """
    Enable data drift detection via pipeline.detect_drift(new_df).
    When True, the pipeline stores a fitted DriftDetector on the training
    data after fit(), which can then be called to compare any new batch.
    Uses KS test (continuous) + chi-squared (categorical) + PSI.
    Requires: pip install scipy  (falls back to PSI-only without it).
    """

    drift_pvalue_threshold: float = 0.05
    """KS / chi-squared p-value threshold below which drift is flagged."""

    drift_psi_threshold: float = 0.20
    """PSI threshold above which drift is flagged (industry standard)."""

    shap_max_samples: int = 200
    """
    Maximum number of rows from the dev set passed to shap.TreeExplainer.
    Larger values = more accurate SHAP estimates but slower computation.
    200 is a good default; use 500+ for production reports.
    """


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
