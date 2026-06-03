"""
OptunaTuner — Bayesian hyperparameter optimisation (HPO) applied after the
Genetic Algorithm has identified a promising region of the search space.

Why Optuna after the GA?
------------------------
The GA is excellent at searching a *wide, discrete* space (which preprocessor?
which model family? which feature engineering step?).  Once the best chromosome
is found those structural choices are fixed.  Optuna then takes over the
*narrow, continuous* hyperparameter space (learning rate, regularisation
strength, tree depth) using TPE (Tree-structured Parzen Estimator) — a
Bayesian method that is far more sample-efficient than the GA for this kind of
fine-tuning.

Typical result: +1–5% improvement on top of the GA's best chromosome, using
only 20–50 additional model evaluations (each cheaper than a full k-fold CV
because we evaluate on a single 80/20 split by default).

Design decisions
----------------
- The tuner is **model-type aware**: it defines a search space appropriate for
  the model_type gene found in the winning chromosome ('lgbm', 'xgb', 'gbm',
  'rf').  Unknown model types fall back gracefully with no tuning.
- Evaluation uses a **single stratified 80/20 hold-out** (fast) by default.
  Optionally, full k-fold CV can be requested via `use_cv=True` for production
  runs where accuracy matters more than speed.
- The tuner is **optional and additive**: it is skipped automatically when
  optuna is not installed (ImportError) or when disabled via `OptunaConfig`.
- All Optuna output is suppressed at WARNING level so the GAML log stays clean.

Usage
-----
Called internally by AutoMLPipeline._build_final_model() when
cfg.automl.optuna.enabled is True.  Not intended for direct user calls, but
fully usable standalone::

    from genetic_automl.genetic.optuna_tuner import OptunaTuner
    tuner = OptunaTuner(config=OptunaConfig(n_trials=50))
    best_params = tuner.tune(
        best_chrom, X_dev_pp, y_dev_pp,
        problem_type=ProblemType.CLASSIFICATION,
        target_column="label",
        metric="f1",
        random_seed=42,
    )
    # best_params is a dict ready to pass to build_automl()
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from genetic_automl.core.problem import ProblemType, compute_metric, fitness_sign
from genetic_automl.genetic.fitness import _split_genes
from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Per-model Optuna search spaces
# Each entry is (param_name, suggest_type, kwargs).
# suggest_type: "float" | "int" | "categorical" | "log_float"
# ---------------------------------------------------------------------------

_SEARCH_SPACES: Dict[str, list] = {
    "lgbm": [
        ("n_estimators",      "int",        {"low": 100,  "high": 1000, "step": 50}),
        ("learning_rate",     "log_float",  {"low": 1e-3, "high": 0.3}),
        ("num_leaves",        "int",        {"low": 15,   "high": 127}),
        ("max_depth",         "int",        {"low": -1,   "high": 12}),
        ("subsample",         "float",      {"low": 0.5,  "high": 1.0}),
        ("colsample_bytree",  "float",      {"low": 0.5,  "high": 1.0}),
        ("reg_alpha",         "log_float",  {"low": 1e-8, "high": 10.0}),
        ("reg_lambda",        "log_float",  {"low": 1e-8, "high": 10.0}),
        ("min_child_samples", "int",        {"low": 5,    "high": 100}),
    ],
    "xgb": [
        ("n_estimators",      "int",        {"low": 100,  "high": 1000, "step": 50}),
        ("learning_rate",     "log_float",  {"low": 1e-3, "high": 0.3}),
        ("max_depth",         "int",        {"low": 2,    "high": 12}),
        ("subsample",         "float",      {"low": 0.5,  "high": 1.0}),
        ("colsample_bytree",  "float",      {"low": 0.5,  "high": 1.0}),
        ("reg_alpha",         "log_float",  {"low": 1e-8, "high": 10.0}),
        ("reg_lambda",        "log_float",  {"low": 1e-8, "high": 10.0}),
        ("min_child_weight",  "int",        {"low": 1,    "high": 10}),
        ("gamma",             "log_float",  {"low": 1e-8, "high": 1.0}),
    ],
    "gbm": [
        ("n_estimators",      "int",        {"low": 50,   "high": 500,  "step": 25}),
        ("learning_rate",     "log_float",  {"low": 1e-3, "high": 0.3}),
        ("max_depth",         "int",        {"low": 2,    "high": 8}),
        ("subsample",         "float",      {"low": 0.5,  "high": 1.0}),
        ("min_samples_leaf",  "int",        {"low": 1,    "high": 50}),
    ],
    "rf": [
        ("n_estimators",      "int",        {"low": 50,   "high": 500,  "step": 25}),
        ("max_depth",         "int",        {"low": 3,    "high": 20}),
        ("min_samples_leaf",  "int",        {"low": 1,    "high": 20}),
        ("max_features",      "categorical", {"choices": ["sqrt", "log2", 0.5, 0.75, 1.0]}),
    ],
}


def _suggest_param(trial, name: str, suggest_type: str, kwargs: dict) -> Any:
    """Dispatch a single suggest call on an Optuna trial."""
    if suggest_type == "float":
        return trial.suggest_float(name, **kwargs)
    if suggest_type == "log_float":
        return trial.suggest_float(name, log=True, **kwargs)
    if suggest_type == "int":
        return trial.suggest_int(name, **kwargs)
    if suggest_type == "categorical":
        return trial.suggest_categorical(name, kwargs.get("choices", []))
    raise ValueError(f"Unknown suggest_type: {suggest_type!r}")


class OptunaTuner:
    """
    Bayesian HPO via Optuna TPE, applied to the model-hyperparameter subspace
    of the best GA chromosome.

    Parameters
    ----------
    n_trials : int
        Number of Optuna trials.  20 is a useful minimum; 50–100 for
        production.  Each trial is one model fit on an 80/20 split (fast).
    timeout : float | None
        Hard wall-clock limit in seconds.  None = no limit.
    use_cv : bool
        If True, evaluate each trial with k-fold CV (slower, more accurate).
        If False (default), use a single 80/20 stratified split (faster).
    n_cv_folds : int
        Number of folds when use_cv=True.
    direction : str
        "maximize" (default) or "minimize".  Inferred automatically from the
        metric's fitness_sign, so callers rarely need to change this.
    verbose : bool
        If True, log Optuna's per-trial output.  Default False (quiet).
    """

    def __init__(
        self,
        n_trials: int = 30,
        timeout: Optional[float] = None,
        use_cv: bool = False,
        n_cv_folds: int = 3,
        verbose: bool = False,
    ) -> None:
        self.n_trials = n_trials
        self.timeout = timeout
        self.use_cv = use_cv
        self.n_cv_folds = n_cv_folds
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tune(
        self,
        best_chromosome,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: ProblemType,
        target_column: str,
        metric: str,
        backend: str = "sklearn",
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Run Bayesian HPO around the best chromosome's model-gene values.

        Returns
        -------
        dict
            Model hyperparameter dict with the same keys as the chromosome's
            model genes, updated with the tuned values.  Pass directly to
            build_automl() as **kwargs.

        Notes
        -----
        - If optuna is not installed, a warning is logged and the original
          chromosome model genes are returned unchanged.
        - If the model_type has no defined search space, the chromosome genes
          are returned unchanged.
        """
        try:
            import optuna
        except ImportError:
            log.warning(
                "optuna is not installed — skipping HPO tuning. "
                "Install it with:  pip install optuna"
            )
            _, model_genes = _split_genes(best_chromosome.genes)
            return model_genes

        _, model_genes = _split_genes(best_chromosome.genes)
        model_type = model_genes.get("model_type", "gbm")
        search_space = _SEARCH_SPACES.get(model_type)

        if not search_space:
            log.info(
                "OptunaTuner: no search space defined for model_type=%r — "
                "returning chromosome genes unchanged.",
                model_type,
            )
            return model_genes

        log.info(
            "OptunaTuner starting | model_type=%s | n_trials=%d | "
            "use_cv=%s | metric=%s",
            model_type, self.n_trials, self.use_cv, metric,
        )

        try:
            sign = fitness_sign(metric)  # +1 = higher is better, -1 = lower is better
        except KeyError:
            # Unknown metric — assume higher is better (safe default)
            log.warning(
                "OptunaTuner: metric %r not in registry; assuming higher=better.", metric
            )
            sign = 1

        # Optuna always maximises; negate score when lower-is-better metric.
        def objective(trial):
            params = {
                name: _suggest_param(trial, name, stype, skwargs)
                for name, stype, skwargs in search_space
            }
            merged = {**model_genes, **params}
            return self._evaluate(
                merged, X, y, problem_type, target_column, metric, sign,
                backend, random_seed, trial.number,
            )

        # Silence Optuna's own loggers unless verbose requested
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(seed=random_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # Seed the study with the GA's best chromosome as trial 0 so Optuna
        # starts from a known-good point rather than a random one.
        seed_params = self._chromosome_to_optuna_params(model_genes, search_space)
        if seed_params:
            study.enqueue_trial(seed_params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=False,
            )

        best_trial = study.best_trial
        best_optuna_params = best_trial.params
        tuned_genes = {**model_genes, **best_optuna_params}

        log.info(
            "OptunaTuner finished | best_trial=%d | best_value=%.6f | "
            "tuned_params=%s",
            best_trial.number,
            best_trial.value,
            {k: v for k, v in best_optuna_params.items()},
        )
        return tuned_genes

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        model_genes: dict,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: ProblemType,
        target_column: str,
        metric: str,
        sign: int,
        backend: str,
        random_seed: int,
        trial_number: int,
    ) -> float:
        """
        Build, fit, and score one set of model_genes.

        Uses a single 80/20 split (fast default) or k-fold CV (accurate).
        Returns float('-inf') on any exception so Optuna can continue.
        """
        from genetic_automl.automl import build_automl

        try:
            if self.use_cv:
                return self._cv_score(
                    model_genes, X, y, problem_type, target_column,
                    metric, sign, backend, random_seed,
                )
            else:
                return self._holdout_score(
                    model_genes, X, y, problem_type, target_column,
                    metric, sign, backend, random_seed,
                )
        except Exception as exc:
            log.debug("OptunaTuner trial %d failed: %s", trial_number, exc)
            return float("-inf")

    def _holdout_score(
        self,
        model_genes: dict,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: ProblemType,
        target_column: str,
        metric: str,
        sign: int,
        backend: str,
        random_seed: int,
    ) -> float:
        """Single 80/20 stratified split evaluation."""
        from sklearn.model_selection import train_test_split
        from genetic_automl.automl import build_automl

        stratify = y if problem_type != ProblemType.REGRESSION else None
        try:
            X_tr, X_hld, y_tr, y_hld = train_test_split(
                X, y, test_size=0.20, random_state=random_seed, stratify=stratify,
            )
        except ValueError:
            # Fallback: no stratify (e.g. too few samples per class)
            X_tr, X_hld, y_tr, y_hld = train_test_split(
                X, y, test_size=0.20, random_state=random_seed,
            )

        model = build_automl(
            backend=backend,
            problem_type=problem_type,
            target_column=target_column,
            random_seed=random_seed,
            **{k: v for k, v in model_genes.items() if v is not None},
        )
        model.fit(X_tr, y_tr)
        raw = model.score(X_hld, y_hld, metric=metric)
        return raw * sign

    def _cv_score(
        self,
        model_genes: dict,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: ProblemType,
        target_column: str,
        metric: str,
        sign: int,
        backend: str,
        random_seed: int,
    ) -> float:
        """Full k-fold CV evaluation (accurate but slower)."""
        from sklearn.model_selection import StratifiedKFold, KFold
        from genetic_automl.automl import build_automl

        if problem_type == ProblemType.REGRESSION:
            cv = KFold(n_splits=self.n_cv_folds, shuffle=True, random_state=random_seed)
            splits = list(cv.split(X))
        else:
            cv = StratifiedKFold(n_splits=self.n_cv_folds, shuffle=True, random_state=random_seed)
            splits = list(cv.split(X, y))

        fold_scores = []
        for train_idx, val_idx in splits:
            X_tr = X.iloc[train_idx].reset_index(drop=True)
            y_tr = y.iloc[train_idx].reset_index(drop=True)
            X_vl = X.iloc[val_idx].reset_index(drop=True)
            y_vl = y.iloc[val_idx].reset_index(drop=True)

            model = build_automl(
                backend=backend,
                problem_type=problem_type,
                target_column=target_column,
                random_seed=random_seed,
                **{k: v for k, v in model_genes.items() if v is not None},
            )
            model.fit(X_tr, y_tr)
            raw = model.score(X_vl, y_vl, metric=metric)
            fold_scores.append(raw * sign)

        return float(np.mean(fold_scores))

    @staticmethod
    def _chromosome_to_optuna_params(
        model_genes: dict,
        search_space: list,
    ) -> dict:
        """
        Extract from model_genes only the parameters that are in the search
        space, so Optuna can enqueue the GA's chromosome as trial 0.

        Parameters whose gene value falls outside Optuna's defined range are
        skipped (Optuna would raise on out-of-range enqueued values).
        """
        result = {}
        for name, stype, kwargs in search_space:
            val = model_genes.get(name)
            if val is None:
                continue
            try:
                if stype in ("float", "log_float"):
                    low, high = kwargs["low"], kwargs["high"]
                    if low <= float(val) <= high:
                        result[name] = float(val)
                elif stype == "int":
                    low, high = kwargs["low"], kwargs["high"]
                    if low <= int(val) <= high:
                        result[name] = int(val)
                elif stype == "categorical":
                    choices = kwargs.get("choices", [])
                    # Compare as strings to handle mixed int/str choice lists
                    str_choices = [str(c) for c in choices]
                    if str(val) in str_choices:
                        idx = str_choices.index(str(val))
                        result[name] = choices[idx]
            except (TypeError, ValueError):
                pass  # skip params that can't be cast
        return result
