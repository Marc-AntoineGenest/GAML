"""
SurrogateModel — accelerates the GA by predicting chromosome fitness before
committing to a full k-fold CV evaluation.

How it works
------------
After enough chromosomes have been evaluated (min_samples), the surrogate
is trained on the accumulated fitness history:
    X_surr : (n_evaluated, n_genes)  — ordinal-encoded gene vectors
    y_surr : (n_evaluated,)          — penalised CV fitness values

Before each new chromosome is evaluated, the surrogate predicts its fitness.
If the prediction is below the current population median AND the model's
uncertainty is low (std_pred < uncertainty_threshold), the chromosome is
skipped — it gets the surrogate's predicted fitness without paying for CV.

Any model in the sklearn zoo (rf, lgbm, xgb, gbm) can act as the surrogate.
The default is 'rf' because it:
  - Requires zero hyperparameter tuning.
  - Provides free uncertainty estimates via per-tree std (predict_with_std).
  - Trains in milliseconds on the tiny fitness-history dataset.

Uncertainty handling
--------------------
For RF (RandomForestModel): predict_with_std() returns per-tree std.
For other models: uncertainty is not available, so the surrogate only skips
when the predicted fitness is well below the median (uses a stricter margin).

Configuration (via GeneticConfig)
----------------------------------
surrogate_enabled         : bool  — toggle the whole mechanism.
surrogate_model_type      : str   — 'rf' | 'lgbm' | 'xgb' | 'gbm'.
surrogate_min_samples     : int   — minimum evaluations before surrogate fires.
surrogate_uncertainty_thr : float — skip only when std_pred < this value.
                                    Ignored for non-RF surrogates.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from genetic_automl.genetic.chromosome import Chromosome, get_gene_space
from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)


class SurrogateModel:
    """
    Fitness surrogate built on top of any registered AutoML model type.

    Parameters
    ----------
    model_type : str
        Model type used as the surrogate predictor.
        'rf' is the default (fast, uncertainty-aware).
        Any value accepted by build_automl(backend='sklearn', model_type=...) works.
    backend_for_ga : str
        The GA's own AutoML backend (e.g. 'sklearn', 'autogluon').
        Used to look up the correct gene space for encoding.
    min_samples : int
        Minimum number of evaluated chromosomes before the surrogate fires.
        Below this threshold, every chromosome is fully CV-evaluated.
    uncertainty_threshold : float
        For RF surrogates: only skip when per-tree std < this value.
        High std = the surrogate is uncertain → don't trust the prediction.
        Set to float('inf') to skip uncertainty gating entirely.
    skip_margin : float
        A chromosome's predicted fitness must be this many units below the
        current population median to be skipped.
        Positive = more conservative (fewer skips, higher safety margin).
        0.0 = skip everything below the median.
    random_seed : int
    """

    def __init__(
        self,
        model_type: str = "rf",
        backend_for_ga: str = "sklearn",
        min_samples: int = 10,
        uncertainty_threshold: float = 0.05,
        skip_margin: float = 0.0,
        random_seed: int = 42,
    ) -> None:
        self.model_type = model_type
        self.backend_for_ga = backend_for_ga
        self.min_samples = min_samples
        self.uncertainty_threshold = uncertainty_threshold
        self.skip_margin = skip_margin
        self.random_seed = random_seed

        self._surrogate = None          # fitted surrogate model (BaseAutoML)
        self._gene_names: list[str] = []  # ordered gene names for encoding
        self._gene_value_maps: dict[str, dict[Any, int]] = {}
        self._n_trained_on: int = 0     # how many samples last fit used
        self._skips: int = 0
        self._total_candidates: int = 0

        self._build_gene_encoder()


    def should_skip(
        self,
        chromosome: Chromosome,
        population_median: float,
    ) -> tuple[bool, float]:
        """
        Decide whether to skip full CV for this chromosome.

        Returns
        -------
        skip : bool
            True → use surrogate prediction, skip CV.
        predicted_fitness : float
            Surrogate's fitness estimate (valid regardless of skip decision).
        """
        self._total_candidates += 1

        if self._surrogate is None:
            return False, float("-inf")

        x = self._encode([chromosome])          # (1, n_genes)
        predicted, std = self._predict_with_uncertainty(x)
        predicted_fitness = float(predicted[0])
        uncertainty = float(std[0]) if std is not None else 0.0

        threshold = population_median - self.skip_margin
        is_low = predicted_fitness < threshold
        is_certain = uncertainty < self.uncertainty_threshold

        skip = is_low and is_certain
        if skip:
            self._skips += 1
            log.debug(
                "Surrogate skip | id=%s | pred=%.4f | median=%.4f | std=%.4f",
                chromosome.id, predicted_fitness, population_median, uncertainty,
            )
        return skip, predicted_fitness

    def update(self, evaluated_chromosomes: list[Chromosome]) -> None:
        """
        Retrain the surrogate on all chromosomes that have a fitness value.

        Call this once per generation, after the generation's evaluations are done.
        """
        ready = [c for c in evaluated_chromosomes
                 if c.fitness is not None and c.fitness != float("-inf")]

        if len(ready) < self.min_samples:
            log.debug(
                "Surrogate not trained yet (%d/%d samples collected).",
                len(ready), self.min_samples,
            )
            return

        X = self._encode(ready)
        y = np.array([c.fitness for c in ready], dtype=float)

        self._surrogate = self._build_surrogate_model()
        # Fit using pandas wrappers to satisfy BaseAutoML interface
        import pandas as pd
        X_df = pd.DataFrame(X, columns=[f"g{i}" for i in range(X.shape[1])])
        y_ser = pd.Series(y, name="_fitness")
        self._surrogate.fit(X_df, y_ser)
        self._n_trained_on = len(ready)
        log.info(
            "Surrogate updated | model=%s | samples=%d | skips_so_far=%d",
            self.model_type, self._n_trained_on, self._skips,
        )

    @property
    def skip_rate(self) -> float:
        """Fraction of candidates that were surrogate-skipped so far."""
        if self._total_candidates == 0:
            return 0.0
        return self._skips / self._total_candidates

    def summary(self) -> dict:
        return {
            "model_type": self.model_type,
            "min_samples": self.min_samples,
            "n_trained_on": self._n_trained_on,
            "total_candidates": self._total_candidates,
            "skips": self._skips,
            "skip_rate": round(self.skip_rate, 3),
        }


    def _build_surrogate_model(self):
        """
        Instantiate a fresh surrogate using build_automl so the model type is
        fully swappable via config — same dispatch table as the GA models.
        """
        from genetic_automl.automl import build_automl
        from genetic_automl.core.problem import ProblemType

        # Use a lightweight configuration for the surrogate:
        #   n_estimators=100  — enough for stable predictions on small datasets
        #   max_depth=None/4  — appropriate defaults per model family
        model_kwargs: dict = {
            "model_type": self.model_type,
            "n_estimators": 100,
        }
        if self.model_type in ("gbm", "xgb"):
            model_kwargs["max_depth"] = 4
            model_kwargs["learning_rate"] = 0.1

        return build_automl(
            backend="sklearn",
            # Surrogate always does regression — it predicts a scalar fitness.
            problem_type=ProblemType.REGRESSION,
            target_column="_fitness",
            random_seed=self.random_seed,
            **model_kwargs,
        )

    def _predict_with_uncertainty(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Return (mean_predictions, std_predictions).
        std is None for models that don't expose predict_with_std().
        """
        import pandas as pd
        X_df = pd.DataFrame(X, columns=[f"g{i}" for i in range(X.shape[1])])

        # RF exposes per-tree uncertainty natively
        if hasattr(self._surrogate, "predict_with_std"):
            mean, std = self._surrogate.predict_with_std(X_df)
            return mean, std

        # For other models, fall back to point estimate only.
        # The skip decision then uses a stricter margin (no uncertainty gating).
        mean = self._surrogate.predict(X_df)
        return mean, None

    def _build_gene_encoder(self) -> None:
        """
        Build ordinal encoders for each gene in the GA's gene space.
        Categorical values → integer indices; numeric values → kept as-is.
        Called once at construction.
        """
        try:
            gene_space = get_gene_space(self.backend_for_ga)
        except ValueError:
            # Unknown backend (e.g. autogluon not installed): use empty space,
            # surrogate will not fire until genes are seen.
            gene_space = []

        self._gene_names = [g.name for g in gene_space]
        for gene in gene_space:
            if any(not isinstance(v, (int, float)) for v in gene.values if v is not None):
                # Categorical gene: map each unique value to an integer
                self._gene_value_maps[gene.name] = {
                    v: idx for idx, v in enumerate(gene.values)
                }

    def _encode(self, chromosomes: list[Chromosome]) -> np.ndarray:
        """
        Convert a list of chromosomes into a (n, n_genes) float matrix.

        Unknown gene values (from crossover producing unseen combos) are
        encoded as -1, which tree-based models handle gracefully.
        """
        rows = []
        for chrom in chromosomes:
            row = []
            for name in self._gene_names:
                val = chrom.genes.get(name)
                if name in self._gene_value_maps:
                    row.append(float(self._gene_value_maps[name].get(val, -1)))
                elif val is None:
                    row.append(-1.0)
                else:
                    try:
                        row.append(float(val))
                    except (TypeError, ValueError):
                        row.append(-1.0)
            rows.append(row)
        return np.array(rows, dtype=float)
