"""
Chromosome — encodes a candidate AutoML pipeline configuration.

Each chromosome is a dictionary of hyper-parameters that the genetic
algorithm evolves. The gene space is defined in GeneSpace and is
extensible per backend.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Gene space definition
# ---------------------------------------------------------------------------

@dataclass
class GeneDefinition:
    """A single gene: name + allowed values."""
    name: str
    values: List[Any]

    def random_value(self, rng: random.Random) -> Any:
        return rng.choice(self.values)


# ---------------------------------------------------------------------------
# Preprocessing genes — shared across all backends
# ---------------------------------------------------------------------------

PREPROCESSING_GENES: List[GeneDefinition] = [
    # Step 1: Correlation filter (reduces dimensionality first — cheaper subsequent steps)
    GeneDefinition("correlation_threshold", [None, 0.85, 0.90, 0.95]),

    # Step 2: Numeric imputation (MUST be first — NaN breaks IQR/IsolationForest)
    GeneDefinition("numeric_imputer", ["mean", "median", "knn", "iterative", "constant"]),

    # Step 3: Outlier handling (on clean numeric data, before scaling distorts distances)
    GeneDefinition("outlier_method", ["none", "iqr", "zscore", "isolation_forest"]),
    GeneDefinition("outlier_threshold", [1.5, 2.0, 3.0]),
    GeneDefinition("outlier_action", ["clip", "flag"]),

    # Step 4: Categorical encoding (encode before scaling — scaling strings is nonsensical)
    GeneDefinition("categorical_encoder", ["onehot", "ordinal", "target", "binary"]),

    # Step 5: Distribution transform (normalize skewness before scaling)
    GeneDefinition("distribution_transform", ["none", "yeo-johnson", "box-cox", "log1p"]),

    # Step 6: Scaling (after all columns are numeric and distributions are shaped)
    GeneDefinition("scaler", ["none", "standard", "minmax", "robust"]),

    # Step 7: Missing indicator (binary flags — added after imputation, signals missingness)
    GeneDefinition("missing_indicator", [True, False]),

    # Step 8: Feature selection (on fully preprocessed data)
    GeneDefinition("feature_selection_method", ["none", "variance_threshold", "mutual_info", "rfe"]),
    GeneDefinition("feature_selection_k", [0.5, 0.75, 1.0]),

    # Step 9: Imbalance handling (ALWAYS LAST — train only, after final feature matrix is ready)
    GeneDefinition("imbalance_method", ["none", "smote", "borderline_smote", "adasyn", "class_weight"]),
]

# Backend-specific model genes
_MODEL_GENES: Dict[str, List[GeneDefinition]] = {
    "autogluon": [
        GeneDefinition("presets", [
            "medium_quality",
            "good_quality",
            "high_quality",
            "optimize_for_deployment",
        ]),
        GeneDefinition("time_limit", [30, 60, 120, 240, 300]),
        GeneDefinition("ag_metric", [None]),  # let AutoGluon choose
    ],
    "sklearn": [
        GeneDefinition("n_estimators", [50, 100, 200, 300, 500]),
        GeneDefinition("max_depth", [2, 3, 4, 5, 6, 8]),
        GeneDefinition("learning_rate", [0.01, 0.05, 0.1, 0.2]),
    ],
}

# Full gene space = preprocessing genes + backend model genes
_GENE_SPACES: Dict[str, List[GeneDefinition]] = {
    backend: PREPROCESSING_GENES + model_genes
    for backend, model_genes in _MODEL_GENES.items()
}


def get_gene_space(backend: str) -> List[GeneDefinition]:
    backend = backend.lower()
    if backend not in _GENE_SPACES:
        raise ValueError(
            f"No gene space for backend '{backend}'. "
            f"Available: {list(_GENE_SPACES.keys())}"
        )
    return _GENE_SPACES[backend]


# ---------------------------------------------------------------------------
# Chromosome
# ---------------------------------------------------------------------------

@dataclass
class Chromosome:
    """
    One individual in the GA population.

    Attributes
    ----------
    genes : dict
        Mapping of gene name → value.
    fitness : float | None
        Evaluated fitness (None = not yet evaluated).
    generation : int
        Which generation this individual was created in.
    parent_ids : list[str]
        IDs of parent chromosomes (for lineage tracking).
    id : str
        Unique 8-character hex id.
    """

    genes: Dict[str, Any]
    fitness: Optional[float] = None
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: _random_id())

    # ------------------------------------------------------------------

    def copy(self) -> "Chromosome":
        c = Chromosome(
            genes=copy.deepcopy(self.genes),
            fitness=self.fitness,
            generation=self.generation,
            parent_ids=list(self.parent_ids),
        )
        return c

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "genes": self.genes,
            "fitness": self.fitness,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
        }

    def __repr__(self) -> str:
        genes_str = ", ".join(f"{k}={v}" for k, v in self.genes.items())
        return (
            f"Chromosome(id={self.id}, fitness={self.fitness:.6f if self.fitness is not None else 'None'}, "
            f"genes=[{genes_str}])"
        )


# ---------------------------------------------------------------------------
# Population factory
# ---------------------------------------------------------------------------

def random_population(
    backend: str,
    size: int,
    rng: random.Random,
    generation: int = 0,
) -> List[Chromosome]:
    """Generate *size* random chromosomes for the given backend."""
    gene_space = get_gene_space(backend)
    population = []
    for _ in range(size):
        genes = {g.name: g.random_value(rng) for g in gene_space}
        population.append(Chromosome(genes=genes, generation=generation))
    return population


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_id() -> str:
    import uuid
    return uuid.uuid4().hex[:8]
