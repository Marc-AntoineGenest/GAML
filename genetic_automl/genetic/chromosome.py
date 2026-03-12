"""
Chromosome — encodes a candidate AutoML pipeline configuration.

Each chromosome is a flat dict of gene name → value. The gene space defines
which names exist and what values each gene is allowed to take.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GeneDefinition:
    """A single gene: its name and the list of allowed values."""
    name: str
    values: List[Any]

    def random_value(self, rng: random.Random) -> Any:
        return rng.choice(self.values)


# Preprocessing genes — shared across all backends.
# Applied in this order by PreprocessingPipeline:
#   imputer → outlier → correlation → encoder → transform → scaler
#   → missing indicator → feature selection → imbalance
PREPROCESSING_GENES: List[GeneDefinition] = [
    GeneDefinition("correlation_threshold", [None, 0.85, 0.90, 0.95]),
    GeneDefinition("numeric_imputer", ["mean", "median", "knn", "iterative", "constant"]),
    GeneDefinition("outlier_method", ["none", "iqr", "zscore", "isolation_forest"]),
    GeneDefinition("outlier_threshold", [1.5, 2.0, 3.0]),
    GeneDefinition("outlier_action", ["clip", "flag"]),
    GeneDefinition("categorical_encoder", ["onehot", "ordinal", "target", "binary"]),
    GeneDefinition("distribution_transform", ["none", "yeo-johnson", "box-cox", "log1p"]),
    GeneDefinition("scaler", ["none", "standard", "minmax", "robust"]),
    GeneDefinition("missing_indicator", [True, False]),
    GeneDefinition("feature_selection_method", ["none", "variance_threshold", "mutual_info", "rfe"]),
    GeneDefinition("feature_selection_k", [0.5, 0.75, 1.0]),
    GeneDefinition("imbalance_method", ["none", "smote", "borderline_smote", "adasyn", "class_weight"]),
]

_MODEL_GENES: Dict[str, List[GeneDefinition]] = {
    "autogluon": [
        GeneDefinition("presets", [
            "medium_quality", "good_quality", "high_quality", "optimize_for_deployment",
        ]),
        GeneDefinition("time_limit", [30, 60, 120, 240, 300]),
        GeneDefinition("ag_metric", [None]),
    ],
    "sklearn": [
        GeneDefinition("n_estimators", [50, 100, 200, 300, 500]),
        GeneDefinition("max_depth", [2, 3, 4, 5, 6, 8]),
        GeneDefinition("learning_rate", [0.01, 0.05, 0.1, 0.2]),
    ],
}

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


def build_gene_space_from_config(
    backend: str,
    overrides: Dict[str, list],
) -> List[GeneDefinition]:
    """
    Return a gene space with candidate values replaced by those in overrides.

    Genes not in overrides keep their default candidates. A single-element list
    pins a gene to one value — the GA will never mutate it away.

    Parameters
    ----------
    backend : str
    overrides : dict[str, list]
        Typically loaded from gaml_config.yaml via load_config().
    """
    base = get_gene_space(backend)
    if not overrides:
        return base
    result = []
    for gene in base:
        if gene.name in overrides:
            candidates = overrides[gene.name]
            if not isinstance(candidates, list) or len(candidates) == 0:
                raise ValueError(
                    f"Gene '{gene.name}' override must be a non-empty list. "
                    f"Got: {candidates!r}"
                )
            result.append(GeneDefinition(name=gene.name, values=candidates))
        else:
            result.append(gene)
    return result


@dataclass
class Chromosome:
    """
    One individual in the GA population.

    Attributes
    ----------
    genes : dict
        Gene name → value mapping.
    fitness : float | None
        Evaluated fitness (None = not yet evaluated).
    generation : int
        Generation index when this chromosome was created.
    parent_ids : list[str]
        IDs of parent chromosomes (empty for generation 0).
    id : str
        Unique 8-character hex identifier.
    """

    genes: Dict[str, Any]
    fitness: Optional[float] = None
    fitness_std: Optional[float] = None
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: _random_id())

    def copy(self) -> "Chromosome":
        return Chromosome(
            genes=copy.deepcopy(self.genes),
            fitness=self.fitness,
            fitness_std=self.fitness_std,
            generation=self.generation,
            parent_ids=list(self.parent_ids),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "genes": self.genes,
            "fitness": self.fitness,
            "fitness_std": self.fitness_std,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
        }

    def __repr__(self) -> str:
        genes_str = ", ".join(f"{k}={v}" for k, v in self.genes.items())
        fitness_str = f"{self.fitness:.6f}" if self.fitness is not None else "None"
        return f"Chromosome(id={self.id}, fitness={fitness_str}, genes=[{genes_str}])"


def random_population(
    backend: str,
    size: int,
    rng: random.Random,
    generation: int = 0,
    gene_space: Optional[List[GeneDefinition]] = None,
) -> List[Chromosome]:
    """
    Generate size random chromosomes.

    Parameters
    ----------
    gene_space : list[GeneDefinition] | None
        Pre-built gene space. None = default space for backend.
    """
    if gene_space is None:
        gene_space = get_gene_space(backend)
    return [
        Chromosome(
            genes={g.name: g.random_value(rng) for g in gene_space},
            generation=generation,
        )
        for _ in range(size)
    ]


def _random_id() -> str:
    import uuid
    return uuid.uuid4().hex[:8]
