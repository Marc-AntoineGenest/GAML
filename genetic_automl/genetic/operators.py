"""
Genetic operators — selection, crossover, mutation.

All operators are pure functions that take / return Chromosome objects
so they are easy to unit-test in isolation.
"""

from __future__ import annotations

import copy
import random
from typing import List, Tuple

from genetic_automl.genetic.chromosome import Chromosome, get_gene_space


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def tournament_selection(
    population: List[Chromosome],
    tournament_size: int,
    rng: random.Random,
) -> Chromosome:
    """
    Pick *tournament_size* individuals at random, return the best one.
    Assumes higher fitness is better (GA always maximizes).
    """
    candidates = rng.sample(population, min(tournament_size, len(population)))
    # individuals with None fitness are ranked last
    best = max(
        candidates,
        key=lambda c: c.fitness if c.fitness is not None else float("-inf"),
    )
    return best.copy()


def elites(
    population: List[Chromosome],
    elite_ratio: float,
) -> List[Chromosome]:
    """Return the top *elite_ratio* fraction (at least 1) unchanged."""
    n = max(1, int(len(population) * elite_ratio))
    sorted_pop = sorted(
        population,
        key=lambda c: c.fitness if c.fitness is not None else float("-inf"),
        reverse=True,
    )
    return [c.copy() for c in sorted_pop[:n]]


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

def single_point_crossover(
    parent_a: Chromosome,
    parent_b: Chromosome,
    rng: random.Random,
) -> Tuple[Chromosome, Chromosome]:
    """
    Uniform single-point crossover on the gene dictionary.

    A random split of gene names determines which genes come from which parent.
    """
    gene_names = list(parent_a.genes.keys())
    if len(gene_names) < 2:
        # Can't really split — just return copies
        return parent_a.copy(), parent_b.copy()

    cut = rng.randint(1, len(gene_names) - 1)
    keys_a = gene_names[:cut]
    keys_b = gene_names[cut:]

    child_a_genes = {k: parent_a.genes[k] for k in keys_a}
    child_a_genes.update({k: parent_b.genes[k] for k in keys_b})

    child_b_genes = {k: parent_b.genes[k] for k in keys_a}
    child_b_genes.update({k: parent_a.genes[k] for k in keys_b})

    child_a = Chromosome(
        genes=child_a_genes,
        parent_ids=[parent_a.id, parent_b.id],
    )
    child_b = Chromosome(
        genes=child_b_genes,
        parent_ids=[parent_a.id, parent_b.id],
    )
    return child_a, child_b


def uniform_crossover(
    parent_a: Chromosome,
    parent_b: Chromosome,
    rng: random.Random,
) -> Tuple[Chromosome, Chromosome]:
    """Each gene independently picked from either parent with p=0.5."""
    gene_names = list(parent_a.genes.keys())
    genes_a, genes_b = {}, {}
    for k in gene_names:
        if rng.random() < 0.5:
            genes_a[k] = parent_a.genes[k]
            genes_b[k] = parent_b.genes[k]
        else:
            genes_a[k] = parent_b.genes[k]
            genes_b[k] = parent_a.genes[k]

    child_a = Chromosome(genes=genes_a, parent_ids=[parent_a.id, parent_b.id])
    child_b = Chromosome(genes=genes_b, parent_ids=[parent_a.id, parent_b.id])
    return child_a, child_b


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def mutate(
    chromosome: Chromosome,
    backend: str,
    mutation_rate: float,
    rng: random.Random,
    gene_space=None,
    _gene_space_dict: dict = None,
) -> Chromosome:
    """
    Randomly replace each gene with a new value from its domain
    with probability *mutation_rate*.

    Parameters
    ----------
    gene_space : list[GeneDefinition] | None
        Pre-built gene space. When None, the default for *backend* is used.
    _gene_space_dict : dict | None
        Pre-built {name: GeneDefinition} dict. Avoids rebuilding on every call
        when provided (pass from GeneticEngine which caches it at construction).
    """
    from genetic_automl.genetic.chromosome import get_gene_space
    if _gene_space_dict is not None:
        resolved = _gene_space_dict
    else:
        resolved = {g.name: g for g in (gene_space if gene_space is not None else get_gene_space(backend))}
    new_genes = copy.deepcopy(chromosome.genes)

    for name, gene_def in resolved.items():
        if rng.random() < mutation_rate:
            new_genes[name] = gene_def.random_value(rng)

    mutant = Chromosome(
        genes=new_genes,
        parent_ids=[chromosome.id],
    )
    return mutant
