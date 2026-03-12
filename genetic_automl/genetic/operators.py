"""
Genetic operators — selection, crossover, mutation.

All operators are pure functions operating on Chromosome objects.
"""

from __future__ import annotations

import copy
import random
from typing import List, Tuple

from genetic_automl.genetic.chromosome import Chromosome, get_gene_space


def tournament_selection(
    population: List[Chromosome],
    tournament_size: int,
    rng: random.Random,
) -> Chromosome:
    """
    Draw tournament_size individuals at random and return the best one.
    Individuals with None fitness are ranked last.
    """
    candidates = rng.sample(population, min(tournament_size, len(population)))
    return max(
        candidates,
        key=lambda c: c.fitness if c.fitness is not None else float("-inf"),
    ).copy()


def elites(
    population: List[Chromosome],
    elite_ratio: float,
) -> List[Chromosome]:
    """Return the top elite_ratio fraction (at least 1 individual) unchanged."""
    n = max(1, int(len(population) * elite_ratio))
    sorted_pop = sorted(
        population,
        key=lambda c: c.fitness if c.fitness is not None else float("-inf"),
        reverse=True,
    )
    return [c.copy() for c in sorted_pop[:n]]


def single_point_crossover(
    parent_a: Chromosome,
    parent_b: Chromosome,
    rng: random.Random,
) -> Tuple[Chromosome, Chromosome]:
    """
    Split gene list at a random cut point and swap the tails between parents.
    """
    gene_names = list(parent_a.genes.keys())
    if len(gene_names) < 2:
        return parent_a.copy(), parent_b.copy()

    cut = rng.randint(1, len(gene_names) - 1)
    keys_a = gene_names[:cut]
    keys_b = gene_names[cut:]

    child_a_genes = {k: parent_a.genes[k] for k in keys_a}
    child_a_genes.update({k: parent_b.genes[k] for k in keys_b})

    child_b_genes = {k: parent_b.genes[k] for k in keys_a}
    child_b_genes.update({k: parent_a.genes[k] for k in keys_b})

    return (
        Chromosome(genes=child_a_genes, parent_ids=[parent_a.id, parent_b.id]),
        Chromosome(genes=child_b_genes, parent_ids=[parent_a.id, parent_b.id]),
    )


def uniform_crossover(
    parent_a: Chromosome,
    parent_b: Chromosome,
    rng: random.Random,
) -> Tuple[Chromosome, Chromosome]:
    """Each gene is independently drawn from either parent with p=0.5."""
    gene_names = list(parent_a.genes.keys())
    genes_a, genes_b = {}, {}
    for k in gene_names:
        if rng.random() < 0.5:
            genes_a[k] = parent_a.genes[k]
            genes_b[k] = parent_b.genes[k]
        else:
            genes_a[k] = parent_b.genes[k]
            genes_b[k] = parent_a.genes[k]

    return (
        Chromosome(genes=genes_a, parent_ids=[parent_a.id, parent_b.id]),
        Chromosome(genes=genes_b, parent_ids=[parent_a.id, parent_b.id]),
    )


def mutate(
    chromosome: Chromosome,
    backend: str,
    mutation_rate: float,
    rng: random.Random,
    gene_space=None,
    _gene_space_dict: dict = None,
) -> Chromosome:
    """
    Replace each gene with a random value from its domain with probability
    mutation_rate.

    Parameters
    ----------
    gene_space : list[GeneDefinition] | None
        Pre-built gene space. None = default for backend.
    _gene_space_dict : dict | None
        Pre-built {name: GeneDefinition} lookup. Pass this from GeneticEngine
        to avoid rebuilding the dict on every call.
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

    return Chromosome(genes=new_genes, parent_ids=[chromosome.id])
