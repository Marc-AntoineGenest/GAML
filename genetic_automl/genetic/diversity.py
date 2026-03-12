"""
PopulationDiversity
-------------------
Tracks genetic diversity of the population across generations and triggers
two counter-measures when convergence is detected:

1. Diversity injection
   When mean pairwise Hamming distance drops below *min_diversity_threshold*,
   a fraction of the worst individuals are replaced with fresh random
   chromosomes. This re-introduces exploration without discarding the
   best solutions found so far (elites are never touched).

2. Adaptive mutation rate
   When no fitness improvement is seen for *stagnation_rounds* generations,
   the mutation rate is temporarily increased by *mutation_boost_factor*
   (e.g. 0.2 → 0.5). Once improvement resumes, the rate decays back toward
   the original value via exponential decay.

Why Hamming distance?
   For categorical/discrete gene spaces, Hamming distance (fraction of genes
   that differ between two chromosomes) is the natural diversity metric.
   A distance of 0 means two chromosomes are identical; 1.0 means every
   gene differs.

Population collapse indicator:
   mean_hamming < min_diversity_threshold  →  population has converged
   Typical threshold: 0.15 (less than 15% of genes differ on average)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from genetic_automl.genetic.chromosome import Chromosome, get_gene_space, random_population
from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Hamming distance utilities
# ---------------------------------------------------------------------------

def _encode_population(population: List[Chromosome]) -> np.ndarray:
    """
    Encode a population as an (N, G) integer matrix for vectorised distance
    computation. Each gene value is mapped to its position in the sorted list
    of unique values seen across the population for that gene.

    Returns
    -------
    np.ndarray of shape (N, G), dtype int32
    """
    if not population:
        return np.empty((0, 0), dtype=np.int32)
    gene_keys = list(population[0].genes.keys())
    n, g = len(population), len(gene_keys)
    matrix = np.zeros((n, g), dtype=np.int32)
    for col, key in enumerate(gene_keys):
        # Map each distinct value to a stable integer
        seen: dict = {}
        for row, chrom in enumerate(population):
            val = chrom.genes.get(key)
            if val not in seen:
                seen[val] = len(seen)
            matrix[row, col] = seen[val]
    return matrix


def hamming_distance(a: Chromosome, b: Chromosome) -> float:
    """
    Fraction of genes that differ between two chromosomes.
    Returns 0.0 (identical) to 1.0 (completely different).
    """
    keys = list(a.genes.keys())
    if not keys:
        return 0.0
    diffs = sum(1 for k in keys if a.genes.get(k) != b.genes.get(k))
    return diffs / len(keys)


def mean_pairwise_hamming(population: List[Chromosome]) -> float:
    """
    Compute mean pairwise Hamming distance across the whole population.

    Vectorised implementation: encodes chromosomes as integer rows, then uses
    numpy broadcasting to compute all pairwise distances in one C-level pass.
    Falls back to the scalar loop for populations smaller than 3.

    Returns value in [0, 1].
    """
    n = len(population)
    if n < 2:
        return 1.0  # trivially diverse

    matrix = _encode_population(population)           # (N, G)
    # Broadcast: matrix[i] != matrix[j]  →  (N, N, G) bool
    diff = matrix[:, None, :] != matrix[None, :, :]   # (N, N, G)
    pairwise = diff.mean(axis=2)                       # (N, N) float, diagonal = 0
    # Upper-triangle mean (exclude self-comparisons)
    i_upper, j_upper = np.triu_indices(n, k=1)
    return float(pairwise[i_upper, j_upper].mean())


def _pairwise_matrix(population: List[Chromosome]) -> np.ndarray:
    """Return full (N, N) pairwise Hamming matrix (vectorised)."""
    matrix = _encode_population(population)
    diff = matrix[:, None, :] != matrix[None, :, :]
    return diff.mean(axis=2)


# ---------------------------------------------------------------------------
# DiversityStats (per generation snapshot)
# ---------------------------------------------------------------------------

@dataclass
class DiversityStats:
    generation: int
    mean_hamming: float
    min_hamming: float
    max_hamming: float
    injection_triggered: bool = False
    mutation_boosted: bool = False
    current_mutation_rate: float = 0.2


# ---------------------------------------------------------------------------
# PopulationDiversity controller
# ---------------------------------------------------------------------------

class PopulationDiversity:
    """
    Monitors diversity and applies counter-measures when needed.

    Parameters
    ----------
    backend : str
        Needed for random_population factory.
    base_mutation_rate : float
        The original mutation rate from GeneticConfig.
    min_diversity_threshold : float
        Mean Hamming distance below which injection is triggered (default 0.15).
    injection_ratio : float
        Fraction of the worst individuals replaced on injection (default 0.2).
    stagnation_rounds : int
        No-improvement streak that triggers mutation boost (default 3).
    mutation_boost_factor : float
        Multiply base mutation rate by this on stagnation (default 2.5).
    mutation_decay : float
        After boost, decay rate back toward base each generation (default 0.85).
    random_seed : int
    """

    def __init__(
        self,
        backend: str,
        base_mutation_rate: float = 0.2,
        min_diversity_threshold: float = 0.15,
        injection_ratio: float = 0.2,
        stagnation_rounds: int = 3,
        mutation_boost_factor: float = 2.5,
        mutation_decay: float = 0.85,
        random_seed: int = 42,
        gene_space=None,
    ) -> None:
        self.backend = backend
        self.base_mutation_rate = base_mutation_rate
        self.min_diversity_threshold = min_diversity_threshold
        self.injection_ratio = injection_ratio
        self.stagnation_rounds = stagnation_rounds
        self.mutation_boost_factor = mutation_boost_factor
        self.mutation_decay = mutation_decay
        self._rng = random.Random(random_seed)
        self._gene_space = gene_space  # None → default space used by random_population

        self._current_mutation_rate = base_mutation_rate
        self._boosted = False
        self.history: List[DiversityStats] = []

    # ------------------------------------------------------------------

    @property
    def current_mutation_rate(self) -> float:
        return self._current_mutation_rate

    def update(
        self,
        population: List[Chromosome],
        generation: int,
        no_improvement_streak: int,
    ) -> Tuple[List[Chromosome], float]:
        """
        Assess diversity and apply counter-measures if needed.

        Called AFTER fitness evaluation and BEFORE breeding.

        Returns
        -------
        (possibly_modified_population, current_mutation_rate)
        """
        mean_h = mean_pairwise_hamming(population)
        fitnesses = [c.fitness for c in population if c.fitness is not None]
        min_h = self._min_hamming(population)
        max_h = self._max_hamming(population)

        injection_triggered = False
        mutation_boosted = False

        # ── 1. Diversity injection ─────────────────────────────────────
        if mean_h < self.min_diversity_threshold:
            population = self._inject_diversity(population)
            injection_triggered = True
            log.warning(
                "Diversity injection triggered at gen %d | mean_hamming=%.3f < threshold=%.3f | "
                "replaced %.0f%% of worst individuals",
                generation, mean_h, self.min_diversity_threshold,
                self.injection_ratio * 100,
            )

        # ── 2. Adaptive mutation ───────────────────────────────────────
        if no_improvement_streak >= self.stagnation_rounds:
            if not self._boosted:
                self._current_mutation_rate = min(
                    0.8,
                    self.base_mutation_rate * self.mutation_boost_factor,
                )
                self._boosted = True
                mutation_boosted = True
                log.warning(
                    "Adaptive mutation boost at gen %d | stagnation=%d rounds | "
                    "rate %.3f → %.3f",
                    generation, no_improvement_streak,
                    self.base_mutation_rate, self._current_mutation_rate,
                )
        else:
            # Improvement detected — decay back toward base rate
            if self._boosted:
                self._current_mutation_rate = max(
                    self.base_mutation_rate,
                    self._current_mutation_rate * self.mutation_decay,
                )
                if abs(self._current_mutation_rate - self.base_mutation_rate) < 0.01:
                    self._current_mutation_rate = self.base_mutation_rate
                    self._boosted = False
                    log.info(
                        "Adaptive mutation decayed back to base rate %.3f at gen %d",
                        self.base_mutation_rate, generation,
                    )

        stats = DiversityStats(
            generation=generation,
            mean_hamming=mean_h,
            min_hamming=min_h,
            max_hamming=max_h,
            injection_triggered=injection_triggered,
            mutation_boosted=mutation_boosted,
            current_mutation_rate=self._current_mutation_rate,
        )
        self.history.append(stats)

        log.info(
            "Diversity gen %d | mean_hamming=%.3f | mutation_rate=%.3f%s",
            generation,
            mean_h,
            self._current_mutation_rate,
            " [BOOSTED]" if self._boosted else "",
        )
        return population, self._current_mutation_rate

    # ------------------------------------------------------------------

    def _inject_diversity(self, population: List[Chromosome]) -> List[Chromosome]:
        """
        Replace the bottom *injection_ratio* fraction with new random individuals.
        Elites (top individuals) are never replaced.
        """
        n_inject = max(1, int(len(population) * self.injection_ratio))

        # Sort: best first
        sorted_pop = sorted(
            population,
            key=lambda c: c.fitness if c.fitness is not None else float("-inf"),
            reverse=True,
        )

        # Keep top (N - n_inject), replace bottom n_inject
        survivors = sorted_pop[:-n_inject]
        fresh = random_population(
            backend=self.backend,
            size=n_inject,
            rng=self._rng,
            generation=sorted_pop[0].generation,  # same gen label
            gene_space=self._gene_space,
        )
        return survivors + fresh

    def _min_hamming(self, population: List[Chromosome]) -> float:
        n = len(population)
        if n < 2:
            return 1.0
        pw = _pairwise_matrix(population)
        i_upper, j_upper = np.triu_indices(n, k=1)
        return float(pw[i_upper, j_upper].min())

    def _max_hamming(self, population: List[Chromosome]) -> float:
        n = len(population)
        if n < 2:
            return 0.0
        pw = _pairwise_matrix(population)
        i_upper, j_upper = np.triu_indices(n, k=1)
        return float(pw[i_upper, j_upper].max())

    def summary(self) -> dict:
        """Return diversity history summary for HTML report."""
        if not self.history:
            return {}
        return {
            "generations": [s.generation for s in self.history],
            "mean_hamming": [round(s.mean_hamming, 4) for s in self.history],
            "mutation_rates": [round(s.current_mutation_rate, 4) for s in self.history],
            "injections": [s.injection_triggered for s in self.history],
            "boosts": [s.mutation_boosted for s in self.history],
            "n_injections_total": sum(s.injection_triggered for s in self.history),
            "n_boosts_total": sum(s.mutation_boosted for s in self.history),
        }
