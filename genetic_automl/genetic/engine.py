"""
GeneticEngine — orchestrates the full evolution loop.

Generation flow:
  0. Warm-start: build gen-0 via default seeds + halving pre-screen
  --- per generation ---
  1. Evaluate unevaluated individuals (k-fold CV via FitnessEvaluator)
  2. Compute stats (best / mean / worst fitness)
  3. Update improvement streak — BEFORE diversity so controller sees correct value
     → reset to 0 on improvement; increment on stagnation
  4. Diversity check: compute mean Hamming distance
     → inject fresh individuals if below threshold
     → boost mutation rate if stagnating; decay back on improvement
  5. Early stopping check
  6. Breed next generation using current (adaptive) mutation rate
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from genetic_automl.config import GeneticConfig
from genetic_automl.genetic.chromosome import Chromosome, build_gene_space_from_config, random_population
from genetic_automl.genetic.diversity import PopulationDiversity
from genetic_automl.genetic.fitness import FitnessEvaluator
from genetic_automl.genetic.operators import (
    elites,
    mutate,
    single_point_crossover,
    tournament_selection,
)
from genetic_automl.genetic.warm_start import WarmStart
from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class GenerationStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    elapsed_seconds: float
    mean_hamming: float = 0.0
    mutation_rate: float = 0.2
    diversity_injected: bool = False
    mutation_boosted: bool = False
    best_chromosome: Optional[Chromosome] = None


@dataclass
class EvolutionHistory:
    generations: List[GenerationStats] = field(default_factory=list)
    all_chromosomes: List[Chromosome] = field(default_factory=list)

    @property
    def best(self) -> Optional[Chromosome]:
        evaluated = [c for c in self.all_chromosomes if c.fitness is not None]
        if not evaluated:
            return None
        return max(evaluated, key=lambda c: c.fitness)

    def fitness_curve(self) -> List[float]:
        return [g.best_fitness for g in self.generations]

    def diversity_curve(self) -> List[float]:
        return [g.mean_hamming for g in self.generations]

    def mutation_rate_curve(self) -> List[float]:
        return [g.mutation_rate for g in self.generations]


class GeneticEngine:
    """
    Runs the genetic algorithm with warm-start, diversity injection,
    and adaptive mutation.

    Parameters
    ----------
    genetic_config : GeneticConfig
    evaluator : FitnessEvaluator
    backend : str
    """

    def __init__(
        self,
        genetic_config: GeneticConfig,
        evaluator: FitnessEvaluator,
        backend: str = "autogluon",
        gene_space_overrides: Optional[Dict[str, list]] = None,
    ) -> None:
        self.cfg = genetic_config
        self.evaluator = evaluator
        self.backend = backend
        self._rng = random.Random(genetic_config.random_seed)
        self.history = EvolutionHistory()
        # Resolve gene space once — used by all population/mutation operations
        self._gene_space = build_gene_space_from_config(backend, gene_space_overrides or {})

        # Diversity controller (always instantiated; disabled by threshold=0 if needed)
        self._diversity = PopulationDiversity(
            backend=backend,
            base_mutation_rate=genetic_config.mutation_rate,
            min_diversity_threshold=genetic_config.diversity_threshold,
            injection_ratio=genetic_config.diversity_injection_ratio,
            stagnation_rounds=(
                genetic_config.adaptive_mutation_stagnation_rounds
                if genetic_config.adaptive_mutation else 999_999
            ),
            mutation_boost_factor=genetic_config.adaptive_mutation_boost_factor,
            mutation_decay=genetic_config.adaptive_mutation_decay,
            random_seed=genetic_config.random_seed,
            gene_space=self._gene_space,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,   # kept for API compat
        y_val: pd.Series = None,
    ) -> Chromosome:
        """Evolve the population and return the best chromosome found."""
        cfg = self.cfg
        log.info(
            "GeneticEngine | pop=%d | gens=%d | cv_folds=%d | backend=%s | "
            "warm_start=%s | adaptive_mutation=%s",
            cfg.population_size, cfg.generations, cfg.n_cv_folds,
            self.backend, cfg.warm_start, cfg.adaptive_mutation,
        )

        # ── Step 0: build initial population ──────────────────────────
        population = self._build_initial_population(X_train, y_train)

        no_improvement_streak = 0
        best_fitness_so_far = float("-inf")

        for gen_idx in range(cfg.generations):
            gen_start = time.perf_counter()
            log.info("── Generation %d / %d ──", gen_idx + 1, cfg.generations)

            # ── Step 1: evaluate unevaluated individuals ───────────────
            for chrom in population:
                if chrom.fitness is None:
                    self.evaluator.evaluate(chrom, X_train, y_train)
                    self.history.all_chromosomes.append(chrom)

            # ── Step 2: compute stats ──────────────────────────────────
            valid = [c for c in population if c.fitness is not None]
            fitnesses = [c.fitness for c in valid]
            best_fit   = max(fitnesses)
            mean_fit   = sum(fitnesses) / len(fitnesses)
            worst_fit  = min(fitnesses)
            elapsed    = time.perf_counter() - gen_start
            best_chrom = max(valid, key=lambda c: c.fitness)

            log.info(
                "Gen %d | best=%.6f | mean=%.6f | worst=%.6f | %.1fs",
                gen_idx + 1, best_fit, mean_fit, worst_fit, elapsed,
            )

            # ── Step 3: update improvement streak (must happen before diversity
            #    so the controller sees the correct stale-free streak value) ──
            if best_fit > best_fitness_so_far:
                best_fitness_so_far = best_fit
                no_improvement_streak = 0
            else:
                no_improvement_streak += 1
                log.info(
                    "No improvement for %d / %d rounds",
                    no_improvement_streak, cfg.early_stopping_rounds,
                )

            # ── Step 4: diversity + adaptive mutation ──────────────────
            population, current_mut_rate = self._diversity.update(
                population, gen_idx, no_improvement_streak,
            )
            div_stats = self._diversity.history[-1]

            # ── Step 5: record stats ───────────────────────────────────
            self.history.generations.append(GenerationStats(
                generation=gen_idx,
                best_fitness=best_fit,
                mean_fitness=mean_fit,
                worst_fitness=worst_fit,
                elapsed_seconds=elapsed,
                mean_hamming=div_stats.mean_hamming,
                mutation_rate=current_mut_rate,
                diversity_injected=div_stats.injection_triggered,
                mutation_boosted=div_stats.mutation_boosted,
                best_chromosome=best_chrom,
            ))

            # ── Step 6: early stopping ─────────────────────────────────
            if no_improvement_streak >= cfg.early_stopping_rounds:
                log.info("Early stopping triggered at generation %d.", gen_idx + 1)
                break

            # ── Step 7: breed next generation ─────────────────────────
            if gen_idx < cfg.generations - 1:
                population = self._breed(population, gen_idx + 1, current_mut_rate)

        best = self.history.best
        div_summary = self._diversity.summary()
        log.info(
            "Evolution complete | best=%.6f | diversity_injections=%d | mutation_boosts=%d",
            best.fitness,
            div_summary.get("n_injections_total", 0),
            div_summary.get("n_boosts_total", 0),
        )
        log.info("Best genes: %s", best.genes)
        return best

    # ------------------------------------------------------------------
    # Population initialisation
    # ------------------------------------------------------------------

    def _build_initial_population(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> List[Chromosome]:
        """Build gen-0 with warm-start or fall back to pure random."""
        if not self.cfg.warm_start:
            log.info("Warm-start disabled — using pure random population")
            return random_population(
                backend=self.backend,
                size=self.cfg.population_size,
                rng=self._rng,
                generation=0,
                gene_space=self._gene_space,
            )

        ws = WarmStart(
            backend=self.backend,
            n_default_seeds=self.cfg.warm_start_n_seeds,
            halving_pool_ratio=self.cfg.warm_start_halving_pool_ratio,
            halving_keep_ratio=self.cfg.warm_start_halving_keep_ratio,
            random_seed=self.cfg.random_seed,
            gene_space=self._gene_space,
        )
        return ws.build_initial_population(
            population_size=self.cfg.population_size,
            evaluator=self.evaluator,
            X_train=X_train,
            y_train=y_train,
        )

    # ------------------------------------------------------------------
    # Breeding
    # ------------------------------------------------------------------

    def _breed(
        self,
        population: List[Chromosome],
        next_gen: int,
        mutation_rate: float,
    ) -> List[Chromosome]:
        """Produce the next generation using the current (adaptive) mutation rate."""
        new_pop: List[Chromosome] = []

        # Preserve elites unchanged
        elite_individuals = elites(population, self.cfg.elite_ratio)
        new_pop.extend(elite_individuals)

        # Fill remaining slots via crossover + mutation
        while len(new_pop) < self.cfg.population_size:
            if self._rng.random() < self.cfg.crossover_rate:
                parent_a = tournament_selection(population, self.cfg.tournament_size, self._rng)
                parent_b = tournament_selection(population, self.cfg.tournament_size, self._rng)
                child_a, child_b = single_point_crossover(parent_a, parent_b, self._rng)
                for child in (child_a, child_b):
                    if len(new_pop) < self.cfg.population_size:
                        child = mutate(child, self.backend, mutation_rate, self._rng, self._gene_space)
                        child.generation = next_gen
                        new_pop.append(child)
            else:
                parent = tournament_selection(population, self.cfg.tournament_size, self._rng)
                child = mutate(parent, self.backend, mutation_rate, self._rng, self._gene_space)
                child.generation = next_gen
                new_pop.append(child)

        return new_pop[: self.cfg.population_size]

    # ------------------------------------------------------------------
    # Accessors for reporting
    # ------------------------------------------------------------------

    def diversity_summary(self) -> dict:
        return self._diversity.summary()
