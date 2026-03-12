"""
GeneticEngine — orchestrates the full evolution loop.

Generation flow (per generation):
  1. Evaluate unevaluated individuals via k-fold CV
  2. Compute generation stats (best / mean / worst fitness)
  3. Update no-improvement streak
  4. Diversity check: inject fresh individuals if Hamming distance is too low
  5. Adaptive mutation: boost rate on stagnation, decay on improvement
  6. Record stats
  7. Early stopping check
  8. Breed next generation
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
from sklearn.utils.parallel import Parallel, delayed

try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

from genetic_automl.config import GeneticConfig
from genetic_automl.genetic.chromosome import Chromosome, build_gene_space_from_config, random_population
from genetic_automl.genetic.diversity import PopulationDiversity
from genetic_automl.genetic.fitness import FitnessEvaluator
from genetic_automl.genetic.operators import (
    elites,
    mutate,
    single_point_crossover,
    uniform_crossover,
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
    gene_space_overrides : dict, optional
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
        self._gene_space = build_gene_space_from_config(backend, gene_space_overrides or {})
        self._gene_space_dict = {g.name: g for g in self._gene_space}

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

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
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

        population = self._build_initial_population(X_train, y_train)

        no_improvement_streak = 0
        best_fitness_so_far = float("-inf")

        gen_range = range(cfg.generations)
        pbar = (
            _tqdm(gen_range, desc="Evolution", unit="gen", dynamic_ncols=True)
            if _TQDM_AVAILABLE else gen_range
        )

        for gen_idx in pbar:
            gen_start = time.perf_counter()
            log.info("Generation %d / %d", gen_idx + 1, cfg.generations)

            # Evaluate unevaluated individuals
            unevaluated = [c for c in population if c.fitness is None]
            if unevaluated:
                self._evaluate_population(unevaluated, X_train, y_train)
                for chrom in unevaluated:
                    self.history.all_chromosomes.append(chrom)
                log.info(
                    "Gen %d | evaluated=%d | cache_hits=%d",
                    gen_idx + 1, len(unevaluated), self.evaluator._cache_hits,
                )

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

            if best_fit > best_fitness_so_far:
                best_fitness_so_far = best_fit
                no_improvement_streak = 0
            else:
                no_improvement_streak += 1
                log.info(
                    "No improvement for %d / %d rounds",
                    no_improvement_streak, cfg.early_stopping_rounds,
                )

            population, current_mut_rate = self._diversity.update(
                population, gen_idx, no_improvement_streak,
            )
            div_stats = self._diversity.history[-1]

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

            if _TQDM_AVAILABLE and hasattr(pbar, "set_postfix"):
                pbar.set_postfix(
                    best=f"{best_fit:.4f}",
                    mut=f"{current_mut_rate:.2f}",
                    stale=no_improvement_streak,
                    refresh=True,
                )

            if no_improvement_streak >= cfg.early_stopping_rounds:
                log.info("Early stopping triggered at generation %d.", gen_idx + 1)
                break

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
        self._log_leaderboard(top_n=5)
        return best

    def _evaluate_population(
        self,
        population: List[Chromosome],
        X_train: pd.DataFrame,
        y_train,
    ) -> None:
        """
        Evaluate all chromosomes, writing fitness back in-place.

        n_jobs=1 (default): sequential, cache fully effective.
        n_jobs!=1: joblib loky workers. The fitness cache is not shared across
        workers — each process gets its own copy. Elites that already have
        fitness set are skipped before this method is called, so they are
        unaffected.
        """
        if self.cfg.n_jobs == 1:
            for chrom in population:
                self.evaluator.evaluate(chrom, X_train, y_train)
            return

        def _worker(chrom: Chromosome) -> tuple:
            fitness = self.evaluator.evaluate(chrom, X_train, y_train)
            return chrom.id, fitness, chrom.fitness_std

        results = Parallel(n_jobs=self.cfg.n_jobs, backend="loky", prefer="processes")(
            delayed(_worker)(chrom) for chrom in population
        )
        result_map = {r[0]: (r[1], r[2]) for r in results}
        for chrom in population:
            if chrom.id in result_map:
                chrom.fitness, chrom.fitness_std = result_map[chrom.id]

    def _build_initial_population(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> List[Chromosome]:
        """Build generation 0 with warm-start or fall back to pure random."""
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

    def _breed(
        self,
        population: List[Chromosome],
        next_gen: int,
        mutation_rate: float,
    ) -> List[Chromosome]:
        """Produce the next generation using selection, crossover, and mutation."""
        new_pop: List[Chromosome] = []

        elite_individuals = elites(population, self.cfg.elite_ratio)
        new_pop.extend(elite_individuals)

        crossover_fn = (
            uniform_crossover
            if self.cfg.crossover_type == "uniform"
            else single_point_crossover
        )
        while len(new_pop) < self.cfg.population_size:
            if self._rng.random() < self.cfg.crossover_rate:
                parent_a = tournament_selection(population, self.cfg.tournament_size, self._rng)
                parent_b = tournament_selection(population, self.cfg.tournament_size, self._rng)
                child_a, child_b = crossover_fn(parent_a, parent_b, self._rng)
                for child in (child_a, child_b):
                    if len(new_pop) < self.cfg.population_size:
                        child = mutate(child, self.backend, mutation_rate, self._rng, self._gene_space, self._gene_space_dict)
                        child.generation = next_gen
                        new_pop.append(child)
            else:
                parent = tournament_selection(population, self.cfg.tournament_size, self._rng)
                child = mutate(parent, self.backend, mutation_rate, self._rng, self._gene_space, self._gene_space_dict)
                child.generation = next_gen
                new_pop.append(child)

        return new_pop[: self.cfg.population_size]

    def _log_leaderboard(self, top_n: int = 5) -> None:
        """Log the top-N unique chromosomes found across the entire run."""
        evaluated = [c for c in self.history.all_chromosomes if c.fitness is not None]
        if not evaluated:
            return
        seen: dict = {}
        for c in evaluated:
            key = tuple(sorted(c.genes.items()))
            if key not in seen or c.fitness > seen[key].fitness:
                seen[key] = c
        ranked = sorted(seen.values(), key=lambda c: c.fitness, reverse=True)[:top_n]

        sep = "-" * 72
        log.info(sep)
        log.info("  TOP-%d LEADERBOARD", min(top_n, len(ranked)))
        log.info(sep)
        log.info("  %-4s  %-10s  %-8s  %-10s  Key genes", "Rank", "Fitness", "Std", "ID")
        log.info(sep)
        for rank, c in enumerate(ranked, 1):
            std_str = f"{c.fitness_std:.4f}" if c.fitness_std is not None else "n/a"
            key_genes = {
                k: v for k, v in c.genes.items()
                if k in ("scaler", "numeric_imputer", "categorical_encoder",
                         "imbalance_method", "n_estimators", "presets")
            }
            log.info(
                "  %-4d  %-10.6f  %-8s  %-10s  %s",
                rank, c.fitness, std_str, c.id, key_genes,
            )
        log.info(sep)

    def diversity_summary(self) -> dict:
        return self._diversity.summary()
