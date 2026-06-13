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

import pandas as pd
from sklearn.utils.parallel import Parallel, delayed

try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

import os

import joblib

from genetic_automl.config import GeneticConfig
from genetic_automl.genetic.chromosome import (
    Chromosome,
    build_gene_space_from_config,
    random_population,
)
from genetic_automl.genetic.diversity import PopulationDiversity
from genetic_automl.genetic.fitness import FitnessEvaluator
from genetic_automl.genetic.nsga2 import (
    build_objective_values,
    crowding_distance_assignment,
    fast_non_dominated_sort,
    nsga2_select,
    nsga2_survive,
    pareto_front_summary,
)
from genetic_automl.genetic.operators import (
    elites,
    mutate,
    single_point_crossover,
    tournament_selection,
    uniform_crossover,
)
from genetic_automl.genetic.surrogate import SurrogateModel
from genetic_automl.genetic.warm_start import WarmStart
from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)

# Sentinel: no stagnation limit when adaptive mutation is disabled.
_NO_STAGNATION_LIMIT = 999_999


@dataclass(slots=True)
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
    best_chromosome: Chromosome | None = None


@dataclass(slots=True)
class EvolutionHistory:
    generations: list[GenerationStats] = field(default_factory=list)
    all_chromosomes: list[Chromosome] = field(default_factory=list)
    pareto_front: list[dict] = field(default_factory=list)
    """Pareto-front summary populated when nsga2_enabled=True."""

    @property
    def best(self) -> Chromosome | None:
        evaluated = [c for c in self.all_chromosomes if c.fitness is not None]
        if not evaluated:
            return None
        return max(evaluated, key=lambda c: c.fitness)

    def fitness_curve(self) -> list[float]:
        return [g.best_fitness for g in self.generations]

    def diversity_curve(self) -> list[float]:
        return [g.mean_hamming for g in self.generations]

    def mutation_rate_curve(self) -> list[float]:
        return [g.mutation_rate for g in self.generations]

    def top_chromosomes(self, k: int) -> list[Chromosome]:
        """
        Return the top-k unique chromosomes by fitness (best first).

        "Unique" means distinct gene dictionaries — duplicate configs that
        happened to be evaluated multiple times are deduplicated so the
        ensemble members cover different regions of the search space.

        Parameters
        ----------
        k : int
            Maximum number of chromosomes to return.  If fewer unique
            evaluated chromosomes exist, all of them are returned.
        """
        evaluated = [c for c in self.all_chromosomes if c.fitness is not None]
        if not evaluated:
            return []

        ranked = sorted(evaluated, key=lambda c: c.fitness, reverse=True)

        seen: set = set()
        unique: list[Chromosome] = []
        for chrom in ranked:
            key = str(sorted(chrom.genes.items()))
            if key not in seen:
                seen.add(key)
                unique.append(chrom)
            if len(unique) >= k:
                break
        return unique


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
        gene_space_overrides: dict[str, list] | None = None,
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
                if genetic_config.adaptive_mutation else _NO_STAGNATION_LIMIT
            ),
            mutation_boost_factor=genetic_config.adaptive_mutation_boost_factor,
            mutation_decay=genetic_config.adaptive_mutation_decay,
            random_seed=genetic_config.random_seed,
            gene_space=self._gene_space,
        )

        if genetic_config.surrogate_enabled:
            surrogate = SurrogateModel(
                model_type=genetic_config.surrogate_model_type,
                backend_for_ga=backend,
                min_samples=genetic_config.surrogate_min_samples,
                uncertainty_threshold=genetic_config.surrogate_uncertainty_threshold,
                random_seed=genetic_config.random_seed,
            )
            self.evaluator.surrogate = surrogate
            log.info(
                "Surrogate enabled | model=%s | min_samples=%d",
                genetic_config.surrogate_model_type,
                genetic_config.surrogate_min_samples,
            )
        else:
            log.info("Surrogate disabled.")

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
        start_gen = 0

        if cfg.resume_from_checkpoint:
            state = self._load_checkpoint(cfg.resume_from_checkpoint)
            if state is not None:
                population          = state["population"]
                self.history        = state["history"]
                no_improvement_streak = state["no_improvement_streak"]
                best_fitness_so_far = state["best_fitness_so_far"]
                start_gen           = state["next_generation"]
                log.info(
                    "Resumed from checkpoint '%s' | starting at generation %d",
                    cfg.resume_from_checkpoint, start_gen + 1,
                )

        gen_range = range(start_gen, cfg.generations)
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

            # Update surrogate on all chromosomes evaluated so far.
            if self.evaluator.surrogate is not None:
                self.evaluator.surrogate.update(self.history.all_chromosomes)

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

            # Checkpoint save
            if cfg.checkpoint_dir and (gen_idx + 1) % cfg.checkpoint_every == 0:
                self._save_checkpoint(
                    cfg.checkpoint_dir, gen_idx, population,
                    no_improvement_streak, best_fitness_so_far,
                )

            if no_improvement_streak >= cfg.early_stopping_rounds:
                log.info("Early stopping triggered at generation %d.", gen_idx + 1)
                break

            if gen_idx < cfg.generations - 1:
                # Compute multi-objective values for NSGA-II selection
                obj_vals = None
                if cfg.nsga2_enabled:
                    objectives = cfg.nsga2_objectives or [self.evaluator.metric, "complexity"]
                    obj_vals = build_objective_values(population, objectives)
                population = self._breed(
                    population, gen_idx + 1, current_mut_rate,
                    objective_values=obj_vals,
                )

        if cfg.nsga2_enabled:
            objectives = cfg.nsga2_objectives or [self.evaluator.metric, "complexity"]
            self.history.pareto_front = pareto_front_summary(
                self.history.all_chromosomes, objectives
            )
            log.info(
                "Pareto front size: %d | objectives: %s",
                len(self.history.pareto_front), objectives,
            )

        best = self.history.best
        div_summary = self._diversity.summary()
        log.info(
            "Evolution complete | best=%.6f | diversity_injections=%d | mutation_boosts=%d",
            best.fitness,
            div_summary.get("n_injections_total", 0),
            div_summary.get("n_boosts_total", 0),
        )
        self._log_leaderboard(top_n=5)
        ev_summary = self.evaluator.evaluator_summary()
        log.info(
            "Evaluator stats | asha_prunes=%d | fold_pool=%d",
            ev_summary.get("asha_prunes", 0),
            ev_summary.get("asha_fold_pool_size", 0),
        )
        if "surrogate" in ev_summary:
            s = ev_summary["surrogate"]
            log.info(
                "Surrogate stats | model=%s | skips=%d / %d | skip_rate=%.1f%%",
                s["model_type"], s["skips"], s["total_candidates"],
                s["skip_rate"] * 100,
            )
        return best

    def _save_checkpoint(
        self,
        checkpoint_dir: str,
        gen_idx: int,
        population: list,
        no_improvement_streak: int,
        best_fitness_so_far: float,
    ) -> None:
        """Persist current evolution state to disk as a joblib file."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f"checkpoint_gen{gen_idx + 1:04d}.joblib")
        state = {
            "population": population,
            "history": self.history,
            "no_improvement_streak": no_improvement_streak,
            "best_fitness_so_far": best_fitness_so_far,
            "next_generation": gen_idx + 1,
            "fitness_cache": self.evaluator._cache,
            "all_fold_scores": self.evaluator._all_fold_scores,
        }
        joblib.dump(state, path)
        log.info("Checkpoint saved to '%s'", path)

    def _load_checkpoint(self, path: str) -> dict:
        """Restore evolution state from a checkpoint file. Returns None on failure."""
        if not os.path.exists(path):
            log.warning("Checkpoint file not found: '%s' — starting fresh.", path)
            return None
        try:
            state = joblib.load(path)
            # Restore fitness cache so we don't re-evaluate chromosomes we already scored
            self.evaluator._cache.update(state.get("fitness_cache", {}))
            self.evaluator._all_fold_scores.extend(state.get("all_fold_scores", []))
            log.info(
                "Checkpoint loaded | gen=%d | cache_entries=%d | fold_scores=%d",
                state["next_generation"],
                len(self.evaluator._cache),
                len(self.evaluator._all_fold_scores),
            )
            return state
        except Exception as exc:
            log.warning("Failed to load checkpoint '%s': %s — starting fresh.", path, exc)
            return None

    def _evaluate_population(
        self,
        population: list[Chromosome],
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
    ) -> list[Chromosome]:
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
        population: list[Chromosome],
        next_gen: int,
        mutation_rate: float,
        objective_values: dict | None = None,
    ) -> list[Chromosome]:
        """
        Produce the next generation using selection, crossover, and mutation.

        When nsga2_enabled=True:
          - Uses nsga2_select (rank + crowding-distance tournament) instead of
            fitness-only tournament selection.
          - After generating offspring, applies nsga2_survive to select the
            best n individuals from combined parents + offspring.

        When nsga2_enabled=False (default):
          - Standard elitism + fitness tournament selection.
        """
        use_nsga2 = self.cfg.nsga2_enabled and objective_values is not None

        crossover_fn = (
            uniform_crossover
            if self.cfg.crossover_type == "uniform"
            else single_point_crossover
        )

        if use_nsga2:
            # NSGA-II: stamp ranks and crowding distances on current population
            fronts = fast_non_dominated_sort(population, objective_values)
            n_obj = len(next(iter(objective_values.values()), []))
            for front in fronts:
                crowding_distance_assignment(front, objective_values, n_obj)

            # Generate offspring (same size as population)
            offspring: list[Chromosome] = []
            while len(offspring) < self.cfg.population_size:
                if self._rng.random() < self.cfg.crossover_rate:
                    parent_a = nsga2_select(population, self._rng)
                    parent_b = nsga2_select(population, self._rng)
                    child_a, child_b = crossover_fn(parent_a, parent_b, self._rng)
                    for child in (child_a, child_b):
                        if len(offspring) < self.cfg.population_size:
                            child = mutate(child, self.backend, mutation_rate,
                                           self._rng, self._gene_space, self._gene_space_dict)
                            child.generation = next_gen
                            offspring.append(child)
                else:
                    parent = nsga2_select(population, self._rng)
                    child = mutate(parent, self.backend, mutation_rate,
                                   self._rng, self._gene_space, self._gene_space_dict)
                    child.generation = next_gen
                    offspring.append(child)

            # Environmental selection: survive from combined pool
            combined = population + offspring
            survived = nsga2_survive(combined, self.cfg.population_size,
                                     objective_values, n_obj)
            return survived[: self.cfg.population_size]

        # Standard single-objective breeding
        new_pop: list[Chromosome] = []
        elite_individuals = elites(population, self.cfg.elite_ratio)
        new_pop.extend(elite_individuals)

        while len(new_pop) < self.cfg.population_size:
            if self._rng.random() < self.cfg.crossover_rate:
                parent_a = tournament_selection(population, self.cfg.tournament_size, self._rng)
                parent_b = tournament_selection(population, self.cfg.tournament_size, self._rng)
                child_a, child_b = crossover_fn(parent_a, parent_b, self._rng)
                for child in (child_a, child_b):
                    if len(new_pop) < self.cfg.population_size:
                        child = mutate(child, self.backend, mutation_rate,
                                       self._rng, self._gene_space, self._gene_space_dict)
                        child.generation = next_gen
                        new_pop.append(child)
            else:
                parent = tournament_selection(population, self.cfg.tournament_size, self._rng)
                child = mutate(parent, self.backend, mutation_rate,
                               self._rng, self._gene_space, self._gene_space_dict)
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
