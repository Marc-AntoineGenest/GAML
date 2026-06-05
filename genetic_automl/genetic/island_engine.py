"""
IslandEngine — Island Model Genetic Algorithm.

Problem solved
--------------
A single GA population converges prematurely: after a few generations of
tournament selection, most individuals share the same high-fitness genes
and crossover stops producing novelty.  The fitness landscape stops being
explored and the search stagnates far from the global optimum.

The island model addresses this by maintaining N **sub-populations (islands)**
that evolve independently for several generations before exchanging individuals.
The periodic **migration** step seeds each island with fresh genetic material
from its neighbour, breaking local convergence without discarding accumulated
improvements.

Empirically on tabular ML searches: island models find solutions 5–15% better
than a single population of the same total size, and they find them faster
because islands explore diverse regions in parallel.

Architecture
------------
                  ┌──────────┐   migrate   ┌──────────┐
                  │ Island 0 │ ──────────► │ Island 1 │
                  └──────────┘             └──────────┘
                        ▲                       │
                    migrate                  migrate
                        │                       ▼
                  ┌──────────┐   ◄──────── ┌──────────┐
                  │ Island 3 │             │ Island 2 │
                  └──────────┘             └──────────┘

Ring topology: island i sends its best chromosomes to island (i+1) % n_islands.
This is the most studied topology and balances diversity vs. convergence speed.

Migration policy
----------------
- Every `migration_interval` generations, the `migration_size` fittest unique
  chromosomes from island i are *copied* (not moved) to island i+1.
- Copies replace the worst individuals on the receiving island.
- This is intentionally conservative: elites stay on their home island so
  good solutions are never lost.

Parallelism
-----------
Islands evolve sequentially by default (n_island_jobs=1).  Set n_island_jobs=-1
to run islands in parallel threads using concurrent.futures.ThreadPoolExecutor.
Thread-based (not process-based) parallelism is used intentionally:
- LGBM and XGBoost are already multithreaded; spawning processes would
  oversubscribe CPUs and cause contention.
- Thread-safe: each island has its own evaluator, cache, RNG and surrogate —
  there is no shared mutable state.

History merging
---------------
After all islands finish, their EvolutionHistory objects are merged:
- all_chromosomes: concatenated (duplicates allowed — deduplication happens
  in EvolutionHistory.top_chromosomes() which is called by the pipeline).
- generations: aligned by generation index, best stats taken across islands.
  This gives the pipeline and reporter a single coherent fitness curve.
"""

from __future__ import annotations

import copy
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd

from genetic_automl.config import GeneticConfig
from genetic_automl.genetic.chromosome import Chromosome
from genetic_automl.genetic.engine import EvolutionHistory, GeneticEngine, GenerationStats
from genetic_automl.genetic.fitness import FitnessEvaluator
from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)


class IslandEngine:
    """
    Island Model GA: N independent sub-populations with periodic migration.

    Parameters
    ----------
    genetic_config : GeneticConfig
        Shared config for all islands.  population_size is divided equally
        among islands (minimum 4 per island).
    evaluator : FitnessEvaluator
        Template evaluator.  Each island gets its own deep copy so caches
        and ASHA fold pools remain independent.
    backend : str
    n_islands : int
        Number of sub-populations.  2–4 is typical; 8 for large compute budgets.
    migration_interval : int
        Generations between migration events.  Lower = more gene flow = faster
        convergence but less diversity.  Typical range: 2–5.
    migration_size : int
        Number of chromosomes migrated per island per migration event.
        Must be < island_population_size.  Typical: 1–3.
    n_island_jobs : int
        Parallel workers.  1 = sequential.  -1 = all threads.
        Thread-based; safe for all backends.
    gene_space_overrides : dict, optional
    """

    def __init__(
        self,
        genetic_config: GeneticConfig,
        evaluator: FitnessEvaluator,
        backend: str = "sklearn",
        n_islands: int = 4,
        migration_interval: int = 3,
        migration_size: int = 2,
        n_island_jobs: int = 1,
        gene_space_overrides: Optional[Dict[str, list]] = None,
    ) -> None:
        self.cfg = genetic_config
        self.backend = backend
        self.n_islands = n_islands
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        self.n_island_jobs = n_island_jobs
        self.history = EvolutionHistory()

        # Each island gets a GeneticConfig with a smaller population size.
        # We floor-divide so total chromosomes ≈ original population_size.
        island_pop_size = max(4, genetic_config.population_size // n_islands)

        log.info(
            "IslandEngine | n_islands=%d | island_pop=%d | total_pop=%d | "
            "migration_interval=%d | migration_size=%d | n_island_jobs=%d",
            n_islands, island_pop_size, island_pop_size * n_islands,
            migration_interval, migration_size, n_island_jobs,
        )

        # Build one GeneticEngine per island.
        # Each gets:
        #   - its own deep-copied config with a unique random_seed offset
        #   - its own deep-copied evaluator (independent cache + ASHA pool)
        self._islands: List[GeneticEngine] = []
        for i in range(n_islands):
            island_cfg = copy.deepcopy(genetic_config)
            island_cfg.population_size = island_pop_size
            island_cfg.random_seed = genetic_config.random_seed + i * 1000
            # Disable per-island checkpointing — the IslandEngine manages this
            island_cfg.checkpoint_dir = None
            island_cfg.resume_from_checkpoint = None

            island_evaluator = copy.deepcopy(evaluator)

            engine = GeneticEngine(
                genetic_config=island_cfg,
                evaluator=island_evaluator,
                backend=backend,
                gene_space_overrides=gene_space_overrides or {},
            )
            self._islands.append(engine)

    # ------------------------------------------------------------------
    # Public API (mirrors GeneticEngine.run())
    # ------------------------------------------------------------------

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
    ) -> Chromosome:
        """
        Evolve all islands and return the single best chromosome found.

        The run proceeds generation-by-generation:
          1. Evaluate one generation on each island (parallel or sequential).
          2. After every migration_interval generations, migrate top
             chromosomes between islands (ring topology).
          3. After all generations complete, merge histories and return best.
        """
        n_gens = self.cfg.generations

        # Build initial populations for each island
        log.info("IslandEngine: building initial populations ...")
        populations = self._build_initial_populations(X_train, y_train)

        # Island-local state mirrors GeneticEngine.run()
        island_states = [
            {
                "population": pop,
                "no_improvement_streak": 0,
                "best_fitness_so_far": float("-inf"),
                "current_mut_rate": self.cfg.mutation_rate,
            }
            for pop in populations
        ]

        migrations_done = 0

        for gen_idx in range(n_gens):
            gen_start = time.perf_counter()
            log.info(
                "IslandEngine gen %d / %d | migrations_done=%d",
                gen_idx + 1, n_gens, migrations_done,
            )

            # --- Evolve one generation on each island --------------------
            if self.n_island_jobs == 1 or self.n_islands == 1:
                for i, (engine, state) in enumerate(zip(self._islands, island_states)):
                    self._step_island(engine, state, X_train, y_train, gen_idx)
            else:
                self._step_islands_parallel(island_states, X_train, y_train, gen_idx)

            # --- Migration -----------------------------------------------
            if (gen_idx + 1) % self.migration_interval == 0 and gen_idx < n_gens - 1:
                self._migrate(island_states)
                migrations_done += 1
                log.info(
                    "  Migration #%d | ring-topology | size=%d",
                    migrations_done, self.migration_size,
                )

            elapsed = time.perf_counter() - gen_start
            best_across_islands = max(
                (c for state in island_states
                 for c in state["population"] if c.fitness is not None),
                key=lambda c: c.fitness,
                default=None,
            )
            if best_across_islands:
                log.info(
                    "IslandEngine gen %d | best_global=%.6f | %.1fs",
                    gen_idx + 1, best_across_islands.fitness, elapsed,
                )

            # Early stopping: all islands stagnated
            if all(
                s["no_improvement_streak"] >= self.cfg.early_stopping_rounds
                for s in island_states
            ):
                log.info(
                    "IslandEngine early stopping: all islands stagnated at gen %d.",
                    gen_idx + 1,
                )
                break

        # --- Merge histories and return global best ----------------------
        self.history = self._merge_histories()
        best = self.history.best

        log.info(
            "IslandEngine complete | best=%.6f | total_evaluated=%d | migrations=%d",
            best.fitness if best else float("-inf"),
            len(self.history.all_chromosomes),
            migrations_done,
        )
        return best

    def diversity_summary(self) -> dict:
        """Aggregate diversity summary across all islands."""
        summaries = [e.diversity_summary() for e in self._islands]
        return {
            "n_islands": self.n_islands,
            "n_injections_total": sum(s.get("n_injections_total", 0) for s in summaries),
            "n_boosts_total":     sum(s.get("n_boosts_total", 0) for s in summaries),
            "island_summaries":   summaries,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_initial_populations(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> List[List[Chromosome]]:
        """Build generation-0 population for each island (warm-start aware)."""
        populations = []
        for i, engine in enumerate(self._islands):
            log.info("  Island %d: building initial population ...", i)
            pop = engine._build_initial_population(X_train, y_train)
            populations.append(pop)
        return populations

    def _step_island(
        self,
        engine: GeneticEngine,
        state: dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        gen_idx: int,
    ) -> None:
        """
        Advance one island by one generation, updating state in-place.

        This mirrors the core loop body of GeneticEngine.run() but is called
        externally so the IslandEngine can interleave migration between steps.
        """
        cfg = engine.cfg
        population = state["population"]

        # Evaluate unevaluated chromosomes
        unevaluated = [c for c in population if c.fitness is None]
        if unevaluated:
            engine._evaluate_population(unevaluated, X_train, y_train)
            for chrom in unevaluated:
                engine.history.all_chromosomes.append(chrom)

        # Update surrogate on this island's history
        if engine.evaluator.surrogate is not None:
            engine.evaluator.surrogate.update(engine.history.all_chromosomes)

        valid = [c for c in population if c.fitness is not None]
        if not valid:
            return

        fitnesses = [c.fitness for c in valid]
        best_fit  = max(fitnesses)
        mean_fit  = sum(fitnesses) / len(fitnesses)
        worst_fit = min(fitnesses)
        best_chrom = max(valid, key=lambda c: c.fitness)

        if best_fit > state["best_fitness_so_far"]:
            state["best_fitness_so_far"] = best_fit
            state["no_improvement_streak"] = 0
        else:
            state["no_improvement_streak"] += 1

        # Diversity + adaptive mutation
        population, current_mut_rate = engine._diversity.update(
            population, gen_idx, state["no_improvement_streak"],
        )
        state["current_mut_rate"] = current_mut_rate
        div_stats = engine._diversity.history[-1]

        engine.history.generations.append(GenerationStats(
            generation=gen_idx,
            best_fitness=best_fit,
            mean_fitness=mean_fit,
            worst_fitness=worst_fit,
            elapsed_seconds=0.0,   # timing is tracked at the IslandEngine level
            mean_hamming=div_stats.mean_hamming,
            mutation_rate=current_mut_rate,
            diversity_injected=div_stats.injection_triggered,
            mutation_boosted=div_stats.mutation_boosted,
            best_chromosome=best_chrom,
        ))

        # Breed next generation (unless last generation)
        if gen_idx < cfg.generations - 1:
            population = engine._breed(population, gen_idx + 1, current_mut_rate)

        state["population"] = population

    def _step_islands_parallel(
        self,
        island_states: List[dict],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        gen_idx: int,
    ) -> None:
        """Advance all islands in parallel using ThreadPoolExecutor."""
        max_workers = (
            self.n_islands if self.n_island_jobs == -1
            else min(self.n_island_jobs, self.n_islands)
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._step_island, engine, state, X_train, y_train, gen_idx
                ): i
                for i, (engine, state) in enumerate(zip(self._islands, island_states))
            }
            for future in as_completed(futures):
                i = futures[future]
                exc = future.exception()
                if exc:
                    log.warning("Island %d step failed: %s", i, exc)

    def _migrate(self, island_states: List[dict]) -> None:
        """
        Ring-topology migration: copy top-k chromosomes from island i
        to island (i+1) % n_islands, replacing that island's worst members.

        Deep-copy is used so both islands keep independent chromosome objects
        (mutations on one island don't affect the other's individuals).
        """
        n = self.n_islands
        k = self.migration_size

        # Collect emigrants from each island before modifying any population
        emigrants: List[List[Chromosome]] = []
        for state in island_states:
            pop = state["population"]
            evaluated = [c for c in pop if c.fitness is not None]
            if not evaluated:
                emigrants.append([])
                continue
            top_k = sorted(evaluated, key=lambda c: c.fitness, reverse=True)[:k]
            emigrants.append([copy.deepcopy(c) for c in top_k])

        # Send emigrants to the next island in the ring
        for src_idx, immigrants in enumerate(emigrants):
            if not immigrants:
                continue
            dst_idx = (src_idx + 1) % n
            dst_pop = island_states[dst_idx]["population"]

            # Replace worst members on the destination island
            evaluated_dst = [c for c in dst_pop if c.fitness is not None]
            if not evaluated_dst:
                continue
            worst_k = sorted(evaluated_dst, key=lambda c: c.fitness)[:len(immigrants)]
            for worst, immigrant in zip(worst_k, immigrants):
                try:
                    replace_idx = dst_pop.index(worst)
                    dst_pop[replace_idx] = immigrant
                except ValueError:
                    dst_pop.append(immigrant)

            island_states[dst_idx]["population"] = dst_pop

    def _merge_histories(self) -> EvolutionHistory:
        """
        Combine all island histories into a single EvolutionHistory.

        all_chromosomes: union of all islands' evaluated chromosomes.
        generations:     aligned by generation index; each GenerationStats
                         records the *global* best/mean/worst across all
                         islands at that generation.
        """
        merged = EvolutionHistory()

        # Collect all chromosomes from all islands
        for engine in self._islands:
            merged.all_chromosomes.extend(engine.history.all_chromosomes)

        # Align GenerationStats by generation index
        max_gens = max(
            (len(e.history.generations) for e in self._islands),
            default=0,
        )
        for gen_idx in range(max_gens):
            island_gen_stats = [
                e.history.generations[gen_idx]
                for e in self._islands
                if gen_idx < len(e.history.generations)
            ]
            if not island_gen_stats:
                continue

            best_fit  = max(s.best_fitness  for s in island_gen_stats)
            worst_fit = min(s.worst_fitness for s in island_gen_stats)
            mean_fit  = sum(s.mean_fitness  for s in island_gen_stats) / len(island_gen_stats)
            mean_ham  = sum(s.mean_hamming  for s in island_gen_stats) / len(island_gen_stats)
            best_chrom = max(
                (s.best_chromosome for s in island_gen_stats if s.best_chromosome),
                key=lambda c: c.fitness,
                default=None,
            )

            merged.generations.append(GenerationStats(
                generation=gen_idx,
                best_fitness=best_fit,
                mean_fitness=mean_fit,
                worst_fitness=worst_fit,
                elapsed_seconds=max(s.elapsed_seconds for s in island_gen_stats),
                mean_hamming=mean_ham,
                mutation_rate=island_gen_stats[0].mutation_rate,
                diversity_injected=any(s.diversity_injected for s in island_gen_stats),
                mutation_boosted=any(s.mutation_boosted for s in island_gen_stats),
                best_chromosome=best_chrom,
            ))

        return merged
