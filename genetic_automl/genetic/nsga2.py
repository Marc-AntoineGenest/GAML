"""
NSGA-II — Non-dominated Sorting Genetic Algorithm II.

Background
----------
Standard single-objective GA collapses a multi-metric trade-off into a single
scalar (e.g. a weighted sum).  This discards information and forces the user to
commit to fixed weights before they have seen any solutions.

NSGA-II (Deb et al. 2002) instead maintains a *Pareto front*: the set of
solutions where improving one objective requires sacrificing another.  Rather
than picking winners by a single score, it ranks solutions by:

  1. **Non-domination rank** — front 0 is the Pareto front (no solution is
     strictly better on all objectives).  Front 1 is the Pareto front of the
     remainder, and so on.
  2. **Crowding distance** — within the same front, prefer solutions that are
     farther from their neighbours in objective space.  This keeps the front
     diverse and spread out rather than clustering at one extreme.

The combination produces a population that simultaneously approximates the
entire trade-off curve instead of a single point on it.

Practical use in GAML
---------------------
Typical multi-objective run in GAML:

  objectives: [f1_macro, roc_auc]            # accuracy vs. discrimination
  objectives: [f1_macro, -n_estimators]      # accuracy vs. model complexity
  objectives: [roc_auc, -fit_latency]        # discrimination vs. speed

The pipeline always selects the *best scalar metric* chromosome (highest
primary objective score) for the final model, so the rest of the system
(ensemble, calibration, SHAP) is unchanged.  The Pareto front is persisted
in EvolutionHistory and rendered as a scatter plot in the HTML report.

Algorithm (this file)
---------------------
fast_non_dominated_sort(population)
    O(M·N²) where M = n_objectives, N = pop_size.
    Returns a list of fronts, each front a list of Chromosomes.

crowding_distance(front, objective_values)
    O(M·|front|·log|front|).
    Assigns a crowding_distance attribute to each chromosome in the front.

nsga2_select(population, n, rng)
    Binary tournament using (rank, -crowding_distance) as the key.
    Replaces tournament_selection when nsga2_enabled=True.

nsga2_survive(population, n_survive)
    Keep the top-n_survive individuals using rank then crowding distance.
    Called at the end of each generation to trim the combined parent+child
    population back to population_size.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

from genetic_automl.genetic.chromosome import Chromosome
from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)

# Attribute names stored on Chromosome objects during NSGA-II selection.
# Using strings rather than a separate dataclass avoids touching Chromosome.
_RANK_ATTR     = "_nsga2_rank"
_CROWD_ATTR    = "_nsga2_crowding"
_INF_CROWD     = float("inf")


# ---------------------------------------------------------------------------
# Core NSGA-II operators
# ---------------------------------------------------------------------------

def dominates(a_obj: List[float], b_obj: List[float]) -> bool:
    """
    Return True if solution *a* dominates solution *b*.

    *a* dominates *b* iff:
      - *a* is no worse than *b* on every objective, AND
      - *a* is strictly better than *b* on at least one objective.

    All objectives are assumed to be **maximised**.  For objectives where
    lower is better (e.g. MSE, latency), negate the value before calling.
    """
    at_least_one_better = False
    for ai, bi in zip(a_obj, b_obj):
        if ai < bi:          # a is worse on this objective → cannot dominate
            return False
        if ai > bi:
            at_least_one_better = True
    return at_least_one_better


def fast_non_dominated_sort(
    population: List[Chromosome],
    objective_values: Dict[str, List[float]],
) -> List[List[Chromosome]]:
    """
    Partition *population* into Pareto fronts F0, F1, F2, …

    Parameters
    ----------
    population : list[Chromosome]
        All individuals (evaluated).
    objective_values : dict[chrom_id -> list[float]]
        Objective vector for each chromosome, already sign-corrected so that
        **higher is always better** for every entry.

    Returns
    -------
    list[list[Chromosome]]
        fronts[0] = Pareto-optimal set; fronts[1] = next best; etc.
    """
    n = len(population)
    if n == 0:
        return []

    # domination counts and dominated-by sets
    dom_count  = [0]  * n          # how many solutions dominate i
    dom_by     = [[] for _ in range(n)]   # indices dominated by i

    objs = [objective_values.get(c.id, []) for c in population]

    for i in range(n):
        for j in range(i + 1, n):
            if not objs[i] or not objs[j]:
                continue
            if dominates(objs[i], objs[j]):
                dom_by[i].append(j)
                dom_count[j] += 1
            elif dominates(objs[j], objs[i]):
                dom_by[j].append(i)
                dom_count[i] += 1

    fronts: List[List[int]] = []
    current_front = [i for i in range(n) if dom_count[i] == 0]
    while current_front:
        fronts.append(current_front)
        next_front = []
        for i in current_front:
            for j in dom_by[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    next_front.append(j)
        current_front = next_front

    # Convert index lists to Chromosome lists and stamp ranks
    result = []
    for rank, front_indices in enumerate(fronts):
        front_chroms = []
        for idx in front_indices:
            c = population[idx]
            setattr(c, _RANK_ATTR, rank)
            front_chroms.append(c)
        result.append(front_chroms)

    return result


def crowding_distance_assignment(
    front: List[Chromosome],
    objective_values: Dict[str, List[float]],
    n_objectives: int,
) -> None:
    """
    Compute crowding distance for each chromosome in *front* (in-place).

    Boundary individuals (best / worst on any objective) receive distance ∞.
    Interior individuals receive the sum of normalised gaps to their neighbours
    across all objectives.
    """
    n = len(front)
    if n == 0:
        return
    if n <= 2:
        for c in front:
            setattr(c, _CROWD_ATTR, _INF_CROWD)
        return

    # Initialise
    for c in front:
        setattr(c, _CROWD_ATTR, 0.0)

    for obj_idx in range(n_objectives):
        # Sort by this objective
        sorted_front = sorted(
            front,
            key=lambda c: (objective_values.get(c.id) or [0.0] * n_objectives)[obj_idx],
        )
        obj_min = (objective_values.get(sorted_front[0].id)  or [0.0]*n_objectives)[obj_idx]
        obj_max = (objective_values.get(sorted_front[-1].id) or [0.0]*n_objectives)[obj_idx]
        obj_range = obj_max - obj_min

        # Boundaries get infinite distance
        setattr(sorted_front[0],  _CROWD_ATTR, _INF_CROWD)
        setattr(sorted_front[-1], _CROWD_ATTR, _INF_CROWD)

        if obj_range == 0:
            continue

        for k in range(1, n - 1):
            prev_val = (objective_values.get(sorted_front[k-1].id) or [0.0]*n_objectives)[obj_idx]
            next_val = (objective_values.get(sorted_front[k+1].id) or [0.0]*n_objectives)[obj_idx]
            current_crowd = getattr(sorted_front[k], _CROWD_ATTR)
            if current_crowd != _INF_CROWD:
                setattr(
                    sorted_front[k], _CROWD_ATTR,
                    current_crowd + (next_val - prev_val) / obj_range,
                )


def nsga2_select(
    population: List[Chromosome],
    rng: random.Random,
) -> Chromosome:
    """
    Binary tournament selection using NSGA-II dominance ranking.

    Prefers the individual with:
      1. Lower rank (closer to Pareto front), then
      2. Higher crowding distance (more isolated = more diverse).

    Falls back to random choice if neither candidate has rank/distance set.
    """
    a, b = rng.sample(population, min(2, len(population)))
    if len(population) == 1:
        return population[0].copy()

    rank_a = getattr(a, _RANK_ATTR, 0)
    rank_b = getattr(b, _RANK_ATTR, 0)
    crowd_a = getattr(a, _CROWD_ATTR, 0.0)
    crowd_b = getattr(b, _CROWD_ATTR, 0.0)

    if rank_a < rank_b:
        return a.copy()
    if rank_b < rank_a:
        return b.copy()
    # Same rank — prefer the less crowded (more diverse) individual
    return (a if crowd_a >= crowd_b else b).copy()


def nsga2_survive(
    combined: List[Chromosome],
    n_survive: int,
    objective_values: Dict[str, List[float]],
    n_objectives: int,
) -> List[Chromosome]:
    """
    Select *n_survive* individuals from *combined* (parents + offspring) using
    NSGA-II environmental selection.

    Fill slots front-by-front.  If a front partially fits, take the most
    crowded-distance-diverse individuals from it.
    """
    evaluated = [c for c in combined if c.fitness is not None]
    if not evaluated:
        return combined[:n_survive]

    fronts = fast_non_dominated_sort(evaluated, objective_values)

    survivors: List[Chromosome] = []
    for front in fronts:
        if len(survivors) + len(front) <= n_survive:
            crowding_distance_assignment(front, objective_values, n_objectives)
            survivors.extend(front)
        else:
            # Partially add from this front — pick by descending crowding distance
            crowding_distance_assignment(front, objective_values, n_objectives)
            remaining = n_survive - len(survivors)
            sorted_front = sorted(
                front,
                key=lambda c: getattr(c, _CROWD_ATTR, 0.0),
                reverse=True,
            )
            survivors.extend(sorted_front[:remaining])
            break

    # Pad with unevaluated if needed (shouldn't happen in normal flow)
    unevaluated = [c for c in combined if c.fitness is None]
    while len(survivors) < n_survive and unevaluated:
        survivors.append(unevaluated.pop(0))

    return survivors


# ---------------------------------------------------------------------------
# Objective extraction helpers
# ---------------------------------------------------------------------------

def build_objective_values(
    population: List[Chromosome],
    objectives: List[str],
    latency_map: Optional[Dict[str, float]] = None,
) -> Dict[str, List[float]]:
    """
    Build the objective value matrix for NSGA-II.

    Parameters
    ----------
    population : list[Chromosome]
    objectives : list[str]
        Each entry is either:
          - A metric name in the fitness registry (e.g. "f1_macro", "roc_auc").
            GAML stores the primary metric fitness on Chromosome.fitness.
            Secondary metrics are stored in Chromosome.extra_scores (dict).
          - "complexity" — negated n_estimators (lower = simpler = better).
          - "latency"    — negated fit duration from latency_map.
    latency_map : dict[chrom_id -> float] | None
        Pre-measured fit durations.  Required when "latency" is in objectives.

    Returns
    -------
    dict[chrom_id -> list[float]]
        Already sign-corrected: higher is always better.
    """
    from genetic_automl.core.problem import fitness_sign

    result: Dict[str, List[float]] = {}

    for chrom in population:
        if chrom.fitness is None:
            continue
        obj_vec = []
        for obj in objectives:
            if obj == "complexity":
                # Negate n_estimators: fewer trees = simpler = better
                n_est = chrom.genes.get("n_estimators", 100)
                val = -float(n_est) if n_est is not None else 0.0
            elif obj == "latency":
                dur = (latency_map or {}).get(chrom.id, 0.0)
                val = -float(dur)   # lower latency is better → negate
            else:
                # Primary metric: use chrom.fitness for the first objective,
                # extra_scores for subsequent ones.
                extra = getattr(chrom, "extra_scores", {}) or {}
                if obj in extra:
                    raw = extra[obj]
                    sign = fitness_sign(obj) if _metric_registered(obj) else 1
                    val = raw * sign
                else:
                    # Fall back to primary fitness for any unrecognised name
                    val = chrom.fitness if chrom.fitness is not None else float("-inf")
            obj_vec.append(val)
        result[chrom.id] = obj_vec

    return result


def _metric_registered(metric: str) -> bool:
    """Return True if the metric is in GAML's registry."""
    try:
        from genetic_automl.core.problem import _METRIC_REGISTRY
        return metric in _METRIC_REGISTRY
    except Exception:
        return False


def pareto_front_summary(
    history_chromosomes: List[Chromosome],
    objectives: List[str],
) -> List[dict]:
    """
    Extract the final Pareto front from the full history and return a
    JSON-serialisable list of dicts for the HTML report.

    Each dict contains:
        chromosome_id, rank, objectives (dict name→value), genes (key subset)
    """
    evaluated = [c for c in history_chromosomes if c.fitness is not None]
    if not evaluated:
        return []

    obj_vals = build_objective_values(evaluated, objectives)
    fronts   = fast_non_dominated_sort(evaluated, obj_vals)

    if not fronts:
        return []

    pareto = fronts[0]  # rank-0 = Pareto-optimal
    crowding_distance_assignment(pareto, obj_vals, len(objectives))

    result = []
    for c in sorted(pareto, key=lambda x: getattr(x, _CROWD_ATTR, 0), reverse=True):
        result.append({
            "id":          c.id,
            "rank":        0,
            "crowding":    round(getattr(c, _CROWD_ATTR, 0.0), 4),
            "objectives":  dict(zip(objectives, obj_vals.get(c.id, []))),
            "fitness":     round(c.fitness, 6),
            "key_genes": {
                k: v for k, v in c.genes.items()
                if k in ("model_type", "n_estimators", "learning_rate",
                         "max_depth", "scaler")
            },
        })
    return result
