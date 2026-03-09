# Genetic AutoML Framework

A production-grade genetic algorithm-driven AutoML pipeline for tabular data.  
Evolves both **preprocessing pipelines** and **model hyperparameters** simultaneously.

---

## Architecture

```
genetic_automl/
├── pipeline.py                 # AutoMLPipeline — top-level entry point
├── config.py                   # All configuration dataclasses
│
├── core/
│   ├── problem.py              # ProblemType enum, metric registry, fitness sign
│   ├── data.py                 # DataManager — 3-way stratified split
│   └── base_automl.py          # BaseAutoML abstract contract
│
├── automl/
│   ├── sklearn_model.py        # Lightweight sklearn GBM backend
│   └── autogluon_model.py      # AutoGluon backend (optional)
│
├── genetic/
│   ├── chromosome.py           # Gene space, Chromosome dataclass, random_population
│   ├── operators.py            # crossover, mutation, tournament_selection, elites
│   ├── fitness.py              # FitnessEvaluator — k-fold CV per chromosome
│   ├── engine.py               # GeneticEngine — full evolution loop
│   ├── warm_start.py           # WarmStart — default seeds + halving pre-screen
│   └── diversity.py            # PopulationDiversity — Hamming tracking + injection
│
├── preprocessing/
│   ├── pipeline.py             # PreprocessingPipeline — orchestrates all steps
│   ├── numeric_imputer.py      # mean / median / knn / iterative / constant
│   ├── outlier_handler.py      # IQR / zscore / IsolationForest
│   ├── correlation_filter.py   # Drop highly correlated features
│   ├── categorical_encoder.py  # onehot / ordinal / target / binary
│   ├── distribution_transform.py  # yeo-johnson / box-cox / log1p
│   ├── scaler.py               # standard / minmax / robust / none
│   ├── missing_indicator.py    # Binary flags for originally-missing columns
│   ├── feature_selector.py     # variance_threshold / mutual_info / RFE
│   └── imbalance_handler.py    # SMOTE / BorderlineSMOTE / ADASYN / class_weight
│
├── reporting/
│   ├── html_reporter.py        # Self-contained HTML report with charts
│   └── mlflow_logger.py        # MLflow local experiment tracking
│
└── utils/
    └── logger.py               # Structured logging
```

---

## Pipeline Execution Flow

```
Input DataFrame
      │
      ▼
DataManager.three_way_split()
      │
      ├── Train (67%) ──► GeneticEngine.run()
      │                         │
      │                    Gen-0: WarmStart (default seeds + halving pre-screen)
      │                         │
      │                    Per generation:
      │                      1. FitnessEvaluator  (k-fold CV — test never seen)
      │                      2. PopulationDiversity (Hamming distance tracking)
      │                      3. Diversity injection if population collapsed
      │                      4. Adaptive mutation boost if stagnating
      │                      5. Breed next generation
      │                         │
      │                    Best chromosome
      │                         │
      ├── Val  (17%) ──────────► Refit PreprocessingPipeline on train+val
      │                          Retrain Model on preprocessed train+val
      │
      └── Test (15%) ──► Final score (NEVER touched during GA) ──► HTML Report
```

---

## Preprocessing Step Order

```
1. NumericImputer         — FIRST: NaN breaks IQR/IsolationForest
2. OutlierHandler         — on clean data, before scaling distorts distances
3. CorrelationFilter      — after imputation, correlation stats are reliable
4. CategoricalEncoder     — encode before scaling
5. DistributionTransform  — yeo-johnson / box-cox / log1p to reduce skewness
6. Scaler                 — scale all numeric columns uniformly
7. MissingIndicator       — add __missing_{col}__ binary flags
8. FeatureSelector        — mutual_info / RFE on fully preprocessed data
9. ImbalanceHandler       — ALWAYS LAST, train only
```

---

## Quickstart

```python
from genetic_automl import AutoMLPipeline, PipelineConfig, GeneticConfig, AutoMLConfig
from genetic_automl.config import DataConfig, ReportConfig
from genetic_automl.core.problem import ProblemType

config = PipelineConfig(
    problem_type=ProblemType.CLASSIFICATION,
    target_column="label",
    genetic=GeneticConfig(
        population_size=20,
        generations=15,
        n_cv_folds=3,
        warm_start=True,
        adaptive_mutation=True,
    ),
    automl=AutoMLConfig(backend="sklearn"),
    data=DataConfig(test_size=0.15),
    report=ReportConfig(output_dir="reports/"),
)

pipeline = AutoMLPipeline(config)
pipeline.fit(train_df)

print(f"Best F1-macro: {pipeline.final_score:.4f}")
print(f"Report: {pipeline.report_path}")

predictions = pipeline.predict(new_df)
```

---

## Configuration Reference

### `GeneticConfig`

| Parameter | Default | Description |
|---|---|---|
| `population_size` | 20 | Chromosomes per generation |
| `generations` | 15 | Max generations |
| `mutation_rate` | 0.2 | Per-gene mutation probability |
| `crossover_rate` | 0.7 | Crossover vs clone probability |
| `elite_ratio` | 0.1 | Fraction of top individuals preserved |
| `tournament_size` | 3 | Tournament selection pool size |
| `early_stopping_rounds` | 5 | Stop if no improvement for N gens |
| `n_cv_folds` | 3 | CV folds per chromosome evaluation |
| `warm_start` | True | Enable default seeds + halving pre-screen |
| `warm_start_n_seeds` | 3 | Number of archetype seed configs injected |
| `warm_start_halving_pool_ratio` | 2.0 | Pool = ratio × pop_size, 1-fold screened |
| `warm_start_halving_keep_ratio` | 0.5 | Fraction of pool kept as survivors |
| `diversity_threshold` | 0.15 | Mean Hamming below which injection fires |
| `diversity_injection_ratio` | 0.2 | Fraction of worst individuals replaced |
| `adaptive_mutation` | True | Boost mutation rate on stagnation |
| `adaptive_mutation_stagnation_rounds` | 3 | Stagnation gens to trigger boost |
| `adaptive_mutation_boost_factor` | 2.5 | Multiply base rate on boost |
| `adaptive_mutation_decay` | 0.85 | Per-gen decay back to base rate |

### `DataConfig`

| Parameter | Default | Description |
|---|---|---|
| `test_size` | 0.15 | Locked test fraction (never seen by GA) |
| `val_size` | 0.2 | Val fraction of remaining dev data |
| `stratify` | True | Stratified splits (classification only) |

---

## Gene Space

Each chromosome encodes 15 genes (sklearn backend):

**Preprocessing genes (12):**
`numeric_imputer`, `outlier_method`, `outlier_threshold`, `outlier_action`,
`correlation_threshold`, `categorical_encoder`, `distribution_transform`,
`scaler`, `missing_indicator`, `feature_selection_method`, `feature_selection_k`,
`imbalance_method`

**Model genes (3, sklearn):**
`n_estimators`, `max_depth`, `learning_rate`

---

## Known Bugs (pending fixes)

| # | Severity | File | Description |
|---|---|---|---|
| B1 | 🔴 | `pipeline.py:154` | `X_test` passed to final `model.fit()` — AutoGluon leakage |
| B2 | 🔴 | `pipeline.py:106` | `val_split` discarded — final model misses 17% of available data |
| B3 | 🔴 | `core/problem.py` | `roc_auc` crashes on multiclass (needs `predict_proba`, not `predict`) |
| B4 | 🔴 | `automl/sklearn_model.py` | Internal `StandardScaler` double-preprocesses already-scaled data |
| B5 | 🟡 | `genetic/warm_start.py` | Halving uses `n_folds=1` → `StratifiedKFold` raises `ValueError`, strategy B silent no-op |
| B6 | 🟡 | `preprocessing/outlier_handler.py:137` | IsolationForest clip computes `col.median()` at transform time — leakage |
| B7 | 🟡 | `preprocessing/feature_selector.py` | `transform()` silently returns empty DataFrame on column mismatch |
| B8 | 🟡 | `preprocessing/categorical_encoder.py` | Ordinal maps unseen categories to `-1` instead of neutral index |
| B9 | 🟠 | `genetic/chromosome.py` | Step comments duplicated/out-of-order |
| B10 | 🟠 | `genetic/fitness.py` | `_PREPROCESSING_GENE_KEYS_OLD` dead code |
| B11 | 🟠 | `genetic/warm_start.py` | `import copy` unused |

---

## Roadmap

- [ ] Fix bugs B1–B8
- [ ] Parallel chromosome evaluation (`joblib.Parallel`)
- [ ] Evaluation result cache (skip re-evaluating identical gene combos)
- [ ] Switch `SklearnModel` to `HistGradientBoostingClassifier`
- [ ] Greedy ensemble of top-k chromosomes (Caruana 2004)
- [ ] Multi-fidelity evaluation (Hyperband in GA loop)
- [ ] Dataset meta-features for adaptive warm-start defaults
- [ ] Run history persistence for cross-session warm-start

---

## Install

```bash
git clone <repo-url>
cd genetic_automl
pip install -e .

# Optional backends
pip install -e ".[autogluon]"   # AutoGluon backend
pip install -e ".[imbalanced]"  # SMOTE / imbalanced sampling
pip install -e ".[reporting]"   # MLflow experiment tracking
```

## Run Tests

```bash
pytest tests/ -v
```
