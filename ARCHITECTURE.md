# GAML — Architecture Reference

For usage, see [README.md](README.md).

---

## Module layout

```
genetic_automl/
├── pipeline.py                 # AutoMLPipeline — top-level entry point
├── config.py                   # PipelineConfig, GeneticConfig, DataConfig, etc.
├── config_loader.py            # load_config() — parses gaml_config.yaml
│
├── core/
│   ├── problem.py              # ProblemType enum, metric registry
│   ├── data.py                 # DataManager — 3-way stratified split
│   └── base_automl.py          # BaseAutoML abstract interface
│
├── automl/
│   ├── sklearn_model.py        # sklearn GradientBoosting backend
│   └── autogluon_model.py      # AutoGluon backend (optional)
│
├── genetic/
│   ├── chromosome.py           # Gene space, Chromosome dataclass, random_population
│   ├── operators.py            # crossover, mutation, tournament selection, elites
│   ├── fitness.py              # FitnessEvaluator — k-fold CV per chromosome
│   ├── engine.py               # GeneticEngine — full evolution loop
│   ├── warm_start.py           # WarmStart — archetype seeding + halving pre-screen
│   └── diversity.py            # PopulationDiversity — Hamming tracking + injection
│
├── preprocessing/
│   ├── pipeline.py             # PreprocessingPipeline — orchestrates all steps
│   ├── numeric_imputer.py      # mean / median / knn / iterative / constant
│   ├── outlier_handler.py      # IQR / zscore / IsolationForest
│   ├── correlation_filter.py   # Drops highly correlated feature pairs
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

## Execution flow

```
Input DataFrame
      │
      ▼
DataManager.three_way_split()
      │
      ├── Train (67%) ──► GeneticEngine.run()
      │                         │
      │                    Gen 0: WarmStart (archetypes + halving pre-screen)
      │                         │
      │                    Per generation:
      │                      1. FitnessEvaluator  (k-fold CV on train only)
      │                      2. Compute generation stats
      │                      3. Update no-improvement streak
      │                      4. PopulationDiversity: inject if Hamming < threshold
      │                      5. Adaptive mutation boost / decay
      │                      6. Early stopping check
      │                      7. Breed next generation
      │                         │
      │                    Best chromosome
      │                         │
      ├── Val  (17%) ──────────► Refit PreprocessingPipeline on train + val
      │                          Retrain model on preprocessed train + val
      │
      └── Test (15%) ──► Final score (never touched during GA) ──► HTML report
```

---

## Zero-leakage guarantee

`FitnessEvaluator` creates a **fresh** `PreprocessingPipeline` for every (chromosome, fold) pair. All fit steps see only the fold's training portion. Val and test data only ever pass through `transform()`, never `fit()`.

---

## Gene space

Each chromosome is a flat dict of 15 genes (sklearn backend):

**Preprocessing genes (12):** `numeric_imputer`, `outlier_method`, `outlier_threshold`, `outlier_action`, `correlation_threshold`, `categorical_encoder`, `distribution_transform`, `scaler`, `missing_indicator`, `feature_selection_method`, `feature_selection_k`, `imbalance_method`

**Model genes (3, sklearn):** `n_estimators`, `max_depth`, `learning_rate`

Candidate values are defined in `genetic/chromosome.py` and overridden at runtime via `gaml_config.yaml` → `load_config()` → `AutoMLPipeline(gene_space_overrides=...)`.

---

## Key design decisions

**Why k-fold CV instead of a single val split?**
Prevents the GA from exploiting a lucky split. Fitness signal is less noisy, leading to genuinely better-performing configurations.

**Why Hamming distance for diversity?**
The gene space is categorical and discrete. Hamming distance (fraction of genes that differ) is the natural metric. Values range from 0 (identical) to 1 (every gene differs).

**Why warm-start archetypes?**
Three hand-crafted chromosomes representing common real-world patterns (clean data, messy tabular data, tree-friendly) avoid wasting early generations on obviously poor configs.

**Fitness stability penalty**
`fitness = mean_cv - penalty × std_cv` penalises chromosomes whose CV scores vary widely across folds. This favours pipelines that are consistently good rather than occasionally excellent.
