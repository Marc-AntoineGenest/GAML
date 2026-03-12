# GAML — Architecture Reference

Developer documentation. For usage, see [README.md](README.md).

---

## Module layout

```
genetic_automl/
├── pipeline.py                 # AutoMLPipeline — top-level entry point
├── config.py                   # PipelineConfig, GeneticConfig, DataConfig, etc.
├── config_loader.py            # load_config() — parses gaml_config.yaml
│
├── core/
│   ├── problem.py              # ProblemType enum, metric registry, pareto_front
│   ├── data.py                 # DataManager — 3-way stratified split
│   └── base_automl.py          # BaseAutoML abstract contract
│
├── automl/
│   ├── sklearn_model.py        # Lightweight sklearn GradientBoosting backend
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

## Execution flow

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
      │                      3. Update no_improvement_streak
      │                      4. Diversity injection if population collapsed
      │                      5. Adaptive mutation boost if stagnating
      │                      6. Breed next generation
      │                         │
      │                    Best chromosome
      │                         │
      ├── Val  (17%) ──────────► Refit PreprocessingPipeline on train+val
      │                          Retrain Model on preprocessed train+val
      │
      └── Test (15%) ──► Final score (NEVER touched during GA) ──► HTML Report
```

---

## Zero-leakage guarantee

`FitnessEvaluator` creates a **fresh** `PreprocessingPipeline` for every (chromosome, fold) pair. All preprocessing steps fit only on the fold's training portion. Val/test data is only passed to `transform()`, never to `fit()`.

---

## Gene space

Each chromosome is a flat dict of 15 genes (sklearn backend):

**Preprocessing genes (12):** `numeric_imputer`, `outlier_method`, `outlier_threshold`, `outlier_action`, `correlation_threshold`, `categorical_encoder`, `distribution_transform`, `scaler`, `missing_indicator`, `feature_selection_method`, `feature_selection_k`, `imbalance_method`

**Model genes (3, sklearn):** `n_estimators`, `max_depth`, `learning_rate`

The candidate values for each gene are defined in `genetic/chromosome.py` and can be overridden at runtime via `gaml_config.yaml` → `load_config()` → `AutoMLPipeline(gene_space_overrides=...)`.

---

## Roadmap

| Priority | Improvement | Impact | Effort |
|---|---|---|---|
| 1 | Parallel fitness eval (`joblib.Parallel`) | Very High | Low |
| 2 | Successive Halving fitness schedule (n_folds=1→2→3 across gens) | Very High | Low |
| 3 | Fitness evaluation LRU cache | High | Low |
| 4 | `fitness_std` penalty: `fitness = mean_cv - α * std_cv` | Medium | Trivial |
| 5 | Switch default crossover to `uniform_crossover` | Medium | Trivial |
| 6 | Surrogate fitness model (RF on genes→fitness after gen 3) | High | Medium |
| 7 | Optuna/BOHB for continuous model hyperparams | High | Medium |
| 8 | Island model GA (structural diversity) | Medium | High |
