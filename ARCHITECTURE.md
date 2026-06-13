# GAML — Architecture Reference

For usage, see [README.md](README.md).

---

## Module layout

```
genetic_automl/
├── pipeline.py                 # AutoMLPipeline — top-level entry point
├── cli.py                      # gaml fit / predict / update / version CLI
├── config.py                   # PipelineConfig, GeneticConfig, DataConfig, etc.
├── config_loader.py            # load_config() — parses gaml_config.yaml
│
├── core/
│   ├── problem.py              # ProblemType enum, metric registry
│   ├── data.py                 # DataManager — 3-way split (pandas + polars backends)
│   └── base_automl.py          # BaseAutoML abstract interface
│
├── automl/
│   ├── sklearn_model.py        # sklearn GradientBoosting backend (gbm)
│   ├── lgbm_model.py           # LightGBM backend (lgbm)
│   ├── xgb_model.py            # XGBoost backend (xgb)
│   ├── rf_model.py             # RandomForest backend (rf)
│   ├── ensemble_model.py       # EnsembleModel — soft-voting / averaging over top-k
│   ├── incremental_model.py    # IncrementalModel — partial_fit() wrapper
│   └── autogluon_model.py      # AutoGluon backend (optional)
│
├── genetic/
│   ├── chromosome.py           # Gene space, Chromosome dataclass, random_population
│   ├── operators.py            # crossover, mutation, tournament selection, elites
│   ├── fitness.py              # FitnessEvaluator — k-fold CV per chromosome
│   ├── engine.py               # GeneticEngine — full evolution loop + checkpointing
│   ├── island_engine.py        # IslandEngine — multi-population GA with migration
│   ├── nsga2.py                # NSGA-II — Pareto ranking + crowding distance
│   ├── surrogate.py            # SurrogateModel — skip CV for predicted low-scorers
│   ├── optuna_tuner.py         # OptunaTuner — Bayesian HPO after GA finishes
│   ├── warm_start.py           # WarmStart — archetype seeding + halving pre-screen
│   └── diversity.py            # PopulationDiversity — Hamming tracking + injection
│
├── preprocessing/
│   ├── pipeline.py             # PreprocessingPipeline — orchestrates all 10 steps
│   ├── numeric_imputer.py      # mean / median / knn / iterative / constant
│   ├── outlier_handler.py      # IQR / zscore / IsolationForest
│   ├── feature_engineer.py     # log1p / ratio / poly2 feature generation
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
│   ├── mlflow_logger.py        # MLflow local experiment tracking
│   ├── shap_explainer.py       # SHAP feature attribution summary
│   └── drift_detector.py       # KS / chi-squared / PSI drift detection
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
      ├── Train (67%) ──► GeneticEngine.run()  [or IslandEngine.run()]
      │                         │
      │                    Gen 0: WarmStart (archetypes + halving pre-screen)
      │                         │
      │                    Per generation:
      │                      1. SurrogateModel: skip low-predicted chromosomes (optional)
      │                      2. FitnessEvaluator: k-fold CV on train only
      │                           └── ASHA pruning: cut bad chromosomes mid-fold (optional)
      │                      3. Compute generation stats
      │                      4. Update no-improvement streak
      │                      5. PopulationDiversity: inject if Hamming < threshold
      │                      6. Adaptive mutation boost / decay
      │                      7. Early stopping check
      │                      8. NSGA-II Pareto selection (optional, replaces step 7 ranking)
      │                      9. Checkpoint to disk (optional)
      │                     10. Breed next generation
      │                         │
      │                    Best chromosome(s)
      │                         │
      │                    OptunaTuner: fine-tune hyperparameters (optional)
      │                         │
      ├── Val  (17%) ──────────► Refit PreprocessingPipeline on train + val
      │                          Retrain model(s) on preprocessed train + val
      │                          EnsembleModel: combine top-k (optional)
      │                          CalibrationModel: calibrate probabilities (optional)
      │
      └── Test (15%) ──► Final score (never touched during GA) ──► HTML report
```

---

## Zero-leakage guarantee

`FitnessEvaluator` creates a **fresh** `PreprocessingPipeline` for every (chromosome, fold) pair. All fit steps see only the fold's training portion. Val and test data only ever pass through `transform()`, never `fit()`.

---

## Gene space

Each chromosome is a flat dict of **18 genes** (sklearn backend):

**Preprocessing genes (14):** `numeric_imputer`, `outlier_method`, `outlier_threshold`, `outlier_action`, `feature_engineering`, `max_interaction_features`, `correlation_threshold`, `categorical_encoder`, `distribution_transform`, `scaler`, `missing_indicator`, `feature_selection_method`, `feature_selection_k`, `imbalance_method`

**Model genes (4, sklearn):** `model_type`, `n_estimators`, `max_depth`, `learning_rate`

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

**Why surrogate-assisted evaluation?**
Full k-fold CV is the bottleneck. A RandomForest trained on past (genes → fitness) pairs can predict fitness cheaply. Chromosomes predicted below the current median are skipped, saving 30–50% of CV calls with negligible accuracy loss.

**Why island model instead of larger population?**
A single large population converges quickly to a local optimum. Independent islands maintain diverse search trajectories; periodic migration shares discoveries without homogenising the populations prematurely.

**Why Optuna after the GA?**
The GA is efficient at categorical/structural search (which model family, which preprocessors) but wasteful at continuous tuning (learning rate, regularisation). Optuna's TPE Bayesian optimiser is far more sample-efficient for the latter once the structure is fixed.
