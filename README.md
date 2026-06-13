<div align=center>

<img src="images/GAML.png" width="600">

## Turn raw tabular data into production-ready ML models with zero code.

![Python Versions](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
[![GitHub license](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)  
[![Continuous Integration](https://github.com/Marc-AntoineGenest/GAML/actions/workflows/ci.yml/badge.svg)](https://github.com/Marc-AntoineGenest/GAML/actions/workflows/ci.yml)

</div>

GAML is a genetic algorithm that simultaneously searches over **preprocessing pipelines** and **model hyperparameters** for tabular data. Every candidate is scored with cross-validation; the best configuration is automatically selected and refit on your full dataset.

---

## Installation

```bash
git clone https://github.com/Marc-AntoineGenest/GAML.git
cd GAML
pip install -e .
pip install pyyaml   # required to use gaml_config.yaml
```

**Optional extras:**

| Feature | Install command |
|---|---|
| AutoGluon backend | `pip install -e ".[autogluon]"` |
| SMOTE / imbalanced sampling | `pip install -e ".[imbalanced]"` |
| MLflow experiment tracking | `pip install -e ".[reporting]"` |
| Polars fast data loading | `pip install polars pyarrow` |
| Optuna Bayesian HPO | `pip install optuna` |
| SHAP feature attribution | `pip install shap` |
| Drift detection (PSI) | `pip install scipy` |
| LightGBM backend | `pip install lightgbm` |
| XGBoost backend | `pip install xgboost` |

**Run tests:**

```bash
pytest genetic_automl/tests/ -v
```

---

## Quick start

### Option A — YAML config (recommended)

Edit `gaml_config.yaml` at the project root, then:

```python
import pandas as pd
from genetic_automl import load_config, AutoMLPipeline

df = pd.read_csv("your_data.csv")

config, gene_overrides = load_config("gaml_config.yaml")
pipeline = AutoMLPipeline(config, gene_space_overrides=gene_overrides)
pipeline.fit(df)

print(f"Test score: {pipeline.final_score:.4f}")
print(f"Report:     {pipeline.report_path}")

predictions = pipeline.predict(df)
```

### Option B — Pure Python

```python
import pandas as pd
from genetic_automl import AutoMLPipeline, PipelineConfig, GeneticConfig, AutoMLConfig
from genetic_automl.core.problem import ProblemType

config = PipelineConfig(
    problem_type=ProblemType.CLASSIFICATION,
    target_column="label",
    genetic=GeneticConfig(population_size=20, generations=15),
    automl=AutoMLConfig(backend="sklearn"),
)

pipeline = AutoMLPipeline(config)
pipeline.fit(df)
predictions = pipeline.predict(new_df)
```

---

## CLI

GAML ships a `gaml` command so you can train and predict without writing Python.

### `gaml fit` — train a pipeline

```bash
gaml fit data.csv --target label
gaml fit data.csv --target label --config gaml_config.yaml --save model.joblib
```

| Flag | Default | Description |
|---|---|---|
| `DATA_CSV` | *(required)* | Path to input CSV file |
| `--target`, `-t` | *(required if not in config)* | Name of the target column |
| `--config`, `-c` | `null` | Path to `gaml_config.yaml` |
| `--problem`, `-p` | `classification` | `classification` or `regression` |
| `--backend`, `-b` | `sklearn` | `sklearn` or `autogluon` |
| `--generations`, `-g` | `15` | Number of GA generations |
| `--population`, `-P` | `20` | GA population size |
| `--cv-folds` | `3` | Cross-validation folds |
| `--cv-strategy` | `stratified` | `stratified` · `group` · `timeseries` |
| `--group-column` | `null` | Column for group-based CV |
| `--seed` | `42` | Random seed |
| `--output-dir`, `-o` | `reports/` | HTML report output directory |
| `--save`, `-s` | `null` | Save fitted pipeline to `.joblib` |
| `--run-name` | `null` | Label shown in the HTML report |
| `--data-backend` | `pandas` | `pandas` or `polars` |
| `--no-shap` | `false` | Disable SHAP in the report |
| `--island-model` | `false` | Enable island model GA |
| `--n-islands` | `4` | Number of islands |
| `--migration-interval` | `3` | Generations between migrations |
| `--migration-size` | `2` | Chromosomes migrated per island |
| `--nsga2` | `false` | Enable NSGA-II multi-objective |
| `--objectives` | `[metric, complexity]` | Objective names for NSGA-II |
| `--calibrate` | `false` | Calibrate final model probabilities |
| `--calibration-method` | `sigmoid` | `sigmoid` or `isotonic` |

### `gaml predict` — run inference

```bash
gaml predict model.joblib new_data.csv --output predictions.csv
gaml predict model.joblib new_data.csv --detect-drift reference.csv
```

| Flag | Description |
|---|---|
| `MODEL_JOBLIB` | Path to a pipeline saved with `gaml fit --save` |
| `DATA_CSV` | CSV to predict on |
| `--output`, `-o` | Output CSV path (default: `<data>_predictions.csv`) |
| `--detect-drift REF_CSV` | Compare input against a reference CSV and print a drift report |

### `gaml update` — incremental update

```bash
gaml update model.joblib new_batch.csv --epochs 3 --save model.joblib
```

| Flag | Default | Description |
|---|---|---|
| `MODEL_JOBLIB` | *(required)* | Saved pipeline |
| `DATA_CSV` | *(required)* | New labelled batch (must include target column) |
| `--epochs` | `1` | Passes over the new batch |
| `--save` | *(overwrite input)* | Output path for updated pipeline |

### `gaml version`

```bash
gaml version
```

---

## Saving and loading

```python
# Save after fitting
pipeline.save("models/my_pipeline.joblib")

# Load later — no re-fitting needed
from genetic_automl import AutoMLPipeline
pipeline = AutoMLPipeline.load("models/my_pipeline.joblib")
predictions = pipeline.predict(df)
```

---

## Outputs

| Attribute / Method | Description |
|---|---|
| `pipeline.final_score` | Test set score (metric depends on problem type) |
| `pipeline.report_path` | Path to the generated HTML report |
| `pipeline.history` | Full `EvolutionHistory`: fitness curve, all chromosomes, diversity |
| `pipeline.best_preprocessor` | Fitted `PreprocessingPipeline` |
| `pipeline.best_model` | Fitted AutoML model (or `EnsembleModel` when ensemble is on) |
| `pipeline.predict(df)` | Predictions on new data (returns `np.ndarray`) |
| `pipeline.predict_proba(df)` | Class probabilities — classification only |
| `pipeline.feature_importances_` | Named `pd.Series` of feature importances, sorted descending |
| `pipeline.summary()` | `dict` with run metadata, best genes, final score, and timing |
| `pipeline.detect_drift(new_df)` | Drift report comparing `new_df` to training distribution |
| `pipeline.partial_fit(new_df, epochs=1)` | Incrementally update the model on a new labelled batch |

The HTML report includes a generation-by-generation fitness curve, diversity tracking, the best chromosome's gene values, SHAP feature importance (if enabled), and the final test score.

---

## Configuration (`gaml_config.yaml`)

The YAML file is the single place to control everything without touching Python code.

### A. Run settings

```yaml
run:
  problem_type: classification   # classification | regression
  target_column: target          # column name in your DataFrame
  backend: sklearn               # sklearn | autogluon
  metric: null                   # null = default (f1_macro / mse)
  name: null                     # human-readable run label for reports
```

### B. Data settings

```yaml
data:
  backend: pandas      # pandas (default) | polars (2-10x faster for large files)
  test_size: 0.15      # fraction locked as the final test set
  val_size: 0.20       # validation fraction used during the final refit
  stratify: true       # stratified splits (classification only)
  random_seed: 42
```

> **Polars backend** — set `backend: polars` to load CSVs 2–10× faster on large datasets (> 100k rows). Requires `pip install polars pyarrow`. Falls back to pandas automatically if not installed.

### C. Model backends

| `model_type` | Install | Best for |
|---|---|---|
| `gbm` | *(included)* | Default — reliable on most datasets |
| `lgbm` | `pip install lightgbm` | Large datasets, fastest training |
| `xgb` | `pip install xgboost` | Kaggle-style tasks, competitive accuracy |
| `rf` | *(included)* | Low-variance baseline, good for noisy data |
| `autogluon` | `pip install autogluon.tabular` | Maximum accuracy, slow, set `backend: autogluon` |

```yaml
sklearn_search_space:
  model_type: ["gbm", "lgbm", "xgb", "rf"]   # GA picks from this list
  n_estimators: [50, 100, 200, 300, 500]
  max_depth: [2, 3, 4, 5, 6, 8]
  learning_rate: [0.01, 0.05, 0.1, 0.2]
```

Pin a single model: `model_type: ["lgbm"]`

### D. Genetic algorithm settings

| Parameter | Default | Description |
|---|---|---|
| `population_size` | `20` | Pipeline configs evaluated per generation. Larger = broader search, slower. |
| `generations` | `15` | Maximum evolution cycles. |
| `n_cv_folds` | `3` | CV folds per evaluation. Higher = more reliable, slower. |
| `early_stopping_rounds` | `5` | Stop if no improvement for N consecutive generations. |
| `mutation_rate` | `0.20` | Probability a gene changes value during reproduction. |
| `crossover_rate` | `0.70` | Probability two parents recombine vs. direct clone. |
| `crossover_type` | `uniform` | `uniform` (each gene from either parent) or `single_point`. |
| `elite_ratio` | `0.10` | Fraction of top individuals kept unchanged each generation. |
| `tournament_size` | `3` | Candidates competing in each tournament selection draw. |
| `warm_start` | `true` | Seed generation 0 with known-good archetype configs. |
| `warm_start_n_seeds` | `3` | Number of hand-crafted archetypes injected (max 3). |
| `warm_start_halving_pool_ratio` | `2.0` | Pool size = ratio × population; screened with 1-fold CV. |
| `warm_start_halving_keep_ratio` | `0.50` | Fraction of screened pool kept as generation-0 survivors. |
| `diversity_threshold` | `0.15` | Mean Hamming distance below which fresh individuals are injected. |
| `diversity_injection_ratio` | `0.20` | Fraction of worst individuals replaced on low-diversity trigger. |
| `adaptive_mutation` | `true` | Boost mutation rate on stagnation, decay on improvement. |
| `adaptive_mutation_stagnation_rounds` | `3` | Stagnant generations before boost fires. |
| `adaptive_mutation_boost_factor` | `2.5` | Mutation rate multiplier when boosted. |
| `adaptive_mutation_decay` | `0.85` | Per-generation decay back toward the base rate. |
| `fitness_std_penalty` | `0.5` | `fitness = mean_cv − penalty × std_cv`. Higher = prefer stable pipelines. |
| `n_jobs` | `1` | Parallel workers. `-1` = all cores (sklearn only). |
| `random_seed` | `42` | Reproducibility seed. |

### E. Advanced GA features

#### NSGA-II multi-objective

The GA maintains a Pareto front instead of a single fitness score. Solutions are ranked by non-domination and crowding distance. The final model is still selected by the primary metric (first objective).

Built-in special objectives (no extra CV cost): `complexity` (penalises large models), `latency` (penalises slow models).

```yaml
genetic:
  nsga2_enabled: false
  nsga2_objectives:        # null = [primary_metric, complexity]
    - f1_macro
    - complexity
```

#### Island model GA

The population is split into isolated sub-populations that evolve independently and periodically exchange top chromosomes (ring topology). Maintains diversity and typically finds solutions 5–15% better than a single population.

```yaml
genetic:
  island_model: false
  n_islands: 4             # sub-populations (2–8)
  migration_interval: 3    # generations between migration events
  migration_size: 2        # chromosomes sent per island per migration
  n_island_jobs: 1         # 1 = sequential, -1 = all threads
```

> Only supported for `backend: sklearn`.

#### Surrogate-assisted GA

A small model is trained on accumulated fitness history and used to predict whether a new chromosome is worth a full CV evaluation. Skips ~30–50% of expensive CV calls with no accuracy loss.

```yaml
genetic:
  surrogate_enabled: true
  surrogate_model_type: "rf"       # rf | lgbm | xgb | gbm
  surrogate_min_samples: 10        # fully-evaluated chromosomes before surrogate activates
  surrogate_uncertainty_threshold: 0.05   # skip only when prediction confidence is high
```

#### ASHA pruning

Cuts the fold loop short for clearly bad chromosomes. After `asha_min_folds_before_prune` folds, if a chromosome's running mean score is below the field median, remaining folds are skipped. Saves 20–30% of total CV time at no accuracy cost.

```yaml
genetic:
  asha_enabled: true
  asha_min_folds_before_prune: 1   # folds needed before pruning can fire
  asha_prune_margin: 0.0           # extra clearance below median to prune
```

#### CV strategy

```yaml
genetic:
  cv_strategy: stratified   # stratified | group | timeseries
  group_column: null        # column for group-based CV (cv_strategy: group)
```

| Strategy | Use when |
|---|---|
| `stratified` | Default. Keeps class ratios balanced across folds. |
| `group` | Rows share group IDs (patient, store, user) — prevents leakage across groups. |
| `timeseries` | Rows have a time axis — preserves temporal ordering. |

#### Generation checkpointing

Save state to disk so runs can be resumed after a crash or time limit.

```yaml
genetic:
  checkpoint_dir: "checkpoints"         # directory to save checkpoint files
  checkpoint_every: 1                   # save every N generations
  resume_from_checkpoint: null          # path to a .joblib checkpoint to resume from
```

### F. Post-GA refinements

#### Ensemble

After evolution, the top-k unique chromosomes are refitted and combined into a soft-voting (classification) or averaging (regression) ensemble. Consistently adds +3–8% on held-out test score.

```yaml
ensemble:
  enabled: true
  top_k: 3                   # chromosomes combined
  weight_by_fitness: true    # weight members proportionally to CV fitness
```

#### Optuna Bayesian HPO

After the GA finds the best structural pipeline (model family, preprocessors), Optuna fine-tunes continuous hyperparameters (learning rate, depth, regularisation) using TPE — a Bayesian method far more sample-efficient than the GA for this task. Typical result: +1–5% on top of the GA's best chromosome.

```yaml
optuna:
  enabled: false
  n_trials: 30       # 20–30 good default; 50–100 for production
  timeout: null      # hard wall-clock cap in seconds
  use_cv: false      # false = fast 80/20 split per trial; true = full k-fold CV
  n_cv_folds: 3
  verbose: false
```

> Requires `pip install optuna`. Only supported for `backend: sklearn`.

#### Probability calibration

Tree models often produce poorly calibrated probabilities. CalibratedClassifierCV corrects this using cross-validated held-out predictions. Only applied for classification.

```yaml
calibration:
  enabled: false
  method: sigmoid    # sigmoid (Platt scaling) | isotonic (needs >= 1000 samples)
  cv: 5             # internal CV folds for the calibrator
```

### G. Preprocessing search space

Steps are always applied in this fixed order. The GA evolves which option to use at each step, not the order.

Use a single-element list to pin any step: `scaler: [standard]`

| Step | Gene | Options |
|---|---|---|
| 1. Numeric imputation | `numeric_imputer` | `mean` · `median` · `knn` · `iterative` · `constant` |
| 2. Outlier handling | `outlier_method` | `none` · `iqr` · `zscore` · `isolation_forest` |
| | `outlier_threshold` | `1.5` · `2.0` · `3.0` |
| | `outlier_action` | `clip` · `flag` |
| 3. Feature engineering | `feature_engineering` | `none` · `log1p` · `ratio` · `poly2` · `all` |
| | `max_interaction_features` | `4` · `6` · `8` |
| 4. Correlation filter | `correlation_threshold` | `null` · `0.85` · `0.90` · `0.95` |
| 5. Categorical encoding | `categorical_encoder` | `onehot` · `ordinal` · `target` · `binary` |
| 6. Distribution transform | `distribution_transform` | `none` · `yeo-johnson` · `box-cox` · `log1p` |
| 7. Scaler | `scaler` | `none` · `standard` · `minmax` · `robust` |
| 8. Missing indicator | `missing_indicator` | `true` · `false` |
| 9. Feature selection | `feature_selection_method` | `none` · `variance_threshold` · `mutual_info` · `rfe` |
| | `feature_selection_k` | `0.50` · `0.75` · `1.0` |
| 10. Imbalance handling | `imbalance_method` | `none` · `smote` · `borderline_smote` · `adasyn` · `class_weight` |

> **Feature engineering** — generates new features from existing numeric columns (applied after imputation, before scaling). `poly2` adds squared terms and cross-products; `ratio` adds pairwise ratios of correlated columns; `all` applies log1p + ratio + poly2 in sequence. Engineered features that don't improve CV score are removed by FeatureSelector automatically.

### H. Reporting and monitoring

```yaml
report:
  output_dir: reports
  mlflow_tracking_uri: mlflow_runs   # null to disable MLflow
  open_html_on_finish: false

  # Drift detection
  drift_enabled: false
  drift_pvalue_threshold: 0.05       # KS / chi-squared p-value threshold
  drift_psi_threshold: 0.20          # PSI threshold

  # SHAP feature attribution
  shap_enabled: true
  shap_max_samples: 200              # rows sampled from dev set
```

#### Drift detection

When `drift_enabled: true`, GAML fits a `DriftDetector` on the training distribution. After deployment, call `pipeline.detect_drift(new_df)` (or `gaml predict --detect-drift ref.csv`) to compare any new batch against the reference using KS test (continuous), chi-squared (categorical), and PSI.

```python
report = pipeline.detect_drift(new_df)
# report is a dict: {"drifted": bool, "features": {col: {stat, p_value, psi}}}
```

> Requires `pip install scipy` for PSI. Falls back to KS/chi-squared only without it.

#### SHAP attribution

When `shap_enabled: true` and `shap` is installed, GAML generates a SHAP feature importance summary and embeds it in the HTML report. Only active for sklearn tree models (`lgbm`, `xgb`, `gbm`, `rf`).

```bash
pip install shap
```

#### Incremental learning

Update a fitted pipeline on new labelled batches without full re-training. Requires the model to support incremental learning (set `model_type` to a backend that wraps a sklearn estimator with `partial_fit`, e.g. an SGD-based model, or use the `IncrementalModel` wrapper).

```python
pipeline.partial_fit(new_df, epochs=3)
pipeline.save("model_updated.joblib")
```

---

## Preprocessing step order

GAML always applies preprocessing in this fixed order. The GA evolves which option to use at each step, not the order itself.

```
1.   NumericImputer         — fill NaN before anything else
2.   OutlierHandler         — on clean numeric data, before scaling
3.   FeatureEngineer        — generate poly/ratio/log1p features
4.   CorrelationFilter      — reliable stats after imputation
5.   CategoricalEncoder     — encode before scaling
6.   DistributionTransform  — reduce skewness before scaling
7.   Scaler                 — after all columns are numeric
8.   MissingIndicator       — binary flags for originally-missing columns
9.   FeatureSelector        — on fully preprocessed data
10.  ImbalanceHandler       — always last, training data only
```

---

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for module layout, execution flow, and the zero-leakage guarantee.
