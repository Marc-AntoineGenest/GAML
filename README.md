# GAML — Genetic AutoML

A genetic algorithm that simultaneously searches over **preprocessing pipelines** and **model hyperparameters** for tabular data. Every candidate is scored with cross-validation; the best configuration is automatically selected and refit on your full dataset.

---

## Installation

```bash
git clone https://github.com/Marc-AntoineGenest/GAML.git
cd GAML
pip install -e .
pip install pyyaml   # required to use gaml_config.yaml
```

**Optional extras:**

```bash
pip install -e ".[autogluon]"   # AutoGluon backend (more powerful, slower)
pip install -e ".[imbalanced]"  # SMOTE / imbalanced-learn sampling methods
pip install -e ".[reporting]"   # MLflow experiment tracking
```

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

## Configuration (`gaml_config.yaml`)

The YAML file is the single place to control everything without touching Python code.

### Run settings

```yaml
run:
  problem_type: classification   # classification | regression
  target_column: target          # column name in your DataFrame
  backend: sklearn               # sklearn | autogluon
  metric: null                   # null = default (f1_macro for clf, mse for reg)
```

### Data settings

```yaml
data:
  test_size: 0.15    # fraction locked as the final test set (never seen by the GA)
  val_size: 0.20     # validation fraction used during the final refit
  stratify: true     # stratified splits (classification only)
```

### Genetic algorithm settings

| Parameter | Default | Description |
|---|---|---|
| `population_size` | 20 | Pipeline configs evaluated per generation. Larger = broader search, slower. |
| `generations` | 15 | Maximum evolution cycles. |
| `n_cv_folds` | 3 | CV folds per evaluation. Higher = more reliable, slower. |
| `early_stopping_rounds` | 5 | Stop if no improvement for N generations. |
| `mutation_rate` | 0.20 | Probability a gene changes value each reproduction. |
| `crossover_rate` | 0.70 | Probability two parents recombine vs. clone. |
| `crossover_type` | uniform | `uniform` or `single_point`. |
| `elite_ratio` | 0.10 | Fraction of top individuals kept unchanged. |
| `warm_start` | true | Seed generation 0 with known-good archetype configs. |
| `adaptive_mutation` | true | Boost mutation rate on stagnation, decay on improvement. |
| `diversity_threshold` | 0.15 | Mean Hamming distance below which fresh individuals are injected. |
| `fitness_std_penalty` | 0.5 | Penalises unstable pipelines: `fitness = mean_cv - penalty × std_cv`. |
| `n_jobs` | 1 | Parallel workers. Use `-1` for all cores (sklearn backend only). |

### Search space

This controls which parameter values the GA is allowed to explore. Use a single-element list to fix a parameter.

```yaml
preprocessing_search_space:
  scaler: [standard]              # fixed — always use standard scaling
  numeric_imputer: [mean, median] # search between mean and median only
  outlier_method: [none]          # disabled
```

Full preprocessing gene options:

| Gene | Options |
|---|---|
| `numeric_imputer` | `mean` `median` `knn` `iterative` `constant` |
| `outlier_method` | `none` `iqr` `zscore` `isolation_forest` |
| `outlier_threshold` | `1.5` `2.0` `3.0` |
| `outlier_action` | `clip` `flag` |
| `correlation_threshold` | `null` `0.85` `0.90` `0.95` |
| `categorical_encoder` | `onehot` `ordinal` `target` `binary` |
| `distribution_transform` | `none` `yeo-johnson` `box-cox` `log1p` |
| `scaler` | `none` `standard` `minmax` `robust` |
| `missing_indicator` | `true` `false` |
| `feature_selection_method` | `none` `variance_threshold` `mutual_info` `rfe` |
| `feature_selection_k` | `0.50` `0.75` `1.0` |
| `imbalance_method` | `none` `smote` `borderline_smote` `adasyn` `class_weight` |

sklearn model gene options:

| Gene | Options |
|---|---|
| `n_estimators` | `50` `100` `200` `300` `500` |
| `max_depth` | `2` `3` `4` `5` `6` `8` |
| `learning_rate` | `0.01` `0.05` `0.1` `0.2` |

### Reporting

```yaml
report:
  output_dir: reports
  mlflow_tracking_uri: mlflow_runs   # set to null to disable MLflow
  open_html_on_finish: false
```

---

## Outputs

| Attribute | Description |
|---|---|
| `pipeline.final_score` | Test set score (metric depends on problem type) |
| `pipeline.report_path` | Path to the generated HTML report |
| `pipeline.history` | Full evolution history (fitness curve, diversity, etc.) |
| `pipeline.best_preprocessor` | Fitted PreprocessingPipeline |
| `pipeline.best_model` | Fitted AutoML model |
| `pipeline.predict(df)` | Predictions on new data |
| `pipeline.predict_proba(df)` | Class probabilities (classification only) |

The HTML report includes a generation-by-generation fitness curve, diversity tracking, the best chromosome's gene values, and the final test score.

---

## Preprocessing step order

GAML always applies preprocessing in this fixed order. The GA evolves which option to use at each step, not the order itself.

```
1. NumericImputer         — fill NaN before anything else
2. OutlierHandler         — on clean numeric data, before scaling
3. CorrelationFilter      — reliable stats after imputation
4. CategoricalEncoder     — encode before scaling
5. DistributionTransform  — reduce skewness before scaling
6. Scaler                 — after all columns are numeric
7. MissingIndicator       — binary flags for originally-missing columns
8. FeatureSelector        — on fully preprocessed data
9. ImbalanceHandler       — always last, training data only
```

---

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for module layout, execution flow, and the zero-leakage guarantee.
